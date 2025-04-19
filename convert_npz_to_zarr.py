#!/usr/bin/env python

import numpy as np
import zarr
from pathlib import Path
import os
from tqdm import tqdm
import numcodecs # For Blosc
import argparse

# --- Configuration Defaults ---
# These can be overridden by command-line arguments
DEFAULT_CHUNK_SHAPE = (64, 64, 64)
DEFAULT_COMPRESSOR_CNAME = 'lz4'
DEFAULT_COMPRESSOR_CLEVEL = 5
DEFAULT_COMPRESSOR_SHUFFLE = zarr.Blosc.BITSHUFFLE

# Naming conventions assumed in the NPZ files
IMAGE_SUFFIX = "_image_"
SEG_SUFFIX = "_maskArtifact_"
NPZ_PATTERN = "*.np[yz]"
# --- End Configuration Defaults ---

# Find IMAGE files first
def find_image_npz_files(base_path: Path):
    """Finds files matching the pattern containing the image suffix."""
    return sorted([f for f in base_path.glob(NPZ_PATTERN) if IMAGE_SUFFIX in f.name])

def convert_dataset(source_dir_str: str, target_dir_str: str, chunk_shape: tuple, compressor):
    """Converts paired NPZ files (image & seg) to paired Zarr structures."""
    source_dir = Path(source_dir_str)
    target_dir = Path(target_dir_str)

    if not source_dir.is_dir():
        print(f"Error: Source directory not found: {source_dir}")
        return

    print(f"Converting NPZ from {source_dir} to Zarr in {target_dir}")
    print(f"Using chunks: {chunk_shape}, Compressor: {compressor}")

    conversion_errors = 0
    files_converted = 0
    pairs_processed = 0 # Count pairs

    for split in ["train", "validate", "test"]:
        source_split_path = source_dir / split
        target_split_path = target_dir / split

        if not source_split_path.is_dir():
            print(f"Source split '{split}' not found at {source_split_path}, skipping.")
            continue

        print(f"\nProcessing split: {split}")
        # Find image files first
        image_files = find_image_npz_files(source_split_path)

        if not image_files:
            print("  No image NPZ files found in this split.")
            continue

        os.makedirs(target_split_path, exist_ok=True)
        print(f"  Found {len(image_files)} image files. Looking for pairs and saving to {target_split_path}")

        # Iterate through IMAGE files and find corresponding SEG files
        for image_npz_path in tqdm(image_files, desc=f"Converting {split} pairs", unit="pair"):
            pairs_processed += 1
            try:
                # --- Derive Segmentation Path --- 
                seg_filename_stem = image_npz_path.stem.replace(IMAGE_SUFFIX, SEG_SUFFIX)
                # Find potential matching segmentation NPZ file (could be .npy or .npz)
                possible_seg_paths = list(source_split_path.glob(f"{seg_filename_stem}{NPZ_PATTERN[1:]}"))
                if not possible_seg_paths:
                    print(f"\nWarning: No corresponding segmentation file found for {image_npz_path.name}. Skipping pair.")
                    conversion_errors += 1
                    continue
                seg_npz_path = possible_seg_paths[0] # Assume first match is correct
                # --- End Derive Seg Path --- 

                # --- Define Target Zarr Paths --- 
                target_image_zarr_name = image_npz_path.stem + ".zarr"
                target_image_zarr_path = target_split_path / target_image_zarr_name

                target_seg_zarr_name = seg_npz_path.stem + ".zarr"
                target_seg_zarr_path = target_split_path / target_seg_zarr_name
                # --- End Define Target Paths --- 

                # Skip if BOTH already exist
                if target_image_zarr_path.exists() and target_seg_zarr_path.exists():
                    # print(f"Targets already exist for {image_npz_path.name}, skipping.")
                    continue

                # --- Load Data --- 
                image_data = None
                seg_data = None
                with np.load(str(image_npz_path)) as npz_data:
                    if 'arr_0' not in npz_data:
                        print(f"\nWarning: 'arr_0' not found in image file {image_npz_path}, skipping pair.")
                        conversion_errors += 1
                        continue
                    image_data = npz_data['arr_0']

                with np.load(str(seg_npz_path)) as npz_data:
                    if 'arr_0' not in npz_data:
                        print(f"\nWarning: 'arr_0' not found in seg file {seg_npz_path}, skipping pair.")
                        conversion_errors += 1
                        continue
                    seg_data = npz_data['arr_0']
                # --- End Load Data --- 

                # --- Save Zarr Arrays --- 
                # Save Image
                if not target_image_zarr_path.exists():
                    zarr.save_array(
                        store=str(target_image_zarr_path),
                        arr=image_data,
                        chunks=chunk_shape,
                        compressor=compressor,
                    )
                    files_converted += 1 # Count individual files
                
                # Save Segmentation
                if not target_seg_zarr_path.exists():
                    zarr.save_array(
                        store=str(target_seg_zarr_path),
                        arr=seg_data,
                        chunks=chunk_shape,
                        compressor=compressor,
                    )
                    files_converted += 1 # Count individual files
                # --- End Save Zarr --- 

            except Exception as e:
                print(f"\nError processing pair starting with {image_npz_path.name}: {e}")
                conversion_errors += 1

    print(f"\nConversion finished.")
    print(f"  Pairs processed: {pairs_processed}")
    print(f"  Individual Zarr files created/updated: {files_converted}")
    if conversion_errors > 0:
        print(f"  Errors encountered (pairs skipped or failed): {conversion_errors}")

def main():
    parser = argparse.ArgumentParser(description="Convert a dataset of NPZ files (arr_0) to Zarr format.")
    parser.add_argument("source_dir", type=str, help="Root directory containing train/validate subdirs with NPZ files.")
    parser.add_argument("target_dir", type=str, help="Root directory where the Zarr dataset structure will be created.")
    parser.add_argument("--chunks", type=int, nargs=3, default=list(DEFAULT_CHUNK_SHAPE),
                        help=f"Chunk shape (D H W) for Zarr arrays. Default: {DEFAULT_CHUNK_SHAPE}")
    parser.add_argument("--clevel", type=int, default=DEFAULT_COMPRESSOR_CLEVEL,
                        help=f"Blosc compression level (0-9). Default: {DEFAULT_COMPRESSOR_CLEVEL}")
    parser.add_argument("--cname", type=str, default=DEFAULT_COMPRESSOR_CNAME,
                        choices=['zlib', 'blosc', 'lzma', 'zstd', 'lz4'],
                        help=f"Blosc compressor name. Default: {DEFAULT_COMPRESSOR_CNAME}")
    parser.add_argument("--shuffle", type=str, default="bitshuffle", choices=["noshuffle", "byte", "bitshuffle"],
                        help=f"Blosc shuffle type. Default: bitshuffle")

    args = parser.parse_args()

    # Map shuffle string argument to Blosc constant
    shuffle_map = {
        "noshuffle": zarr.Blosc.NOSHUFFLE,
        "byte": zarr.Blosc.SHUFFLE,
        "bitshuffle": zarr.Blosc.BITSHUFFLE
    }
    shuffle_val = shuffle_map[args.shuffle]

    # Create compressor object
    compressor = zarr.Blosc(cname=args.cname, clevel=args.clevel, shuffle=shuffle_val)

    convert_dataset(args.source_dir, args.target_dir, tuple(args.chunks), compressor)

if __name__ == "__main__":
    main() 