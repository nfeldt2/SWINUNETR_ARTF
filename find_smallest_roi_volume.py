import argparse
import numpy as np
from pathlib import Path
import sys
import math
import re # Import regex module
import concurrent.futures # Import concurrent.futures
from functools import partial # Import partial for ProcessPoolExecutor map
from typing import Union # Import Union for older Python versions

# Define the strict pattern using regex
# Allow prefix, matches 'maskArtifactROI_', 4 groups of (digit underscore), ends with '.npz'
STRICT_PATTERN_RE = re.compile(r"maskArtifactROI_(\d+)_(\d+)_(\d+)_(\d+)\.npz$")

def process_file(file_path: Path) -> Union[dict, None]:
    """
    Processes a single NPZ file to find the ROI bounding box shape and volume.
    Returns a dictionary with shape, volume, and path, or None if invalid.
    """
    try:
        with np.load(file_path) as data:
            if 'arr_0' not in data:
                print(f"Warning: Key 'arr_0' not found in {file_path}. Skipping.", file=sys.stderr)
                return None # Indicate error/skip
            volume = data['arr_0']

        coords = np.argwhere(volume == 1)

        if coords.size == 0:
            # print(f"Info: No '1's found in {file_path}. Skipping.")
            return None # Indicate skip (no ROI)

        min_indices = coords.min(axis=0)
        max_indices = coords.max(axis=0)
        current_shape = tuple(max_indices - min_indices + 1)
        current_volume = np.prod(current_shape)

        return {
            "shape": current_shape,
            "volume": current_volume,
            "path": file_path
        }

    except FileNotFoundError:
        print(f"Warning: File not found during processing: {file_path}. Skipping.", file=sys.stderr)
        return None # Indicate error/skip
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}. Skipping.", file=sys.stderr)
        return None # Indicate error/skip


def find_smallest_roi_volume(directory: Path, num_workers: Union[int, None]):
    """
    Finds the smallest bounding box shape containing '1's across all strictly matching
    maskArtifactROI_#_#_#_#.npz files in a directory, using parallel processing.
    """
    min_shape = (math.inf, math.inf, math.inf)
    min_volume = math.inf
    min_file = None
    processed_files_count = 0
    error_or_skipped_files_count = 0
    skipped_no_roi_count = 0 # Count files skipped specifically due to no ROI found

    # Broad pattern to find potential candidates
    broad_pattern = "*maskArtifactROI*.npz"
    print(f"Searching for files matching broad pattern '{broad_pattern}' in '{directory}'...")

    potential_files = list(directory.glob(broad_pattern))
    if not potential_files:
        print(f"Error: No files found matching the broad pattern '{broad_pattern}' in {directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(potential_files)} potential files. Filtering for strict pattern 'maskArtifactROI_#_#_#_#.npz'...")

    # Filter using the strict regex pattern
    # Use re.search to find the pattern anywhere, but ensure it ends with .npz
    strict_matching_files = [f for f in potential_files if STRICT_PATTERN_RE.search(f.name)]

    if not strict_matching_files:
        print(f"Error: No files found matching the strict pattern 'maskArtifactROI_#_#_#_#.npz' after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(strict_matching_files)} strictly matching files. Processing in parallel using up to {num_workers or 'available'} workers...")

    results = []
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the process_file function to the list of files
        # Use list() to ensure all futures are completed before proceeding
        results = list(executor.map(process_file, strict_matching_files))

    # Process results
    found_valid_roi = False
    for result in results:
        if result is None:
            # Could distinguish between errors and no ROI later if needed
            error_or_skipped_files_count += 1
            continue # Skip None results (errors or files without ROI)

        # Check if this is the first valid ROI found or if it's smaller
        if not found_valid_roi or result["volume"] < min_volume:
            min_shape = result["shape"]
            min_volume = result["volume"]
            min_file = result["path"]
            found_valid_roi = True
        processed_files_count += 1 # Count files that were successfully processed and had an ROI

    # Recalculate skipped_no_roi_count based on how many Nones were returned vs total strict files
    total_files_processed_or_skipped = len(results)
    skipped_no_roi_or_error_count = total_files_processed_or_skipped - processed_files_count
    # We can't perfectly distinguish skipped vs error from None, so group them
    
    print(f"\nProcessed {processed_files_count} files containing valid ROIs.")
    if skipped_no_roi_or_error_count > 0:
         print(f"Skipped or encountered errors in {skipped_no_roi_or_error_count} files (includes files with no ROI).")
    
    if not found_valid_roi:
        print("\nError: No valid ROIs (containing '1's) found in any successfully processed file matching the strict pattern.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nSmallest ROI bounding box shape found: {min_shape}")
        print(f"Corresponding volume (number of voxels): {min_volume}")
        print(f"File containing this smallest ROI: {min_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the smallest ROI bounding box volume in NPZ files.")
    parser.add_argument("directory", type=str, help="Directory containing the maskArtifactROI NPZ files.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (defaults to available cores).")
    args = parser.parse_args()

    target_dir = Path(args.directory)
    if not target_dir.is_dir():
        print(f"Error: Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    find_smallest_roi_volume(target_dir, args.workers) 