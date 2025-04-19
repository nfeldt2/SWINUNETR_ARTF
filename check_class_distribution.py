import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import sys
from tqdm import tqdm
import time
import os

# --- MONAI Imports (New) ---
from monai.transforms import (
    Compose as MonaiCompose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped
)
from monai.data.image_reader import ImageReader, ITKReader # Base and default reader
# --- End MONAI Imports ---

# --- Typing Import (New) ---
from typing import Sequence, Union
# --- End Typing Import ---

# --- Batchgenerators Imports ---
from batchgenerators.dataloading.data_loader import DataLoader # Base class
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose as BgCompose, AbstractTransform
# --- End Batchgenerators Imports ---

# --- Import augmentations from Augmentations.py ---
from Augmentations import get_augmentations
# --- End Import ---

# --- Add Zarr import ---
import zarr

# --- Custom Zarr Reader for MONAI (New) ---
class ZarrReader(ImageReader):
    """MONAI ImageReader for loading arrays from Zarr stores (assumes 'arr_0' structure indirectly)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MONAI LoadImaged expects reader to handle the 'meta_keys_postfix' etc.
        # We'll keep it simple and assume LoadImaged gives us the path directly.
        self.img = None # Initialize instance variables
        self.meta = None

    def read(self, data, **kwargs):
        """Reads a Zarr array from the given path and stores it."""
        # 'data' is expected to be the path string or Path object from LoadImaged
        # Handle the case where it might be passed as a single-element tuple
        path_input = data
        if isinstance(data, tuple) and len(data) == 1:
            path_input = data[0]

        if isinstance(path_input, Path):
            path = str(path_input)
        elif isinstance(path_input, str):
            path = path_input
        else:
            # Log the original received data for debugging
            print(f"DEBUG: ZarrReader.read received unexpected type: {type(data)}, value: {data}")
            raise TypeError(f"ZarrReader expected a path string, Path object, or single-element tuple, got {type(data)}")

        try:
            # Load the Zarr array directly using the string path
            arr = zarr.load(path)
            # Minimal metadata - MONAI needs affine, usually derived from headers
            # For Zarr, we don't have standard headers. Provide identity affine.
            meta = {
                "affine": np.eye(4), # Simple identity affine
                "original_affine": np.eye(4),
                "spatial_shape": np.array(arr.shape),
            }
            # Store the loaded data and metadata
            self.img = arr
            self.meta = meta
            # Return the array as read expects, get_data will retrieve them
            return arr
        except Exception as e:
            raise RuntimeError(f"Error loading Zarr store at {path}: {e}") from e

    def get_data(self, img):
        """Returns the loaded image array and metadata.
           The 'img' argument is required by MONAI's API but not used here
           as data is stored during the read() call."""
        # Return the stored image and metadata
        return self.img, self.meta

    def get_supported_modalities(self):
        # Indicate support for general modalities
        return {""}

    def verify_suffix(self, filename: Union[str, Sequence[str], Path, Sequence[Path]]) -> bool:
        """Check if the filename ends with .zarr (directory)."""
        # MONAI's LoadImaged calls this. It expects a filename string.
        # Handle list/single path, check if it's a directory ending in .zarr
        if isinstance(filename, (list, tuple)):
            filename = filename[0] # Check the first filename if list provided
        path = Path(filename)
        return path.is_dir() and path.name.lower().endswith(".zarr")
# --- End Custom Zarr Reader ---

# --- Helper function for Center Cropping (New) ---
def center_crop_np(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Performs a center crop on a NumPy array (assuming CDHW or DHW).

    Args:
        img: Input NumPy array, expects shape (C, D, H, W) or (D, H, W).
        target_shape: Tuple representing the target spatial shape (D, H, W).

    Returns:
        The center-cropped NumPy array.
    """
    current_shape = img.shape
    has_channel = len(current_shape) == 4
    spatial_dims = current_shape[1:] if has_channel else current_shape

    if len(target_shape) != 3:
        raise ValueError(f"Target shape must be 3D (D, H, W), got {target_shape}")

    if any(t > s for t, s in zip(target_shape, spatial_dims)):
        raise ValueError(f"Target shape {target_shape} is larger than image shape {spatial_dims}")

    start_indices = [(s - t) // 2 for s, t in zip(spatial_dims, target_shape)]
    end_indices = [start + t for start, t in zip(start_indices, target_shape)]

    if has_channel:
        # Assumes single channel (C=1)
        cropped_img = img[:, start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
    else:
        cropped_img = img[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]

    return cropped_img
# --- End Helper Function ---

# --- Custom Transform to load only seg arr_0 (Batchgenerators compatible) ---
# --- MODIFIED to use MONAI transforms for resizing/spacing BEFORE stacking ---
class LoadSegArr0dZarr(AbstractTransform):
    """Loads segmentation array from a batch of Zarr array paths into 'seg' key,
       applies basic MONAI transforms (Load, EnsureChannel), finds the minimum 
       spatial dimensions in the batch, center-crops all arrays to that minimum,
       and adds dummy 'data' key."""
    def __init__(self, monai_transforms, data_key="data", seg_key="seg", seg_path_key="seg_path"):
        # Stores the MONAI Compose object passed during initialization
        self.monai_transforms = monai_transforms
        self.data_key = data_key
        self.seg_key = seg_key
        self.seg_path_key = seg_path_key

    def __call__(self, **data_dict):
        t_start_load = time.monotonic()
        # Expects data_dict['seg_path'] to be a list of Path objects
        seg_paths = data_dict.get(self.seg_path_key)
        if seg_paths is None or not isinstance(seg_paths, list):
            raise ValueError(f"LoadSegArr0dZarr expects '{self.seg_path_key}' to be a list of Paths.")

        loaded_seg_arrays_uncropped = []
        error_paths = []

        # --- 1. Load all arrays using MONAI (without resizing/spacing) --- 
        for i, seg_zarr_path in enumerate(seg_paths):
            try:
                item_dict = {self.seg_key: seg_zarr_path}
                transformed_item = self.monai_transforms(item_dict)
                seg_tensor = transformed_item[self.seg_key] # Should be Tensor (C, D, H, W)
                seg_data_np = seg_tensor.numpy() # Shape (C, D, H, W)
                loaded_seg_arrays_uncropped.append(seg_data_np)
            except Exception as e:
                error_paths.append(str(seg_zarr_path))
                print(f"ERROR: Failed loading/initial transforming Zarr {seg_zarr_path}: {e}")
                raise RuntimeError(f"Error loading/initial transforming Zarr {seg_zarr_path}: {e}") from e

        if not loaded_seg_arrays_uncropped: # Handle empty batch case
            # Return empty structures if loading failed for all
            print("Warning: LoadSegArr0dZarr produced an empty batch after loading phase.")
            # Assuming C=1, D=H=W=0 for empty placeholder
            empty_shape = (0, 1, 0, 0, 0) # (B, C, D, H, W)
            data_dict[self.seg_key] = np.empty(empty_shape, dtype=np.int16)
            data_dict[self.data_key] = np.empty(empty_shape, dtype=np.float32)
            return data_dict

        # --- 2. Find minimum spatial dimensions across the batch --- 
        # Arrays have shape (C, D, H, W), C should be 1
        min_d = min(arr.shape[1] for arr in loaded_seg_arrays_uncropped)
        min_h = min(arr.shape[2] for arr in loaded_seg_arrays_uncropped)
        min_w = min(arr.shape[3] for arr in loaded_seg_arrays_uncropped)
        min_spatial_shape = (min_d, min_h, min_w)
        # print(f"DEBUG: Batch min spatial shape: {min_spatial_shape}") # Optional debug

        # --- 3. Center crop each array to the minimum dimensions --- 
        loaded_seg_arrays_cropped = []
        loaded_data_arrays_cropped = []
        final_cropped_shape = (1,) + min_spatial_shape # Add channel dim back (C, D, H, W)

        for seg_array_uncropped in loaded_seg_arrays_uncropped:
            try:
                # Crop using the helper function
                cropped_seg = center_crop_np(seg_array_uncropped, min_spatial_shape)
                # Ensure shape is correct (C, D, H, W)
                if cropped_seg.shape != final_cropped_shape:
                     # This might happen if helper returned DHW, add channel dim
                     if cropped_seg.shape == min_spatial_shape and final_cropped_shape[0] == 1:
                         cropped_seg = np.expand_dims(cropped_seg, axis=0)
                     else:
                         raise ValueError(f"Cropped shape {cropped_seg.shape} mismatch expected {final_cropped_shape}")
                
                loaded_seg_arrays_cropped.append(cropped_seg)
                loaded_data_arrays_cropped.append(np.zeros_like(cropped_seg, dtype=np.float32))
            except Exception as crop_e:
                 print(f"ERROR: Failed center cropping array with shape {seg_array_uncropped.shape} to target {min_spatial_shape}: {crop_e}")
                 # Decide how to handle: skip this item or raise error? Raising for now.
                 raise RuntimeError(f"Error during center cropping: {crop_e}") from crop_e


        # --- 4. Stack the CROPPED arrays --- 
        try:
            # Input shapes are (C, D, H, W), stacked shape becomes (B, C, D, H, W)
            data_dict[self.seg_key] = np.stack(loaded_seg_arrays_cropped, axis=0)
            data_dict[self.data_key] = np.stack(loaded_data_arrays_cropped, axis=0)
        except ValueError as e:
            # This error should be prevented by the cropping step
            shapes = [arr.shape for arr in loaded_seg_arrays_cropped]
            print(f"ERROR: Stacking failed even after cropping! Shapes: {shapes}. Error: {e}")
            raise ValueError(f"Stacking failed after cropping. Shapes: {shapes}. Error: {e}") from e

        t_end_load = time.monotonic()
        pid = os.getpid()
        # print(f"[LOG Loader {pid}] LoadSegArr0dZarr (with MONAI) took {t_end_load - t_start_load:.4f}s for batch size {len(seg_paths)}")

        return data_dict

# --- Custom Transform to add dimensions --- Required before Augmentations.py transforms?
# This now needs to handle 4D input (B, D, H, W) -> 5D (B, 1, D, H, W)
class AddChannelBatchDimension(AbstractTransform):
    """Adds channel dimension if needed, expecting (B, D, H, W) or (B, 1, D, H, W).
       Operates on both 'data' and 'seg' keys."""
    def __init__(self, data_key="data", seg_key="seg"):
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data_dict):
        t_start_add_dim = time.monotonic()
        pid = os.getpid()
        processed_keys = []
        # --- No longer needed if MONAI EnsureChannelFirstd is used before stacking ---
        # --- Keep the check for safety but it should pass through ---
        for key in [self.data_key, self.seg_key]:
            if key in data_dict:
                arr = data_dict[key]
                # Expect 5D input (B, C, D, H, W) after MONAI + stacking
                if arr.ndim == 5:
                     if arr.shape[1] != 1: # Check channel dim
                         print(f"WARN: AddChannelBatchDimension found {arr.shape[1]} channels for key '{key}'. Expected 1. Shape: {arr.shape}")
                     pass # Already 5D, likely C=1
                elif arr.ndim == 4:
                    # This case should ideally not happen if EnsureChannelFirstd worked
                    print(f"WARN: AddChannelBatchDimension received 4D input {arr.shape} for key '{key}'. Adding channel dim.")
                    arr = np.expand_dims(arr, axis=1)
                    data_dict[key] = arr
                else:
                    raise ValueError(f"AddChannelBatchDimension expected 5D array (B, C, D, H, W) or 4D for key '{key}', got {arr.ndim}D shape {arr.shape}")
                processed_keys.append(key)
        t_end_add_dim = time.monotonic()
        # print(f"[LOG AddDim {pid}] AddChannelBatchDimension took {t_end_add_dim - t_start_add_dim:.4f}s for keys {processed_keys}")
        return data_dict

# --- Updated get_image_label (operates on the final transformed segmentation array) ---
def get_image_label(segmentation_mask_data: np.ndarray, min_pixels_threshold: int = 40) -> int:
    """
    Derives a single image-level classification label from a segmentation mask,
    requiring a minimum number of artifact pixels.
    Returns: 0 = No Artifact, 1 = Artifact Type 1, 2 = Artifact Type 2
    Optimized to use np.any for potentially faster checks.
    """
    # Check for Type 1 first (highest priority)
    if np.any(segmentation_mask_data == 1):
        if np.sum(segmentation_mask_data == 1) >= min_pixels_threshold:
            return 1 # Return label for Artifact Type 1

    # Check for Type 2 if Type 1 not found
    if np.any(segmentation_mask_data == 2):
        if np.sum(segmentation_mask_data == 2) >= min_pixels_threshold:
            return 2 # Return label for Artifact Type 2

    # If neither artifact type is present
    num_class_1 = np.sum(segmentation_mask_data == 1)
    num_class_2 = np.sum(segmentation_mask_data == 2)
    print(f"DEBUG: No artifact type found in segmentation mask. Returning label 0. Num class 1: {num_class_1}, Num class 2: {num_class_2}")
    return 0 # Return label for No Artifact

# --- Batchgenerators Basic Dataset ---
class NpzFileListLoader(DataLoader):
    """
    Basic DataLoader for batchgenerators. Takes a list of dicts [{"seg_path": path}, ...],
    shuffles them, and returns individual dicts for the MultiThreadedAugmenter.
    """
    def __init__(self, data, batch_size, num_threads_in_multithreaded=1):
        # data is expected to be a list of dicts, e.g., [{"seg_path": "/path/to/seg1.npz"}, ...]
        super().__init__(data, batch_size, num_threads_in_multithreaded)
        self.indices = list(range(len(self._data))) # Indices for shuffling/iteration

    def generate_train_batch(self):
        # Called by MultiThreadedAugmenter to get the next item to process.
        # We operate in 'inference' mode, so just return one item dict at a time.
        # MultiThreadedAugmenter handles the parallel fetching.

        # Determine number of items to fetch for the batch
        current_batch_size = min(self.batch_size, len(self.indices))
        if current_batch_size == 0:
            # Reset indices if epoch ends
            self.indices = list(range(len(self._data)))
            # Optional: Shuffle if needed
            np.random.shuffle(self.indices)
            current_batch_size = min(self.batch_size, len(self.indices))
            if current_batch_size == 0: # Still zero means no data
                raise StopIteration # Signal end of data

        # Pop indices for the batch
        batch_indices = [self.indices.pop(0) for _ in range(current_batch_size)]
        # Get item dictionaries for the batch
        items = [self._data[i] for i in batch_indices]

        # Collate the dictionaries manually into a single batch dictionary
        batch_dict = {key: [] for key in items[0].keys()} # Initialize with keys from first item
        batch_dict['roi'] = [] # Ensure ROI key exists

        for item in items:
            for key, value in item.items():
                batch_dict[key].append(value)
            # Add the required 'roi' key for each item in the batch
            batch_dict['roi'].append(None)

        return batch_dict # Return e.g. {"seg_path": [p1, p2..], "roi": [None, None..]}

# --- NEW: Function using MultiThreadedAugmenter ---
def calculate_distribution_batched(
    file_dicts, # List of {"seg_path": path}
    batch_size, # Now used by the loader
    num_workers,
    transform_pipeline,
    min_pixels_threshold,
    limit=None
):
    """Calculates class distribution using MultiThreadedAugmenter."""
    label_counts = Counter()
    errors = 0

    # Apply limit if provided
    if limit is not None and limit > 0:
        print(f"  Processing a limited subset of {limit} files for timing.")
        file_dicts_to_process = file_dicts[:limit]
    else:
        file_dicts_to_process = file_dicts
        limit = len(file_dicts) # Use full length for reporting if no limit

    total_files = len(file_dicts_to_process)

    if total_files == 0:
        print("  No files to process.")
        return label_counts, errors, 0.0, 0.0

    print(f"  Initializing batchgenerators DataLoader for {total_files} files...")
    # Pass the actual batch_size to the loader
    base_dataloader = NpzFileListLoader(file_dicts_to_process, batch_size=batch_size, num_threads_in_multithreaded=num_workers)

    print(f"  Initializing MultiThreadedAugmenter with {num_workers} workers...")

    # Instantiate MultiThreadedAugmenter, setting useroi=False
    multi_threaded_augmenter = MultiThreadedAugmenter(
        data_loader=base_dataloader,
        transform=transform_pipeline,
        num_processes=num_workers,
        num_cached_per_queue=2,
        seeds=None,
        pin_memory=False,
        useroi=False # Set to False to avoid internal ROI calculation/KeyError
    )

    print(f"  Processing {total_files} files using MultiThreadedAugmenter...")
    start_time = time.monotonic()

    processed_count = 0
    # Iterate through the augmenter. It yields batches processed by workers.
    # Since base_dataloader yields single items and MTA probably aggregates based on its internal logic,
    # check the structure of the yielded 'batch'. It might be a dict containing keys like 'data', 'seg_path'.
    # Batch size in MTA is implicitly handled by how many items it processes.
    pbar = tqdm(total=total_files, desc="Processing files (batchgenerators)", unit="file")
    try:
        batch_num = 0
        for batch in multi_threaded_augmenter: # Each 'batch' contains results for batch_size file paths
            t_start_batch_process = time.monotonic()
            batch_num += 1
            # Expected batch structure: {'data': (B,C,D,H,W), 'seg': (B,C,D,H,W), '_seg_path_str': [p1,..], ...}

            # Determine the actual number of items in this batch (might be < batch_size at the end)
            current_batch_actual_size = batch['seg'].shape[0]
            seg_paths_in_batch = batch.get('_seg_path_str', ["Unknown Path"] * current_batch_actual_size)
            item_process_times = []

            # Iterate through items *within* the batch
            for i in range(current_batch_actual_size):
                t_start_item_process = time.monotonic()
                label = -1 # Default label in case of error before assignment
                try:
                    # Extract the i-th segmentation mask from the batch
                    seg_data_np = batch['seg'][i] # Shape (C, D, H, W)

                    # Remove channel dim if it's singular
                    if seg_data_np.ndim == 4 and seg_data_np.shape[0] == 1:
                        seg_data_np = seg_data_np.squeeze(0)
                    # Now seg_data_np should be 3D (D, H, W)

                    # --- Remove DEBUG PRINT --- 
                    # if processed_count < args.batch_size * 2: # Print for first ~2 batches
                    #     print(f"DEBUG (Item {processed_count}): Unique values in seg_data_np: {np.unique(seg_data_np)}")
                    # --- END DEBUG REMOVAL --- 

                    t_start_get_label = time.monotonic()
                    label = get_image_label(seg_data_np, min_pixels_threshold=min_pixels_threshold)
                    t_end_get_label = time.monotonic()
                    label_counts[label] += 1
                    processed_count += 1
                    item_process_times.append((t_end_get_label - t_start_get_label)) # Log get_image_label time

                except Exception as e:
                    # Catch errors during label extraction for a specific item in the batch
                    error_path = seg_paths_in_batch[i]
                    print(f"Warning: Error processing item {i} from batch (path: {error_path}): {e}", file=sys.stderr)
                    errors += 1
                finally:
                     # Update progress bar for each individual item processed or errored
                     pbar.update(1)

            t_end_batch_process = time.monotonic()
            avg_label_time = np.mean(item_process_times) if item_process_times else 0
            print(f"[LOG MainLoop] Batch {batch_num} (size {current_batch_actual_size}) processing took {t_end_batch_process - t_start_batch_process:.4f}s. Avg get_image_label time: {avg_label_time:.6f}s")

            # Check if we have processed the limited number of files after finishing a batch
            if total_files > 0 and (processed_count + errors) >= total_files:
                 print(f"\n  Reached limit ({total_files}), breaking processing loop.")
                 # Break outer loop (batch iteration)
                 break

    except Exception as e:
        # Catch potential errors from the augmenter itself (e.g., worker crashes)
        print(f"ERROR: MultiThreadedAugmenter failed: {e}", file=sys.stderr)
        # Potentially re-raise or handle differently depending on requirements
        # errors += (total_files - processed_count - errors) # Count remaining as errors?
    finally:
        pbar.close()
        print("  Shutting down MultiThreadedAugmenter...")
        # Correctly finish the augmenter processes
        multi_threaded_augmenter._finish()
        print("  Augmenter shutdown complete.")

    end_time = time.monotonic()
    total_time = end_time - start_time

    # Use 'total_files' (the number submitted) for performance calculation
    time_per_sample = total_time / total_files if total_files > 0 else 0
    samples_per_second = total_files / total_time if total_time > 0 else float('inf')

    print("\n  Aggregating results...") # Already done in the loop

    if errors > 0:
        print(f"  Encountered {errors} errors during processing.")

    # Return counts, errors, and performance metrics
    return label_counts, errors, time_per_sample, samples_per_second


def print_counts(title, counts, total_samples):
    """Prints formatted class counts and percentages."""
    print(f" {title} Distribution ({total_samples} samples):")
    if total_samples == 0:
        print("   No samples.")
        return

    total_foreground = counts.get(1, 0) + counts.get(2, 0)

    for label in sorted(counts.keys()):
        count = counts[label]
        percentage = (count / total_samples) * 100
        if label > 0 and total_foreground > 0:
             fg_percentage = (count / total_foreground) * 100
             print(f"  Class {label}: {count:>6} ({percentage:6.2f}%) - Foreground Share: {fg_percentage:.2f}%")
        else:
             print(f"  Class {label}: {count:>6} ({percentage:6.2f}%)")


def main(args):
    root_path = Path(args.dataset_dir)
    if not root_path.is_dir():
        print(f"Error: Dataset directory not found: {root_path}")
        sys.exit(1)

    # Use the custom Zarr reader
    zarr_reader = ZarrReader()

    # --- Define MONAI Transform Pipeline for Loading/Preprocessing --- 
    # Keys match the temporary key used in LoadSegArr0dZarr
    seg_key = "seg"
    monai_preprocessing_transforms = MonaiCompose([
        LoadImaged(keys=[seg_key], reader=zarr_reader, image_only=False),
        EnsureChannelFirstd(keys=[seg_key], channel_dim="no_channel"),
        # Removed Spacingd and Resized as requested
        # Spacingd(keys=[seg_key], pixdim=args.target_spacing, mode="nearest"),
        # Resized(keys=[seg_key], spatial_size=args.target_size, mode="nearest"),
        EnsureTyped(keys=[seg_key], dtype=np.int16) # Use int16 for masks, adjust if needed
    ])
    # --- End MONAI Transform Definition ---

    # --- Define Batchgenerators Transform Pipeline ---
    print("Defining transform pipeline using Augmentations.py...")
    # 1. Load Zarr and apply MONAI preprocessing (spacing, resizing)
    # Pass the MONAI pipeline to the custom loader
    load_transform = LoadSegArr0dZarr(monai_transforms=monai_preprocessing_transforms, seg_key=seg_key)

    # 2. Add channel dimension if needed (should already be done by MONAI)
    add_dim_transform = AddChannelBatchDimension(seg_key=seg_key)

    # 3. Get spatial augmentations from Augmentations.py
    # Note: These expect keys 'data' and 'seg'. LoadSegArr0dZarr creates dummy 'data'.
    spatial_augs = get_augmentations() # Assuming this returns the spatial part
    spatial_augs_list = [] # Initialize list to hold transforms
    if spatial_augs is None:
        print("Warning: get_augmentations returned None. No spatial augmentations will be applied.")
        # spatial_augs_list remains empty
    elif isinstance(spatial_augs, BgCompose): # Correctly check for batchgenerators Compose
        print(f"DEBUG: get_augmentations returned BgCompose. Extracting transforms.")
        spatial_augs_list = list(spatial_augs.transforms)
    elif isinstance(spatial_augs, (list, tuple)):
        print(f"DEBUG: get_augmentations returned list/tuple.")
        spatial_augs_list = list(spatial_augs)
    elif isinstance(spatial_augs, AbstractTransform): # Handle case where it might return a single transform
        print(f"DEBUG: get_augmentations returned a single transform.")
        spatial_augs_list = [spatial_augs]
    else:
        # This case should be less likely now
        print(f"Warning: get_augmentations returned unexpected type {type(spatial_augs)}. No spatial augmentations applied.")
        # spatial_augs_list remains empty

    # Combine batchgenerators transforms
    # Order: Load+Preprocess -> AddDim (safety) -> Spatial Augs
    batchgenerators_pipeline = BgCompose([ 
        load_transform,
        add_dim_transform, 
        *spatial_augs_list # Unpack the processed list
    ])
    # --- End Batchgenerators Transform Pipeline ---


    print(f"Searching for Zarr segmentation stores ({args.seg_suffix}.zarr) in {root_path}...")
    all_files = []
    for split in ["train", "validate", "test"]:
        split_path = root_path / split
        if split_path.is_dir():
            print(f" Searching in {split}...")
            # Adjusted glob pattern to find directories ending with the suffix
            pattern = f"*{args.seg_suffix}.zarr"
            split_files = sorted([p for p in split_path.glob(pattern) if p.is_dir()])
            print(f"  Found {len(split_files)} matching stores in {split}.")
            all_files.extend([{"seg_path": f} for f in split_files])
        else:
            print(f" Directory {split} not found, skipping.")

    if not all_files:
        print("Error: No segmentation Zarr stores found. Please check the dataset directory and suffix.")
        sys.exit(1)

    print(f"\nProcessing combined dataset of {len(all_files)} Zarr samples using batchgenerators.")

    # --- Calculate Distribution --- 
    start_calc_time = time.time()
    counts, errors, total_time, rate = calculate_distribution_batched(
        all_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_pipeline=batchgenerators_pipeline, # Pass the combined pipeline
        min_pixels_threshold=args.min_artifact_pixels,
        limit=args.limit
    )
    end_calc_time = time.time()
    # --- End Calculation ---

    print("\n  Aggregating results...")
    total_processed = sum(counts.values())
    print_counts("Combined Dataset Class Distribution (batchgenerators)", counts, total_processed)

    if errors > 0:
        print(f"\nEncountered {errors} errors during processing.")

    if total_processed > 0:
        print("-" * 40)
        actual_files_processed = total_processed + errors # Files that were attempted
        limit_str = f" (based on {args.limit} submitted files)" if args.limit else ""
        print(f"Performance (batchgenerators){limit_str}:")
        print(f"  Total processing time: {end_calc_time - start_calc_time:.2f} seconds")
        if actual_files_processed > 0:
            avg_time = (end_calc_time - start_calc_time) / actual_files_processed
            print(f"  Avg. time per sample: {avg_time:.4f} seconds")
            print(f"  Samples per second: {1.0 / avg_time:.2f}")
        print("-" * 40)
    else:
        print("No files were successfully processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate class distribution from segmentation Zarr files using batchgenerators.")
    parser.add_argument("dataset_dir", type=str, help="Root directory containing train/validate/test subdirs with Zarr stores.")
    parser.add_argument("--seg_suffix", type=str, default="_maskArtifact", help="Suffix identifying segmentation Zarr directories (e.g., '_maskArtifact' for files like case1_maskArtifact.zarr).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for batchgenerators.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for loading file paths (passed to internal loader).")
    parser.add_argument("--min_artifact_pixels", type=int, default=10, help="Minimum number of artifact pixels (1 or 2) required in the *transformed* mask to assign label 1 or 2.")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N files for testing/timing.")

    # --- NEW Arguments for MONAI transforms ---
    # parser.add_argument('--target_size', type=int, nargs=3, required=True, help='Target spatial size (D, H, W) for Resized transform.')
    # parser.add_argument('--target_spacing', type=float, nargs=3, required=True, help='Target voxel spacing (X, Y, Z) for Spacingd transform.')
    # --- End NEW Arguments ---


    args = parser.parse_args()
    main(args) 