import argparse
import os
import time
import shutil
from pathlib import Path
import warnings
import pickle
import random
import math
import concurrent.futures # Import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as PyTorchDataLoader # Alias PyTorch DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

import monai
from monai.data import Dataset, list_data_collate
from monai.transforms.transform import MapTransform
from monai.transforms import (
    Compose, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, RandRotate90d, RandGaussianNoised, EnsureTyped,
    ScaleIntensityRanged, RandCropByPosNegLabeld
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
# import monai focal loss
# from monai.losses import FocalLoss # <-- Comment out FocalLoss import

from Augmentations import get_augmentations

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

# --- WandB Import ---
try:
    import wandb
except ImportError:
    print("wandb not installed, pip install wandb to enable logging.")
    wandb = None
# --- End WandB Import ---

warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", message=".*weights_only=False.*") # Suppress torch.load warning

# --- Updated get_image_label with threshold ---
def get_image_label(segmentation_mask_data: np.ndarray, min_pixels_threshold: int = 100) -> int:
    """
    Derives a single image-level classification label from a segmentation mask,
    requiring a minimum number of artifact pixels.

    Args:
        segmentation_mask_data: NumPy array of the *transformed* segmentation mask.
                                 Assumes labels 1 and 2 represent artifact types.
        min_pixels_threshold: Minimum number of voxels required for class 1 or 2.

    Returns:
        0 = No Artifact, 1 = Artifact Type 1, 2 = Artifact Type 2
    """
    unique_labels = np.unique(segmentation_mask_data)

    # Check for class 1 first (priority)
    if 1 in unique_labels:
        count1 = np.count_nonzero(segmentation_mask_data == 1)
        if count1 >= min_pixels_threshold:
            return 1
    # Check for class 2 if class 1 wasn't dominant enough
    elif 2 in unique_labels:
        count2 = np.count_nonzero(segmentation_mask_data == 2)
        if count2 >= min_pixels_threshold:
            return 2

    return 0


# --- Custom Transform for NPZ Loading ---
class LoadPairedArr0d(MapTransform):
    """
    Custom dictionary transform to load image and segmentation data from
    separate NPZ files specified by 'image_path' and 'seg_path' keys.
    It assumes the relevant data in both files is stored under the key 'arr_0'.
    Outputs the loaded arrays under the keys 'image' and 'seg'.
    """
    def __init__(self, keys=("image_path", "seg_path"), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img_path = d.get("image_path")
        seg_path = d.get("seg_path")

        if img_path is None or seg_path is None:
            raise KeyError("Input dictionary must contain 'image_path' and 'seg_path' keys.")

        try:
            img_npz = np.load(img_path)
            if 'arr_0' not in img_npz:
                raise KeyError(f"Key 'arr_0' not found in image npz file: {img_path}. Available keys: {list(img_npz.keys())}")
            d["image"] = img_npz['arr_0']
            img_npz.close()

            seg_npz = np.load(seg_path)
            if 'arr_0' not in seg_npz:
                raise KeyError(f"Key 'arr_0' not found in seg npz file: {seg_path}. Available keys: {list(seg_npz.keys())}")
            d["seg"] = seg_npz['arr_0']
            seg_npz.close()

            # Remove original path keys after successful loading
            del d["image_path"]
            del d["seg_path"]

        except Exception as e:
            print(f"Error loading paired NPZ files ({img_path}, {seg_path}): {e}")
            raise e

        return d
# --- End Custom Transform ---

# --- Custom Transform to get Label from Segmentation ---
class GetLabelFromSegd(MapTransform):
    """
    Calculates the image-level label from the segmentation mask after initial transforms.
    Stores the label under the specified `label_key`.
    Removes the original segmentation mask key.
    """
    def __init__(self, keys: str, label_key: str = 'label', min_pixels_threshold: int = 100, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key
        self.min_pixels_threshold = min_pixels_threshold

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            seg_tensor = d[key]
            # Ensure it's numpy for get_image_label
            seg_np = seg_tensor.cpu().numpy() if isinstance(seg_tensor, torch.Tensor) else np.asarray(seg_tensor)

            image_label = 0 # Default to 0
            if seg_np.ndim > 0 and seg_np.shape[0] > 0:
                 if seg_np.shape[0] == 1:
                     seg_for_label = seg_np[0]
                 else: # More than 1 channel? Use first.
                     seg_for_label = seg_np[0]
                 image_label = get_image_label(seg_for_label, self.min_pixels_threshold)
            else:
                 # Handle unexpected shape, label remains 0
                 # Consider logging this event
                 pass
            
            d[self.label_key] = image_label
            # Remove the segmentation key after processing?
            # Let's keep seg for now, might be useful for debugging test loop
            # if not self.allow_missing_keys and key in d:
            #      del d[key]
        return d

# --- New DataLoader based on Dataloaders.py structure ---
class ArtifactClassificationDataLoader(DataLoader):
    """
    batchgenerators compatible DataLoader for artifact classification.
    Loads paired image/segmentation NPZ files using MONAI transforms for initial preprocessing,
    derives classification labels, and prepares batches for MultiThreadedAugmenter.
    Modeled after CustomDataLoader in Dataloaders.py.
    """
    def __init__(self, data_dicts, batch_size, monai_transforms, min_pixels_threshold=100, num_threads_in_multithreaded=1):
        """
        Args:
            data_dicts: List of dictionaries, each containing 'image_path' and 'seg_path'.
            batch_size: The batch size for generating batches.
            monai_transforms: MONAI Compose object for initial loading and preprocessing.
                              Should output 'image' and 'seg' keys (Tensors or NumPy).
            min_pixels_threshold: Minimum artifact pixels required *after* transforms to assign label 1 or 2.
            num_threads_in_multithreaded: Passed to DataLoader for internal use (used by batchgenerators).
            # num_item_threads: Number of threads to use for loading items within a batch. Defaults to half CPU cores.
        """
        # Pass num_threads_in_multithreaded=1 to parent, as we handle parallelism differently
        # super().__init__(data_dicts, batch_size, 1)
        super().__init__(data_dicts, batch_size, num_threads_in_multithreaded) # Revert to original parent call
        self.monai_transforms = monai_transforms
        self.min_pixels_threshold = min_pixels_threshold
        self.indices = list(range(len(data_dicts)))
        # if num_item_threads is None:
        #      num_item_threads = max(1, os.cpu_count() // 2)
        # self.num_item_threads = num_item_threads
        # No need to store cli_args globally here

    def __len__(self):
        return len(self._data)

    # get_indices is inherited from DataLoader

    # Remove the parallel loading helper function
    # def _load_and_transform_item(self, index):
    #     ...

    def generate_train_batch(self):
        """
        Generates a batch of data suitable for training.
        Loads data sequentially, applies MONAI transforms (including crop),
        checks cropped seg for min pixels, derives labels (1 or 2), returns NumPy arrays.
        """
        indices = self.get_indices() # Get indices for the current batch

        batch_images = []
        batch_segs = []
        batch_labels = [] # Store original labels (1 or 2)
        batch_filenames = []
        skipped_count = 0

        # Sequential processing loop
        for idx in indices:
            data_dict_i = self._data[idx]
            filename = Path(data_dict_i.get('image_path', 'unknown')).name
            try:
                # Apply all MONAI transforms, including the crop
                transformed_data = self.monai_transforms(data_dict_i)

                if "image" not in transformed_data or "seg" not in transformed_data:
                     tqdm.write(f"Warning: MONAI transforms did not produce 'image' and 'seg' keys for {filename}. Skipping.")
                     skipped_count += 1
                     continue

                # Ensure seg is numpy for checking
                seg_tensor = transformed_data['seg']
                seg_np = seg_tensor.cpu().numpy() if isinstance(seg_tensor, torch.Tensor) else np.asarray(seg_tensor)
                image_tensor = transformed_data['image']
                image_np = image_tensor.cpu().numpy() if isinstance(image_tensor, torch.Tensor) else np.asarray(image_tensor)

                # Check if the *cropped* seg has enough artifact pixels
                # Using min_artifact_pixels argument (e.g., 10)
                artifact_pixels = np.count_nonzero(seg_np == 1) + np.count_nonzero(seg_np == 2)
                if artifact_pixels < self.min_pixels_threshold:
                    # tqdm.write(f"Debug: Cropped {filename} had only {artifact_pixels} artifact pixels (threshold {self.min_pixels_threshold}). Skipping.")
                    skipped_count += 1
                    continue # Skip this sample

                # Derive label (1 or 2) from the valid cropped seg
                if seg_np.shape[0] == 1:
                    seg_for_label = seg_np[0]
                elif seg_np.ndim > 0 and seg_np.shape[0] > 1:
                    seg_for_label = seg_np[0]
                else:
                    # Should not happen if artifact_pixels check passed, but as fallback:
                    tqdm.write(f"Warning: Unexpected seg shape {seg_np.shape} for {filename} after passing pixel check. Skipping.")
                    skipped_count += 1
                    continue
                
                # Use get_image_label, thresholding >= 1 pixel since we already checked the minimum count
                image_label = get_image_label(seg_for_label, min_pixels_threshold=1) 
                
                if image_label == 0: # Should ideally not happen if artifact_pixels check is correct
                     tqdm.write(f"Warning: Label derived as 0 for {filename} after passing pixel count check ({artifact_pixels} pixels). Skipping.")
                     skipped_count += 1
                     continue

                # Add the valid sample to the batch lists
                batch_images.append(image_np)
                batch_segs.append(seg_np)
                batch_labels.append(image_label) # Store 1 or 2
                batch_filenames.append(filename)

            except FileNotFoundError as e:
                 tqdm.write(f"File not found error processing sample index {idx} ({data_dict_i.get('image_path', 'N/A')}): {e}. Skipping.")
                 skipped_count += 1
                 continue
            except Exception as e:
                tqdm.write(f"Error processing sample index {idx} ({data_dict_i.get('image_path', 'N/A')}): {e}")
                skipped_count += 1
                # import traceback
                # traceback.print_exc()
                continue # Skip this sample

        if skipped_count > 0 and len(indices) > 0:
            tqdm.write(f"Skipped {skipped_count}/{len(indices)} samples in batch due to missing keys, errors, or insufficient pixels post-crop.")

        if not batch_images:
            # If all samples were skipped, return an empty batch structure
            tqdm.write("Warning: generate_train_batch generated an empty batch after processing/filtering.")
            c, d, h, w = 1, 64, 160, 256 # Placeholder shape, adjust if needed
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.float32),
                'label': np.empty((0,), dtype=np.int64),
                'filenames': [],
                'roi': np.empty((0, c, d, h, w), dtype=np.int64)
             }

        # Stack the collected valid samples
        try:
            image_batch_np = np.stack(batch_images, axis=0)
            seg_batch_np = np.stack(batch_segs, axis=0)
            label_batch_np = np.array(batch_labels, dtype=np.int64) # Labels are 1 or 2 here
        except Exception as stack_e:
            tqdm.write(f"Error stacking batch data: {stack_e}")
            tqdm.write(f"Individual image shapes: {[img.shape for img in batch_images]}")
            tqdm.write(f"Individual seg shapes: {[seg.shape for seg in batch_segs]}")
            # Return empty batch on stacking error
            c, d, h, w = 1, 64, 160, 256 # Placeholder shape
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.float32),
                'label': np.empty((0,), dtype=np.int64),
                'filenames': [],
                'roi': np.empty((0, c, d, h, w), dtype=np.int64)
             }

        return {
            'data': image_batch_np,
            'seg': seg_batch_np,
            'label': label_batch_np, # Labels are 1 or 2
            'roi': np.ones_like(seg_batch_np, dtype=np.int64),
            'filenames': batch_filenames
        }


# --- Original MONAI Dataset (can still be used for validation) ---
class ArtifactDataset(monai.data.Dataset): # Inherit from monai.data.Dataset
    """
    Custom Dataset to load .npy/.npz files (containing image and seg)
    and derive classification labels from segmentations.
    """
    def __init__(self, data_dicts, transforms):
        super().__init__(data=data_dicts, transform=transforms)

    def _transform(self, index: int):
        """
        Loads paired image/seg, applies transforms, derives label.
        """
        data_i = self.data[index]
        filename = Path(data_i['image_path']).name
        transformed_data = self.transform(data_i)

        seg_data_tensor = transformed_data['seg']
        seg_data_np = seg_data_tensor.cpu().numpy()

        global cli_args
        min_pixels = cli_args.min_artifact_pixels if 'cli_args' in globals() else 100 # Default fallback
        image_label = get_image_label(seg_data_np, min_pixels_threshold=min_pixels)

        image_tensor = transformed_data["image"]
        return {"image": image_tensor, "label": image_label, "filename": filename}


# --- Custom Debug Logging Transform ---
class LogKeysd(MapTransform):
    """Logs the keys present in the dictionary at this point."""
    def __init__(self, keys, log_prefix="", allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys)
        # `keys` argument is required by MapTransform but we ignore it
        self.log_prefix = log_prefix

    def __call__(self, data):
        # We don't modify the data, just log its keys
        # Add filename to log if available for better tracking
        filename = data.get('filename', data.get('image_path', ''))
        if isinstance(filename, Path):
             filename = filename.name
        elif isinstance(filename, str):
             filename = Path(filename).name
        print(f"DEBUG Keys {self.log_prefix} ({filename}): {list(data.keys())}")
        return data
# --- End Custom Debug Logging Transform ---

def main(args):
    set_determinism(seed=args.seed)
    output_dir = Path(args.output_dir)
    run_name = f"run_ep{args.epochs}_bs{args.batch_size}_lr{args.initial_lr}_optim{args.optimizer}"
    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_output_dir / f"classify_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    # --- Initialize WandB ---
    if wandb and not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=str(output_dir),
                resume="allow",
                # For simplicity now, generate new ID unless specific resume path given
                id=wandb.util.generate_id() if args.resume is None else None
            )
            print(f"WandB logging enabled. Project: {args.wandb_project}, Run: {run_name}")
            wandb_enabled = True
        except Exception as e:
            print(f"Error initializing WandB: {e}. Disabling WandB logging.")
            wandb_enabled = False
    else:
        print("WandB logging disabled.")
        wandb_enabled = False
    # --- End WandB Init ---

    # --- Data Setup ---
    dataset_path = Path(args.dataset_dir)
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'validate'
    test_path = dataset_path / 'test'

    if not train_path.is_dir() or not val_path.is_dir() or not test_path.is_dir():
        raise NotADirectoryError(
            f"Expected 'train', 'validate', and 'test' subdirectories in {args.dataset_dir}"
        )

    # --- Find and Pair Files (Multithreaded) --- 
    def check_and_pair_file(img_path_str, data_dir, pre_filter_non_artifact):
        """Helper function to check one image file path for pairing and filtering."""
        img_path = Path(img_path_str)
        label_filename = img_path.name.replace('_image_', '_maskArtifact_', 1)
        label_path = img_path.with_name(label_filename)

        if label_path.is_file():
            if pre_filter_non_artifact:
                try:
                    seg_npz = np.load(label_path)
                    if 'arr_0' not in seg_npz:
                        # Log warning or handle appropriately - maybe return special value?
                        # For now, treat as if it might contain artifacts if key is missing
                        seg_data = None 
                    else:
                        seg_data = seg_npz['arr_0']
                        seg_npz.close()

                    if seg_data is not None and (np.any(seg_data == 1) or np.any(seg_data == 2)):
                        return {"image_path": img_path_str, "seg_path": str(label_path), "status": "paired"}
                    elif seg_data is not None: # Contained only background or was empty
                        return {"status": "filtered"}
                    else: # seg_data was None (arr_0 key missing)
                        # Include for now if key was missing
                        # Log this specific case when processing results
                        return {"image_path": img_path_str, "seg_path": str(label_path), "status": "paired_key_missing"}
                except Exception as e:
                    # Log warning or handle error - maybe return special value?
                    # Include file but mark that loading failed
                    return {"image_path": img_path_str, "seg_path": str(label_path), "status": "paired_load_error", "error": str(e)}
            else: # No pre-filtering
                return {"image_path": img_path_str, "seg_path": str(label_path), "status": "paired"}
        else: # Label file not found
            return {"status": "missing_label", "image_name": img_path.name}

    def find_and_pair_files(data_dir, file_pattern, pre_filter_non_artifact=True, num_threads=None):
        if num_threads is None:
             num_threads = os.cpu_count() # Default to number of CPU cores
        
        paired_files = []
        missing_labels = 0
        filtered_out = 0
        load_errors = 0
        included_key_missing = 0

        print(f"Searching for image files ('{file_pattern}' containing '_image_') in {data_dir}...")
        potential_image_files = sorted([str(f) for f in data_dir.glob(file_pattern) if '_image_' in f.name])
        print(f"Found {len(potential_image_files)} potential image files. Pairing and filtering using up to {num_threads} threads...")
        
        results = []
        # Use ThreadPoolExecutor for parallel I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Create futures for each file check
            future_to_path = {executor.submit(check_and_pair_file, img_path, data_dir, pre_filter_non_artifact): img_path for img_path in potential_image_files}
            
            # Process results as they complete, using tqdm for progress
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(potential_image_files), desc=f"Pairing/Filtering {data_dir.name}"):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    tqdm.write(f'Error processing {path}: {exc}')
                    results.append({"status": "processing_error", "image_name": Path(path).name, "error": str(exc)})

        # Process the collected results
        for result in results:
             status = result.get("status")
             if status == "paired":
                 paired_files.append({"image_path": result["image_path"], "seg_path": result["seg_path"]})
             elif status == "paired_key_missing":
                 paired_files.append({"image_path": result["image_path"], "seg_path": result["seg_path"]})
                 included_key_missing += 1
                 tqdm.write(f"Including {result['image_path']} despite missing 'arr_0' in label for filtering.")
             elif status == "paired_load_error":
                 paired_files.append({"image_path": result["image_path"], "seg_path": result["seg_path"]})
                 load_errors += 1
                 tqdm.write(f"Warning: Error loading/checking label for {result['image_path']} during pre-filtering: {result.get('error', '')}. Including file.")
             elif status == "filtered":
                 filtered_out += 1
             elif status == "missing_label":
                missing_labels += 1
                tqdm.write(f"Warning: Derived label file not found for image {result.get('image_name', 'N/A')} in {data_dir}. Skipping.")
             elif status == "processing_error":
                 # Decide how to handle overall processing errors, maybe increment a counter
                 pass # Already logged by the tqdm loop

        filter_msg = f" {filtered_out} files filtered out (no class 1 or 2 pixels)." if pre_filter_non_artifact else ""
        error_msg = f" {load_errors} label load errors during filtering (included anyway)." if load_errors > 0 else ""
        key_msg = f" {included_key_missing} included despite missing label key." if included_key_missing > 0 else ""
        print(f"Successfully paired {len(paired_files)} files in {data_dir}.{filter_msg} {missing_labels} missing labels.{error_msg}{key_msg}")
        return paired_files, missing_labels

    # Determine number of threads (can be made an argument later)
    num_pairing_threads = max(1, os.cpu_count() // 2) # Use half the cores by default 

    # Apply pre-filtering only to training/validation data
    print(f"Using {num_pairing_threads} threads for file pairing/filtering.")
    train_files_list, missing_train = find_and_pair_files(train_path, args.file_pattern, pre_filter_non_artifact=True, num_threads=num_pairing_threads)
    validate_files_list, missing_val = find_and_pair_files(val_path, args.file_pattern, pre_filter_non_artifact=True, num_threads=num_pairing_threads)
    # Keep all test files for comprehensive evaluation, do not pre-filter
    test_files_list, missing_test = find_and_pair_files(test_path, args.file_pattern, pre_filter_non_artifact=False, num_threads=num_pairing_threads) 

    train_files = train_files_list + validate_files_list
    test_files = test_files_list

    total_missing = missing_train + missing_val + missing_test
    if total_missing > 0:
        print(f"Warning: Total {total_missing} label files were missing across all directories.")

    if not train_files:
        raise ValueError("No training image/label pairs were successfully found in 'train' or 'validate' directories.")
    if not test_files:
        raise ValueError("No validation/testing image/label pairs were successfully found in 'test' directory.")

    print(f"Using {len(train_files)} samples from 'train'+'validate' for Training.")
    print(f"Using {len(test_files)} samples from 'test' for Testing during training loop.")

    img_key = "image"
    seg_key = "seg"

    # Define foreground labels for cropping
    foreground_labels = [1, 2]

    # --- Original transforms with interspersed logging --- 
    # print("DEBUG: Using simplified transforms (Load, Channel, Orient, Resize)")
    base_transforms = [
        LoadPairedArr0d(keys=("image_path", "seg_path")),
        LogKeysd(keys=(), log_prefix="After LoadPairedArr0d"), # Log after loading

        EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel", allow_missing_keys=True),
        LogKeysd(keys=(), log_prefix="After EnsureChannelFirstd"), # Log after channel

        Orientationd(keys=[img_key, seg_key], axcodes="RAS", allow_missing_keys=True),
        LogKeysd(keys=(), log_prefix="After Orientationd"), # Log after orientation

        # Crop based on foreground labels before expensive transforms
        RandCropByPosNegLabeld(
            keys=[img_key, seg_key],
            label_key=seg_key,
            spatial_size=args.input_size,
            pos=0.8, # PREFER foreground center (80%)
            neg=0.2, # ALLOW background center (20%) as fallback
            num_samples=1,
            allow_smaller=True # Allow smaller output if input is smaller than crop size
        ),
        LogKeysd(keys=(), log_prefix="After RandCropByPosNegLabeld"), # Log after crop

        Spacingd(keys=[img_key, seg_key], pixdim=args.target_spacing, mode=("bilinear", "nearest"), allow_missing_keys=True),
        LogKeysd(keys=(), log_prefix="After Spacingd"), # Log after spacing
        
        # Resized might be redundant now if spatial_size in RandCropByPosNegLabeld == args.input_size
        # Resized(keys=[img_key, seg_key], spatial_size=args.input_size, mode=("bilinear", "nearest"), allow_missing_keys=True), 
        # LogKeysd(keys=(), log_prefix="After Resized"), # Log after resize (if used)
    ]
    # --- END Original transforms with logging --- 

    try:
        augmentations_result = get_augmentations()
        print("Successfully called get_augmentations().")

        if isinstance(augmentations_result, Compose):
            specific_transforms_list = list(augmentations_result.transforms)
            print(f"Extracted {len(specific_transforms_list)} transforms from returned Compose object.")
        elif isinstance(augmentations_result, (list, tuple)):
            specific_transforms_list = list(augmentations_result)
            print(f"Using {len(specific_transforms_list)} transforms from returned list/tuple.")
        elif isinstance(augmentations_result, monai.transforms.Transform):
            specific_transforms_list = [augmentations_result]
            print("Using the single transform returned by get_augmentations.")
        elif augmentations_result is None:
            print("get_augmentations() returned None. Applying only base transforms.")
            specific_transforms_list = []
        else:
            print(f"Warning: get_augmentations() returned unusable type: {type(augmentations_result)}. Applying only base transforms.")
            specific_transforms_list = []

    except ImportError:
        print("Warning: Could not import get_augmentations from Augmentations.py. Proceeding without custom augmentations.")
        specific_transforms_list = []
    except Exception as e:
        print(f"Warning: Error calling get_augmentations(): {e}. Proceeding without custom augmentations.")
        specific_transforms_list = []


    pre_aug_transforms = Compose([
        *base_transforms,
    ])
    test_transforms = Compose([
        *base_transforms,
        EnsureTyped(keys=[img_key, seg_key], dtype=torch.float32),
    ])


    print("Setting up BatchGenerators training data loader...")
    train_dl = ArtifactClassificationDataLoader(
        data_dicts=train_files,
        batch_size=args.batch_size,
        monai_transforms=pre_aug_transforms,
        min_pixels_threshold=args.min_artifact_pixels,
        num_threads_in_multithreaded=args.num_workers
    )

    print("Setting up MONAI test data loader...")
    # Use standard MONAI dataset/dataloader for testing as it doesn't need batchgenerators structure
    # Make sure test_transforms match what the model expects (e.g., output Tensors)
    # Apply cropping to test set as well for consistency?
    # If yes, use CenterSpatialCropd instead of random
    # from monai.transforms import CenterSpatialCropd 
    test_compose = Compose([
        LoadPairedArr0d(keys=("image_path", "seg_path")), # Load NPZ
        EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel"),
        Orientationd(keys=[img_key, seg_key], axcodes="RAS"),
        # Option: Apply Center Crop to test set for consistency
        # CenterSpatialCropd(keys=[img_key, seg_key], roi_size=args.input_size),
        Spacingd(keys=[img_key, seg_key], pixdim=args.target_spacing, mode=("bilinear", "nearest")),
        Resized(keys=[img_key, seg_key], spatial_size=args.input_size, mode=("bilinear", "nearest")), # Keep resize if not cropping test
        # Define a transform to get the label *after* initial processing
        GetLabelFromSegd(keys=[seg_key], label_key='label', min_pixels_threshold=args.min_artifact_pixels),
        EnsureTyped(keys=[img_key, 'label'], dtype=(torch.float32, torch.int64)), # Ensure image is float, label is long
    ])

    test_ds = monai.data.Dataset(data=test_files, transform=test_compose) if test_files else None

    print("Getting batchgenerators transforms from Augmentations.py...")
    bg_transforms = get_augmentations()
    if bg_transforms is None:
         print("Warning: get_augmentations returned None, training loader will have no batchgenerators transforms.")

    print("Initializing MultiThreadedAugmenter for training...")
    train_loader = MultiThreadedAugmenter(
        train_dl,
        transform=bg_transforms,
        num_processes=args.num_workers,
        num_cached_per_queue=3,
        pin_memory=True,
        seeds=None,
        useroi=False, # Disable ROI calculation
        generate_patches=False # Disable patch generation
    )
    print("MultiThreadedAugmenter initialized.")


    test_loader = PyTorchDataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate
    ) if test_ds else None
    print("Test DataLoader initialized.")

    batches_per_epoch = math.ceil(len(train_dl) / args.batch_size)
    print(f"Calculated batches per epoch: {batches_per_epoch}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_output_classes = 2 # Targeting artifact classes 1 and 2 only
    print(f"Model configured for {num_output_classes} output classes (Artifact 1, Artifact 2)")
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        # out_channels=args.num_classes
        out_channels=num_output_classes # Set to 2
    ).to(device)

    if args.use_weighted_loss:
        if len(args.class_weights) != num_output_classes:
             raise ValueError(f"Expected {num_output_classes} class weights for 2-class output, but got {len(args.class_weights)}. Weights should correspond to original classes 1 and 2.")
        # Weights correspond to mapped classes 0 (original 1) and 1 (original 2)
        weights = torch.tensor(args.class_weights).float().to(device)
        print(f"Using weighted CrossEntropyLoss for 2 classes. Weights (Orig Cls 1, Orig Cls 2): {weights.cpu().numpy()}")
        # criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print("Using standard CrossEntropyLoss for 2 classes.")
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        print(f"Using AdamW optimizer: LR={args.initial_lr}, WeightDecay={args.weight_decay}")
    elif args.optimizer.lower() == 'sgd':
         optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
         print(f"Using SGD optimizer: LR={args.initial_lr}, Momentum={args.momentum}, Nesterov={args.nesterov}, WeightDecay={args.weight_decay}")
    else:
         raise ValueError(f"Unsupported optimizer: {args.optimizer}")


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    resume_path = Path(args.resume) if args.resume else run_output_dir / "checkpoint_latest.pt"

    if resume_path.exists() and resume_path.is_file():
        print(f"Attempting to load checkpoint: {resume_path}")
        ckpt_path = Path(resume_path)
        if ckpt_path.is_file():
            print(f"Resuming training from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            try:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_metric = checkpoint.get('best_metric', -1)
                best_metric_epoch = checkpoint.get('best_metric_epoch', -1)
                print(f"  Loaded model, optimizer, scheduler. Resuming from Epoch {start_epoch}")
                print(f"  Previous best metric (Accuracy): {best_metric:.4f} at epoch {best_metric_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint state: {e}. Starting from scratch.")
                start_epoch = 0
                best_metric = -1
                best_metric_epoch = -1
        else:
            if args.resume:
                 print(f"Resume path specified ({args.resume}), but file not found. Starting from scratch.")
            else:
                 print(f"No latest checkpoint found in {run_output_dir}. Starting from scratch.")
    else:
        start_epoch = 0
        best_metric = -1 # Best validation accuracy
        best_metric_epoch = -1


    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print("-" * 10)
        print(f"Epoch {epoch}/{args.epochs - 1}")

        # --- Training Phase ---
        model.train()
        train_loss = 0
        train_steps = 0
        train_correct = 0
        train_total = 0
        train_correct_12 = 0 # Correct predictions for class 1 vs 2
        train_total_12 = 0   # Total samples of class 1 vs 2
        recent_train_losses = [] # For moving average
        recent_train_accuracies_12 = [] # For moving average of 1-vs-2 accuracy

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Train", unit="batch", leave=False)
        for batch_data in progress_bar:
            try:
                if not batch_data or 'data' not in batch_data or 'label' not in batch_data:
                    tqdm.write(f"Warning: Skipping empty or invalid batch from train_loader in epoch {epoch}.")
                    continue

                image_batch = batch_data['data']
                label_batch = batch_data['label']

                # Handle data type: could be np.ndarray or torch.Tensor
                if isinstance(image_batch, np.ndarray):
                    inputs = torch.from_numpy(image_batch).float().to(device)
                elif isinstance(image_batch, torch.Tensor):
                    inputs = image_batch.float().to(device)
                else:
                    tqdm.write(f"Warning: Unexpected type for image batch: {type(image_batch)}. Skipping batch.")
                    continue

                # Handle label type: could be np.ndarray or torch.Tensor (less likely but safe)
                if isinstance(label_batch, np.ndarray):
                    # Labels are 1 or 2 from loader
                    labels_1_2 = torch.from_numpy(label_batch).long().to(device)
                elif isinstance(label_batch, torch.Tensor):
                    labels_1_2 = label_batch.long().to(device)
                else:
                    tqdm.write(f"Warning: Unexpected type for label batch: {type(label_batch)}. Skipping batch.")
                    continue
                
                # Map labels 1, 2 to 0, 1 for 2-class loss
                mapped_labels = labels_1_2 - 1

                optimizer.zero_grad()
                outputs = model(inputs)
                # Loss expects outputs (B, 2) and mapped_labels (B,) containing 0 or 1
                loss = criterion(outputs, mapped_labels)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                train_loss += loss.item()
                train_steps += 1

                recent_train_losses.append(current_loss)
                if len(recent_train_losses) > args.log_freq:
                    recent_train_losses.pop(0)

                    _, predicted_mapped = torch.max(outputs.data, 1)
                    batch_total_samples = mapped_labels.size(0)
                    batch_correct_mapped = (predicted_mapped == mapped_labels).sum().item()
                    
                    # Since all samples reaching here should be class 1 or 2, total_12 == total_samples
                    train_correct_12 += batch_correct_mapped 
                    train_total_12 += batch_total_samples
                    batch_accuracy_12 = (batch_correct_mapped / batch_total_samples) * 100 if batch_total_samples > 0 else 0

                    recent_train_accuracies_12.append(batch_accuracy_12)
                    if len(recent_train_accuracies_12) > args.log_freq:
                        recent_train_accuracies_12.pop(0)

                if wandb_enabled and train_steps % args.log_freq == 0:
                    moving_avg_loss = np.mean(recent_train_losses) if recent_train_losses else current_loss
                    wandb.log({
                        "train/step_loss": current_loss,
                        "train/step_loss_moving_avg": moving_avg_loss,
                            "train/step_accuracy_12": batch_accuracy_12,
                            "train/step_accuracy_12_moving_avg": np.mean(recent_train_accuracies_12) if recent_train_accuracies_12 else batch_accuracy_12,
                        "epoch": epoch + (train_steps / batches_per_epoch)
                    }, step=epoch * batches_per_epoch + train_steps)

                if train_steps > 0:
                    moving_avg_loss = np.mean(recent_train_losses) if recent_train_losses else current_loss
                    moving_avg_acc_12 = np.mean(recent_train_accuracies_12) if recent_train_accuracies_12 else batch_accuracy_12
                    progress_bar.set_postfix(
                            loss=f"{moving_avg_loss:.4f}",
                            acc12=f"{moving_avg_acc_12:.2f}%"
                        )

            except KeyError as e:
                 tqdm.write(f"Error: Missing key {e} in batch data from MultiThreadedAugmenter.")
                 tqdm.write(f"Batch keys: {batch_data.keys() if isinstance(batch_data, dict) else type(batch_data)}")
                 continue # Skip this batch
            except Exception as e:
                 tqdm.write(f"Error processing batch from MultiThreadedAugmenter: {e}")
                 continue # Skip this batch

        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        train_accuracy_12 = 100 * train_correct_12 / train_total_12 if train_total_12 > 0 else 0
        print(f"Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}, Accuracy (Cls 1&2): {train_accuracy_12:.2f}% ({train_correct_12}/{train_total_12})")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} Learning Rate: {current_lr:.6f}")


        # --- Validation Phase ---
        model.eval()
        avg_test_loss = np.nan # Default if no testing
        test_accuracy = np.nan
        test_metrics = {"epoch": epoch}

        if test_loader:
            test_loss = 0
            test_steps = 0
            all_preds_mapped = [] # Store model predictions (0 or 1)
            all_labels_mapped = [] # Store mapped ground truth (0 or 1)
            print(f"Running Testing for Epoch {epoch}...")

            with torch.no_grad():
                for batch_data in tqdm(test_loader, desc=f"Epoch {epoch} Test", unit="batch", leave=False):
                    # Handle data type consistency for validation/test loader as well
                    image_batch = batch_data["image"]
                    label_batch = batch_data["label"]

                    if isinstance(image_batch, np.ndarray):
                        inputs = torch.from_numpy(image_batch).float().to(device)
                    elif isinstance(image_batch, torch.Tensor):
                        inputs = image_batch.float().to(device)
                    else:
                         tqdm.write(f"Warning: Unexpected type for test image batch: {type(image_batch)}. Skipping.")
                         continue
                    
                    if isinstance(label_batch, np.ndarray):
                        labels_0_1_2 = torch.from_numpy(label_batch).long().to(device)
                    elif isinstance(label_batch, torch.Tensor):
                        labels_0_1_2 = label_batch.long().to(device) # Ensure long type
                    else:
                         tqdm.write(f"Warning: Unexpected type for test label batch: {type(label_batch)}. Skipping.")
                         continue
                    
                    # Filter out samples with label 0 for metrics
                    mask_1_2 = (labels_0_1_2 != 0)
                    
                    if not torch.any(mask_1_2):
                        continue # Skip batch if no class 1 or 2 samples
                        
                    inputs_filtered = inputs[mask_1_2]
                    labels_1_2_filtered = labels_0_1_2[mask_1_2]
                    
                    # Map 1, 2 -> 0, 1 for loss calculation (if needed) and metrics
                    mapped_labels_filtered = labels_1_2_filtered - 1
                   
                    # Get model output for filtered inputs
                    test_outputs = model(inputs_filtered)
                    # Calculate loss only on relevant samples (using mapped labels)
                    loss = criterion(test_outputs, mapped_labels_filtered)

                    test_loss += loss.item() * mapped_labels_filtered.size(0) # Weight loss by number of valid samples
                    test_steps += mapped_labels_filtered.size(0) # Count valid samples

                    _, predicted_mapped = torch.max(test_outputs.data, 1)
                    all_preds_mapped.extend(predicted_mapped.cpu().numpy())
                    all_labels_mapped.extend(mapped_labels_filtered.cpu().numpy())

            avg_test_loss = test_loss / test_steps if test_steps > 0 else 0
            # Calculate accuracy using the mapped predictions and labels (0 and 1)
            test_accuracy_12 = accuracy_score(all_labels_mapped, all_preds_mapped) * 100 if all_labels_mapped else 0

            # --- Filter for 1-vs-2 accuracy --- REMOVED (Already filtered)
            # filtered_pairs = [(p, l) for p, l in zip(all_preds, all_labels) if l != 0]
            # test_accuracy_12 = np.nan
            # report_12 = "N/A"
            # cm_12 = None

            # if filtered_pairs:
            #     filtered_preds = [p for p, l in filtered_pairs]
            #     filtered_labels = [l for p, l in filtered_pairs]
            #     test_accuracy_12 = accuracy_score(filtered_labels, filtered_preds) * 100
            target_names_12 = ["Artifact1", "Artifact2"] # Target names for the 2 classes
            report_12 = "N/A"
            cm_12 = None
            if all_labels_mapped:
                try:
                    report_12 = classification_report(all_labels_mapped, all_preds_mapped, target_names=target_names_12, zero_division=0)
                    # CM labels should be 0, 1
                    cm_12 = confusion_matrix(all_labels_mapped, all_preds_mapped, labels=[0, 1]) 
                except ValueError as e:
                    print(f"Could not generate classification report/CM for test data: {e}")
            # else:
            #     print("No samples of class 1 or 2 found in test set after filtering.")
            # --- End filtering ---

            # print(f"Epoch {epoch} Average Test Loss: {avg_test_loss:.4f}, Overall Accuracy: {test_accuracy:.2f}%, Accuracy (Cls 1&2): {test_accuracy_12:.2f}%")
            print(f"Epoch {epoch} Average Test Loss (Cls 1&2): {avg_test_loss:.4f}, Test Accuracy (Cls 1&2): {test_accuracy_12:.2f}%")

            # Log both overall and filtered accuracy
            test_metrics["test/epoch_loss_12"] = avg_test_loss
            # test_metrics["test/epoch_accuracy_overall"] = test_accuracy 
            test_metrics["test/epoch_accuracy_12"] = test_accuracy_12

            # Log filtered report and CM if available
            # if filtered_pairs and cm_12 is not None:
            if all_labels_mapped and cm_12 is not None:
                # target_names = [f"Class_{i}" for i in range(args.num_classes)]
                print("Test Classification Report (Artifact 1 vs 2):")
                print(report_12)
                print("Test Confusion Matrix (Artifact 1 vs 2):")
                print(cm_12)

            if wandb_enabled:
                    try:
                        # Log the filtered report dictionary
                        report_dict_12 = classification_report(all_labels_mapped, all_preds_mapped, target_names=target_names_12, zero_division=0, output_dict=True)
                        for class_name, metrics_dict in report_dict_12.items():
                            if isinstance(metrics_dict, dict):
                                for metric_name, value in metrics_dict.items():
                                    test_metrics[f"test_report_12/{class_name}_{metric_name}"] = value
                            else:
                                test_metrics[f"test_report_12/{class_name}"] = metrics_dict
                         
                        # Log the filtered confusion matrix
                        test_metrics["test/confusion_matrix_12"] = wandb.Table(
                            columns=target_names_12, # Use 2 class names
                            data=cm_12.tolist(),
                            rows=target_names_12 # Use 2 class names
                        )

                    except Exception as report_e:
                        print(f"Warning: Could not format detailed filtered report/cm for WandB: {report_e}")
                        # Delete old logging for overall report/cm
                        # if all_labels:
                        #     target_names = [f"Class_{i}" for i in range(args.num_classes)]

                    f.write(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Train Acc (1&2): {train_accuracy_12:.2f}, "
                    f"Test Loss: {avg_test_loss:.4f}, Test Acc (1&2): {test_accuracy_12:.2f}, LR: {current_lr:.6f}")
            if all_labels_mapped and cm_12 is not None:
                f.write("\nTest Report (1&2):\n")
                f.write(str(report_12) + "\n")
                f.write("Test Confusion Matrix (1&2):\n")
                f.write(np.array2string(cm_12) + "\n")
            f.write("-" * 20 + "\n")

        if wandb_enabled:
             wandb.log(test_metrics, step=epoch * batches_per_epoch + batches_per_epoch)
             moving_avg_loss_epoch_end = np.mean(recent_train_losses) if recent_train_losses else avg_train_loss
             wandb.log({
                 "train/epoch_loss": avg_train_loss,
                 "train/epoch_accuracy_12": train_accuracy_12,
                 "train/epoch_loss_moving_avg": moving_avg_loss_epoch_end,
                 "train/epoch_accuracy_12_moving_avg": np.mean(recent_train_accuracies_12) if recent_train_accuracies_12 else train_accuracy_12,
                 "learning_rate": current_lr,
                 "epoch": epoch
             }, step=epoch * batches_per_epoch + batches_per_epoch)


    print(f"Training finished. Best Test Accuracy: {best_metric:.2f}% at epoch {best_metric_epoch}")
    print(f"Logs saved to: {log_file}")
    print(f"Best checkpoint saved to: {run_output_dir / 'checkpoint_best.pt'}")

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D CNN for Artifact Classification")

    # --- Data Arguments ---
    parser.add_argument('dataset_dir', type=str, help="Directory containing 'train', 'validate', and 'test' subfolders with NPZ files.")
    parser.add_argument('--output_dir', type=str, default='./results_classify', help="Directory to save checkpoints and logs.")
    parser.add_argument('--file_pattern', type=str, default='*.np[yz]', help="Glob pattern for data files (e.g., '*.npy', '*.npz'). Note: Script assumes _image_ in filename identifies image files.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for data loading.")

    # --- Model & Training Arguments ---
    parser.add_argument('--num_classes', type=int, required=True, help="Number of output classes (e.g., 3 for Bkg, Art1, Art2).")
    parser.add_argument('--input_size', type=int, nargs=3, default=[64, 160, 256], help='Input size for the network (D, H, W).')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.5, 1.5, 1.5], help='Target voxel spacing (x, y, z).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')

    # --- NPZ Loading Arguments ---
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for scheduler.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer type.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW/SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=True, help='Use Nesterov momentum for SGD.')

    # --- Data Labeling Arguments ---
    parser.add_argument('--min_artifact_pixels', type=int, default=20,
                        help='Minimum number of artifact voxels required in transformed mask to assign label 1 or 2.')

    # --- Loss Function Arguments ---
    parser.add_argument('--use_weighted_loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Use weighted CrossEntropyLoss based on class distribution.')
    # Default weights based on transformed distribution (Fold 0) with capped Class 0 weight
    # Weights now correspond to original class 1 and class 2 mapped to 0 and 1
    parser.add_argument('--class_weights', type=float, nargs=2, default=[1.5, 1],
                        help='Weights for Artifact Class 1 and Artifact Class 2 (mapped to outputs 0, 1) for weighted loss. Ignored if --no_use_weighted_loss.')

    # --- System Arguments ---
    parser.add_argument('--device', type=int, default=0, help="GPU device ID to use.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from.')

    # --- Logging Arguments ---
    parser.add_argument('--log_freq', type=int, default=100, help='Log training metrics every N steps.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging.')
    parser.add_argument('--wandb_project', type=str, default='ArtifactClassification', help='WandB project name.')

    args = parser.parse_args()

    global cli_args
    cli_args = args

    if args.num_classes <= 1:
         raise ValueError("num_classes must be at least 2 for classification.")
    # Add check for class weights length if used
    if args.use_weighted_loss and len(args.class_weights) != 2:
        raise ValueError(f"--class_weights must provide exactly 2 weights when using 2-class output, but got {len(args.class_weights)}.")

    main(args) 