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
import itertools # Import itertools
from typing import Sequence, Union # ADDED import, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as PyTorchDataLoader # Alias PyTorch DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

import monai
from monai.data import (
    DataLoader as PyTorchDataLoader,  # Rename to avoid conflict with built-in DataLoader if needed elsewhere
    Dataset,
    CacheDataset,
    partition_dataset,
    load_decathlon_datalist,
    list_data_collate,
    pad_list_data_collate, # Import the padding collate function
    decollate_batch,
    DistributedSampler,
    DistributedWeightedRandomSampler,
)
from monai.transforms.transform import MapTransform
from monai.transforms import (
    Compose, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, RandRotate90d, RandGaussianNoised, EnsureTyped,
    ScaleIntensityRanged, RandCropByPosNegLabeld,
    NormalizeIntensityd,
    RandZoomd, # Removed RandomApply, kept RandZoomd
    CenterSpatialCropd,
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism, BlendMode, ensure_tuple_rep # Keep BlendMode if needed for map, remove later if not ADDED ensure_tuple_rep
from monai.data.utils import dense_patch_slices # ADDED import
# import monai focal loss
# from monai.losses import FocalLoss # <-- Comment out FocalLoss import

from Augmentations import get_augmentations

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

# ADDED Imports for Visualization
from PIL import Image, ImageDraw, ImageFont

# --- WandB Import ---
try:
    import wandb
except ImportError:
    print("wandb not installed, pip install wandb to enable logging.")
    wandb = None
# --- End WandB Import ---

# --- ADDED Memory Logging Imports ---
import psutil
# --- End ADDED ---

warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", message=".*weights_only=False.*") # Suppress torch.load warning

# --- ADDED Memory Logging Function ---
_process = psutil.Process(os.getpid()) # Get current process once

def log_memory(stage: str):
    mem_info = _process.memory_info()
    # Log RSS (Resident Set Size) - physical memory used
    rss_gb = mem_info.rss / (1024**3)
    # Log VMS (Virtual Memory Size) - total virtual address space
    vms_gb = mem_info.vms / (1024**3)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"MEMLOG [{timestamp}] - {stage}: RSS={rss_gb:.2f} GB, VMS={vms_gb:.2f} GB")
# --- End ADDED ---

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


# --- Visualization Function --- ADDED ---
def save_small_artifact_visualization(
    image_crop: np.ndarray, # Expects CHWD, usually 1HWD
    seg_crop: np.ndarray,   # Expects CHWD, usually 1HWD
    artifact_count: int,
    filename: str,
    output_dir: Path,
    text_label: str = "Artifact Pixels",
    highlight_color: tuple[int, int, int] = (255, 0, 0), # Red
):
    """
    Saves a 2D slice visualization of a 3D crop with few artifact pixels highlighted.
    Finds the slice with the most artifact pixels.
    """
    debug_dir = output_dir / "debug_small_artifacts"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if image_crop.shape[0] != 1 or seg_crop.shape[0] != 1:
        # print(f"DEBUG_VIZ: Skipping viz for {filename}, unexpected channel count.")
        return # Expect single channel

    img_data = image_crop[0] # HWD
    seg_data = seg_crop[0]   # HWD

    best_slice_info = {"axis": -1, "index": -1, "count": -1}

    # Find the slice with the most artifact pixels
    for axis in range(3): # 0: Depth (HWD -> WD), 1: Height (HWD -> HD), 2: Width (HWD -> HW)
        num_slices = img_data.shape[axis]
        for i in range(num_slices):
            if axis == 0:
                seg_slice = seg_data[i, :, :]
            elif axis == 1:
                seg_slice = seg_data[:, i, :]
            else: # axis == 2
                seg_slice = seg_data[:, :, i]

            count = np.count_nonzero((seg_slice == 1) | (seg_slice == 2))
            if count > best_slice_info["count"]:
                best_slice_info = {"axis": axis, "index": i, "count": count}

    if best_slice_info["axis"] == -1:
        # print(f"DEBUG_VIZ: No artifacts found in any slice for {filename}, count was {artifact_count}.")
        return # Should not happen if artifact_count > 0, but safeguard

    # Get the best slices
    axis, index = best_slice_info["axis"], best_slice_info["index"]
    if axis == 0:
        img_slice_2d = img_data[index, :, :]
        seg_slice_2d = seg_data[index, :, :]
    elif axis == 1:
        img_slice_2d = img_data[:, index, :]
        seg_slice_2d = seg_data[:, index, :]
    else: # axis == 2
        img_slice_2d = img_data[:, :, index]
        seg_slice_2d = seg_data[:, :, index]

    # Normalize image slice to 0-255 for visualization
    img_min, img_max = img_slice_2d.min(), img_slice_2d.max()
    if img_max > img_min:
        img_slice_norm = ((img_slice_2d - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_slice_norm = np.zeros_like(img_slice_2d, dtype=np.uint8)

    # Convert grayscale image to RGB
    rgb_slice = np.stack([img_slice_norm] * 3, axis=-1)

    # Create highlight overlay
    highlight_mask = (seg_slice_2d == 1) | (seg_slice_2d == 2)
    rgb_slice[highlight_mask] = highlight_color

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_slice)
    draw = ImageDraw.Draw(pil_image)

    # Add text label for artifact count
    try:
        # Try loading a default font; adjust path if necessary or handle failure
        font = ImageFont.truetype("DejaVuSans.ttf", 15) # Common Linux font
    except IOError:
        try:
             font = ImageFont.truetype("arial.ttf", 15) # Common Windows font
        except IOError:
             font = ImageFont.load_default()

    text = f"{text_label}: {artifact_count}"
    # Simple text placement at top-left
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    # Save the image
    save_filename = f"{Path(filename).stem}_artifacts_{artifact_count}_axis{axis}_slice{index}.png"
    save_path = debug_dir / save_filename
    try:
        pil_image.save(save_path)
        # print(f"DEBUG_VIZ: Saved visualization to {save_path}")
    except Exception as e:
        print(f"Error saving debug image {save_path}: {e}")

# --- End Visualization Function ---

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
        
        print(f"size of d image: {d['image'].shape}")
        print(f"size of d seg: {d['seg'].shape}")
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
            
        return d

# --- New DataLoader based on Dataloaders.py structure ---
class ArtifactClassificationDataLoader(DataLoader):
    """
    batchgenerators compatible DataLoader for artifact classification.
    Loads paired image/segmentation NPZ files using MONAI transforms for initial preprocessing,
    derives classification labels, and prepares batches for MultiThreadedAugmenter.
    Modeled after CustomDataLoader in Dataloaders.py.
    """
    def __init__(self, data_dicts, batch_size, monai_transforms, min_pixels_threshold, args, num_threads_in_multithreaded=1):
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
        super().__init__(data_dicts, batch_size, num_threads_in_multithreaded, seed_for_shuffle=12) # Revert to original parent call
        self.monai_transforms = monai_transforms
        self.min_pixels_threshold = min_pixels_threshold
        self.indices = list(range(len(data_dicts)))
        self.args = args # ADDED: Store args

    def __len__(self):
        return len(self._data)

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
            img_path_str = data_dict_i.get('image_path', 'unknown_image')
            seg_path_str = data_dict_i.get('seg_path', 'unknown_seg')
            filename = Path(img_path_str).name
            try:
                # --- Wrap MONAI transform call ---
                try:
                    # tqdm.write(f"DEBUG: Applying MONAI transforms to {filename}") # Optional: Log before transform
                    data_dict_i = {'image_path': '../addedArtifacts_S1/train/CT-PET-VI-16_T60_image_4_4_2_1.npz', 'seg_path': '../addedArtifacts_S1/train/CT-PET-VI-16_T60_maskArtifact_4_4_2_1.npz'}
                    monai_output = self.monai_transforms(data_dict_i)
                    # tqdm.write(f"DEBUG: MONAI transforms completed for {filename}") # Optional: Log after transform
                except Exception as transform_exc:
                    tqdm.write(f"CRITICAL_ERROR: Exception during MONAI transform for index {idx}, image: {img_path_str}, seg: {seg_path_str}. Error: {transform_exc}")
                    
                    skipped_count += 1
                    continue # Skip this sample on transform error
                # --- End wrap ---


                # --- Handle potential unexpected list return from Compose ---
                transformed_data = None
                if isinstance(monai_output, dict):
                    transformed_data = monai_output
                elif isinstance(monai_output, (list, tuple)) and len(monai_output) == 1 and isinstance(monai_output[0], dict):
                    # print(f"DEBUG: monai_transforms returned a list/tuple, taking first element ({filename})") # Keep if useful?
                    transformed_data = monai_output[0]
                else:
                    # Handle cases where it's neither dict nor list containing dict
                    print(f"ERROR: monai_transforms returned unexpected type: {type(monai_output)} ({filename})")
                    # Fall through, the key check below will fail and skip
                    transformed_data = monai_output # Assign anyway so the error check below triggers
                # --- End handling --- 

                # Check if we successfully got a dictionary
                if not (
                    isinstance(transformed_data, dict) and
                    "image" in transformed_data and
                    "seg" in transformed_data
                ):
                     # Log the actual type if it wasn't a dict
                     if not isinstance(transformed_data, dict):
                         tqdm.write(f"Warning: MONAI transforms did not return a dict for {filename}. Got type: {type(transformed_data)}. Skipping.")
                     else:
                         tqdm.write(f"Warning: MONAI transforms did not produce 'image' and 'seg' keys for {filename}. Keys found: {list(transformed_data.keys())}. Skipping.")
                     skipped_count += 1
                     continue

                # Ensure seg is numpy for checking
                seg_tensor = transformed_data['seg']
                seg_np = seg_tensor.cpu().numpy() if isinstance(seg_tensor, torch.Tensor) else np.asarray(seg_tensor)
                image_tensor = transformed_data['image']
                image_np = image_tensor.cpu().numpy() if isinstance(image_tensor, torch.Tensor) else np.asarray(image_tensor)

                artifact_pixels = np.count_nonzero(seg_np == 1) + np.count_nonzero(seg_np == 2)

                # --- ADDED: Debug visualization for small artifacts --- (Commenting out the call)
                if 0 < artifact_pixels < self.args.min_artifact_pixels: # Check if artifact count is small but non-zero
                    try:
                        output_dir_path = Path(self.args.output_dir)
                        # save_small_artifact_visualization(
                        #     image_crop=image_np,    # Should be CHWD (e.g., 1, H, W, D)
                        #     seg_crop=seg_np,      # Should be CHWD (e.g., 1, H, W, D)
                        #     artifact_count=artifact_pixels,
                        #     filename=filename,
                        #     output_dir=output_dir_path
                        # )
                    except AttributeError:
                         tqdm.write("Warning: Cannot save debug viz - self.args not found in DataLoader. Did you pass args during init?")
                    except Exception as viz_e:
                        tqdm.write(f"Warning: Failed to save debug visualization for {filename}: {viz_e}")
                # --- END ADDED ---

                # --- ADJUSTED Label Derivation ---
                image_label = 0 # Default to 0
                if artifact_pixels >= self.min_pixels_threshold:
                    # Only determine label 1 or 2 if threshold is met
                    if seg_np.shape[0] == 1:
                        seg_for_label = seg_np[0]
                    elif seg_np.ndim > 0 and seg_np.shape[0] > 1:
                        seg_for_label = seg_np[0] # Use first channel
                    else:
                        # Handle unexpected shape, but keep label 0
                        tqdm.write(f"Warning: Unexpected seg shape {seg_np.shape} for {filename} when checking for label 1/2. Assigning label 0.")
                        # image_label remains 0

                    # Use get_image_label with the actual threshold to differentiate 1 and 2
                    # (Technically min_pixels_threshold=1 would also work here since we already checked >= self.min_pixels_threshold,
                    # but using the actual threshold is clearer)
                    image_label = get_image_label(seg_for_label, min_pixels_threshold=self.min_pixels_threshold)
                # Else (artifact_pixels < self.min_pixels_threshold): image_label remains 0
                # --- END ADJUSTED Label Derivation ---

                # Add the sample to the batch lists (now includes low-artifact samples labeled 0)
                batch_images.append(image_np)
                batch_segs.append(seg_np)
                batch_labels.append(image_label) # Store 0, 1, or 2
                batch_filenames.append(filename)

            except FileNotFoundError as e:
                 tqdm.write(f"File not found error processing sample index {idx} ({data_dict_i.get('image_path', 'N/A')}): {e}. Skipping.")
                 skipped_count += 1
                 continue
            except Exception as e:
                tqdm.write(f"Error processing sample index {idx} ({data_dict_i.get('image_path', 'N/A')}): {e}")
                skipped_count += 1
                continue 


        if not batch_images:
            # If all samples were skipped, return an empty batch structure
            tqdm.write("Warning: generate_train_batch generated an empty batch after processing/filtering.")
            c, d, h, w = 1, 64, 136, 136 # Placeholder shape, adjust if needed
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
            label_batch_np = np.array(batch_labels, dtype=np.int64) # Labels are 0, 1, or 2 here
        except Exception as stack_e:
            tqdm.write(f"Error stacking batch data: {stack_e}")
            tqdm.write(f"Individual image shapes: {[img.shape for img in batch_images]}")
            tqdm.write(f"Individual seg shapes: {[seg.shape for seg in batch_segs]}")
            # Return empty batch on stacking error
            c, d, h, w = 1, 64, 136, 136 # Placeholder shape
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.float32),
                'label': np.empty((0,), dtype=np.int64),
                'filenames': [],
                'roi': np.empty((0, c, d, h, w), dtype=np.int64)
             }
        
        print(image_batch_np, flush=True)
        print(seg_batch_np, flush=True)
        print(label_batch_np, flush=True)

        return {
            'data': image_batch_np,
            'seg': seg_batch_np,
            'label': label_batch_np, # Labels are 0, 1, or 2
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


# --- Helper function (re-implementation of _get_scan_interval) --- ADDED ---
def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.
    Copied/adapted from monai.inferers.utils._get_scan_interval
    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"image_size len {len(image_size)} != spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"roi_size len {len(roi_size)} != spatial dims {num_spatial_dims}.")
    if len(overlap) != num_spatial_dims:
        raise ValueError(f"overlap len {len(overlap)} != spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)
# --- End Helper function --- ADDED ---


def main(args):
    log_memory("Main Start") # <<< Log Point 1
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
    print(f"Using {num_pairing_threads} threads for file pairing.")
    train_files_list, missing_train = find_and_pair_files(train_path, args.file_pattern, pre_filter_non_artifact=False, num_threads=num_pairing_threads)
    validate_files_list, missing_val = find_and_pair_files(val_path, args.file_pattern, pre_filter_non_artifact=False, num_threads=num_pairing_threads)
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

    # random_zoom_transform = RandomApply(rand_zoom, prob=0.2) # REMOVED - Using prob in RandZoomd directly

    base_transforms = [
        LoadPairedArr0d(keys=("image_path", "seg_path")),
        EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=[img_key, seg_key], axcodes="RAS", allow_missing_keys=True),
        Spacingd(keys=[img_key, seg_key], pixdim=args.target_spacing, mode=("nearest", "nearest")), # Changed bilinear to nearest for image
        
        RandCropByPosNegLabeld(
            keys=[img_key, seg_key],
            label_key=seg_key,
            spatial_size=args.input_size,
            pos=0.99, # PREFER foreground center (99%)
            neg=0.01, # ALLOW background center (1%) as fallback
            num_samples=1,
            allow_smaller=True # Allow smaller output if input is smaller than crop size
        ),
        Resized(keys=[img_key, seg_key], spatial_size=args.input_size, mode=("nearest", "nearest")),
        NormalizeIntensityd(keys=[img_key], subtrahend=0.412456, divisor=0.278396), # Apply Normalization AFTER spatial transforms
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


    log_memory("Before Train DataLoader Init") # <<< Log Point
    print("Setting up BatchGenerators training data loader...")
    train_dl = ArtifactClassificationDataLoader(
        data_dicts=train_files,
        batch_size=args.batch_size,
        monai_transforms=pre_aug_transforms,
        min_pixels_threshold=args.min_artifact_pixels,
        args=args # ADDED: Pass args
    )
    log_memory("After Train DataLoader Init") # <<< Log Point

    print("Setting up MONAI test data loader...")
    
    test_compose = Compose([
        LoadPairedArr0d(keys=("image_path", "seg_path")), # Load NPZ
        EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel"),
        Orientationd(keys=[img_key, seg_key], axcodes="RAS"),
        Spacingd(keys=[img_key, seg_key], pixdim=args.target_spacing, mode=("nearest", "nearest")), # Changed bilinear to nearest for image

        GetLabelFromSegd(keys=[seg_key], label_key='label', min_pixels_threshold=args.min_artifact_pixels),
        # First take a center crop, then resize if necessary to ensure exact dimensions
        CenterSpatialCropd(
            keys=[img_key, seg_key],
            roi_size=args.input_size
        ),
        # Ensure exact dimensions with resize to handle cases where input is smaller than crop size
        Resized(
            keys=[img_key, seg_key],
            spatial_size=args.input_size,
            mode=("nearest", "nearest")
        ),
        NormalizeIntensityd(keys=[img_key], subtrahend=0.412456, divisor=0.278396), # Normalize image after spacing/label derivation
        EnsureTyped(keys=[img_key, 'label'], dtype=(torch.float32, torch.int64)), # Ensure image is float, label is long
    ])

    test_ds = monai.data.Dataset(data=test_files, transform=test_compose) if test_files else None

    print("Getting batchgenerators transforms from Augmentations.py...")
    bg_transforms = get_augmentations()
    if bg_transforms is None:
         print("Warning: get_augmentations returned None, training loader will have no batchgenerators transforms.")

    log_memory("Before MultiThreadedAugmenter Init") # <<< Log Point
    print("Initializing MultiThreadedAugmenter for training...")
    seeds = [i for i in range(args.num_workers)]
    train_loader = MultiThreadedAugmenter(
        train_dl,
        transform=bg_transforms,
        num_processes=args.num_workers,
        num_cached_per_queue=2,
        pin_memory=True,
        seeds=seeds,
        useroi=False, # Disable ROI calculation
        generate_patches=False # Disable patch generation
    )
    print("MultiThreadedAugmenter initialized.")
    log_memory("After MultiThreadedAugmenter Init") # <<< Log Point


    test_loader = PyTorchDataLoader(
        test_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate # Use standard collate function instead of padding
    ) if test_ds else None
    print("Test DataLoader initialized.")

    if test_loader is None:
        print("Warning: Test DataLoader is None. Skipping test phase.")
        # return # Keep return commented out to allow testing memory usage without test phase

    log_memory("DataLoaders Initialized") # <<< Log Point 2

    batches_per_epoch = math.ceil(len(train_dl) / args.batch_size)
    print(f"Calculated batches per epoch: {batches_per_epoch}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    print(f"Model configured for {args.num_classes} output classes (0=Background, 1=Artifact1, 2=Artifact2)")
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=args.num_classes
    ).to(device)

    # --- Loss and Optimizer Setup ---
    if args.use_weighted_loss:
        # if len(args.class_weights) != num_output_classes: # OLD Check
        if len(args.class_weights) != args.num_classes:
             # raise ValueError(f"Expected {num_output_classes} class weights for 2-class output, but got {len(args.class_weights)}. Weights should correspond to original classes 1 and 2.")
             raise ValueError(f"Expected {args.num_classes} class weights (for Bkg, Art1, Art2), but got {len(args.class_weights)}.")

        weights = torch.tensor(args.class_weights).float().to(device)
        print(f"Using weighted CrossEntropyLoss for {args.num_classes} classes. Weights (Cls 0, 1, 2): {weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        # print("Using standard CrossEntropyLoss for 2 classes.")
        print(f"Using standard CrossEntropyLoss for {args.num_classes} classes.")
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
        best_metric = -1 # Best validation accuracy (3-class)
        best_metric_epoch = -1

    # --- Variables for tracking metrics between logs ---
    last_log_loss = np.nan
    last_log_acc = np.nan
    # --- End Variables ---

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        log_memory(f"Epoch {epoch} Start") # <<< Log Point 3
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
        recent_train_accuracies = [] # For moving average of 3-class accuracy
        recent_correct_counts = {c: [] for c in range(args.num_classes)}
        recent_total_counts = {c: [] for c in range(args.num_classes)}

        # --- ANSI Color Codes ---
        COLOR_GREEN = '\033[92m'
        COLOR_RED = '\033[91m'
        COLOR_RESET = '\033[0m'
        # --- End ANSI Color Codes ---

        # --- Define max steps per epoch ---
        MAX_STEPS_PER_EPOCH = 500
        # --- End Define ---

        # Wrap train_loader with islice and set total for tqdm
        progress_bar = tqdm(itertools.islice(train_loader, MAX_STEPS_PER_EPOCH), 
                            desc=f"Epoch {epoch} Train", unit="batch", 
                            leave=False, total=MAX_STEPS_PER_EPOCH)
        for batch_data in progress_bar:
            batch_accuracy_12 = 0 # Initialize for safety
            batch_accuracy = 0 # Initialize for 3-class accuracy
            try:
                batch_correct_mapped = 0
                batch_total_samples = 0
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
                # Labels are 0, 1, or 2 from loader
                if isinstance(label_batch, np.ndarray):
                    labels = torch.from_numpy(label_batch).long().to(device)
                elif isinstance(label_batch, torch.Tensor):
                    labels = label_batch.long().to(device)
                else:
                    tqdm.write(f"Warning: Unexpected type for label batch: {type(label_batch)}. Skipping batch.")
                    continue

                optimizer.zero_grad()
                outputs = model(inputs)
                # Loss expects outputs (B, 3) and labels (B,) containing 0, 1, or 2
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                train_loss += loss.item() # Accumulate total epoch loss
                train_steps += 1

                # --- Calculate Overall and Per-Class Accuracy ---
                _, predicted = torch.max(outputs.data, 1)
                batch_total_samples = labels.size(0)
                batch_correct = (predicted == labels).sum().item()

                # Accumulate total epoch correct/total samples
                train_correct += batch_correct
                train_total += batch_total_samples

                batch_correct_counts_cls = {}
                batch_total_counts_cls = {}
                for c in range(args.num_classes):
                    class_mask = (labels == c)
                    batch_total_counts_cls[c] = class_mask.sum().item()
                    batch_correct_counts_cls[c] = (predicted[class_mask] == labels[class_mask]).sum().item()

                if batch_total_samples > 0:
                    batch_accuracy = (batch_correct / batch_total_samples) * 100
                    recent_train_accuracies.append(batch_accuracy)
                    if len(recent_train_accuracies) > args.log_freq:
                        recent_train_accuracies.pop(0)

                # --- Update moving averages ---
                recent_train_losses.append(current_loss)
                if len(recent_train_losses) > args.log_freq:
                    recent_train_losses.pop(0)

                # Update per-class rolling counts
                for c in range(args.num_classes):
                    recent_correct_counts[c].append(batch_correct_counts_cls[c])
                    recent_total_counts[c].append(batch_total_counts_cls[c])
                    if len(recent_correct_counts[c]) > args.log_freq:
                        recent_correct_counts[c].pop(0)
                        recent_total_counts[c].pop(0)
                # --- End Update moving averages ---


                # --- Log every log_freq steps ---
                if train_steps > 0 and train_steps % args.log_freq == 0:
                    current_avg_loss = np.mean(recent_train_losses) if recent_train_losses else np.nan
                    current_avg_acc = np.mean(recent_train_accuracies) if recent_train_accuracies else np.nan

                    # Calculate rolling per-class accuracies
                    rolling_acc_cls = {}
                    for c in range(args.num_classes):
                        total_c = sum(recent_total_counts[c])
                        correct_c = sum(recent_correct_counts[c])
                        rolling_acc_cls[c] = (correct_c / total_c) * 100 if total_c > 0 else np.nan

                    # Format loss change
                    loss_diff_str = ""
                    if not np.isnan(last_log_loss) and not np.isnan(current_avg_loss):
                        loss_diff = current_avg_loss - last_log_loss
                        color = COLOR_GREEN if loss_diff < 0 else COLOR_RED
                        sign = '+' if loss_diff >= 0 else ''
                        loss_diff_str = f" ({color}{sign}{loss_diff:.4f}{COLOR_RESET})"
                    
                    # Format accuracy change
                    acc_diff_str = ""
                    if not np.isnan(last_log_acc) and not np.isnan(current_avg_acc):
                        acc_diff = current_avg_acc - last_log_acc
                        color = COLOR_GREEN if acc_diff > 0 else COLOR_RED
                        sign = '+' if acc_diff >= 0 else ''
                        acc_diff_str = f" ({color}{sign}{acc_diff:.2f}%{COLOR_RESET})"

                    # Calculate rolling per-class accuracies
                    acc_cls_str = ", ".join([f"Acc Cls{c}: {rolling_acc_cls[c]:.1f}%" for c in range(args.num_classes)])
                    print(f"Step {train_steps}: Loss: {current_avg_loss:.4f}{loss_diff_str}, Accuracy: {current_avg_acc:.2f}%{acc_diff_str} ({acc_cls_str})")

                    # Update last logged values
                    last_log_loss = current_avg_loss
                    last_log_acc = current_avg_acc

                    # Log to WandB if enabled
                    if wandb_enabled:
                        log_data = {
                            "train/step_loss_moving_avg": current_avg_loss,
                            "train/step_accuracy_moving_avg": current_avg_acc,
                            "epoch": epoch + (train_steps / MAX_STEPS_PER_EPOCH) # Use MAX_STEPS_PER_EPOCH
                        }
                        # Add per-class rolling accuracies to WandB log
                        for c in range(args.num_classes):
                            log_data[f"train/step_accuracy_cls{c}_moving_avg"] = rolling_acc_cls[c]

                        wandb.log(log_data, step=epoch * MAX_STEPS_PER_EPOCH + train_steps) # Use MAX_STEPS_PER_EPOCH

                    log_memory(f"Epoch {epoch} Train Step {train_steps}") # <<< Log Point 4 (Inside Train Loop)

                # Re-add TQDM postfix update
                if train_steps > 0:
                    # Revert to calculating moving averages directly for postfix
                    moving_avg_loss = np.mean(recent_train_losses) if recent_train_losses else np.nan
                    moving_avg_acc = np.mean(recent_train_accuracies) if recent_train_accuracies else np.nan
                    progress_bar.set_postfix(
                            loss=f"{moving_avg_loss:.4f}",
                            acc=f"{moving_avg_acc:.2f}%"
                        )

                # --- ADDED: Explicitly delete tensors from this training step ---
                del inputs, labels, outputs, loss, predicted
                if 'image_batch' in locals(): del image_batch # Just in case
                if 'label_batch' in locals(): del label_batch # Just in case
                # --- End ADDED ---

            except KeyError as e:
                 tqdm.write(f"Error: Missing key {e} in batch data from MultiThreadedAugmenter.")
                 tqdm.write(f"Batch keys: {batch_data.keys() if isinstance(batch_data, dict) else type(batch_data)}")
                 continue # Skip this batch
            except Exception as e:
                 tqdm.write(f"Error processing batch from MultiThreadedAugmenter: {e}")
                 continue # Skip this batch

        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}, Accuracy (3-class): {train_accuracy:.2f}% ({train_correct}/{train_total})")

        log_memory(f"Epoch {epoch} Train End") # <<< Log Point 5

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} Learning Rate: {current_lr:.6f}")


        # --- Validation/Testing Phase --- 
        model.eval()
        test_metrics = {"epoch": epoch}
        # Initialize lists to accumulate results
        accumulated_epoch_preds = []
        accumulated_epoch_labels = []
        epoch_test_loss = 0.0
        num_test_batches = 0
        MAX_TEST_BATCHES_PER_EPOCH = 16 # Keep the limit for now

        if test_loader:
            log_memory(f"Epoch {epoch} Test Start") # <<< Log Point 6
            print(f"Running Testing for Epoch {epoch} (max {MAX_TEST_BATCHES_PER_EPOCH} batches)...")

            with torch.no_grad():
                test_pbar = tqdm(itertools.islice(test_loader, MAX_TEST_BATCHES_PER_EPOCH),
                                 desc=f"Epoch {epoch} Test", unit="batch",
                                 leave=False, total=MAX_TEST_BATCHES_PER_EPOCH)
                
                for batch_data in test_pbar:
                    if "image" not in batch_data or "label" not in batch_data:
                        tqdm.write(f"Skipping test batch due to missing keys. Keys: {batch_data.keys()}")
                        continue
                        
                    inputs = batch_data["image"] # NCHWD
                    labels = batch_data["label"] # N

                    if isinstance(inputs, np.ndarray):
                        inputs = torch.from_numpy(inputs).float().to(device)
                    elif isinstance(inputs, torch.Tensor):
                        inputs = inputs.float().to(device)
                    else:
                        tqdm.write(f"Warning: Unexpected type for test image batch: {type(inputs)}. Skipping.")
                        continue
                    
                    if isinstance(labels, np.ndarray):
                        labels = torch.from_numpy(labels).long().to(device)
                    elif isinstance(labels, torch.Tensor):
                        labels = labels.long().to(device)
                    else:
                        tqdm.write(f"Warning: Unexpected type for test label batch: {type(labels)}. Skipping.")
                        continue
                    
                    # Direct forward pass on entire batch
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    
                    # Get predictions from logits
                    _, predicted = torch.max(outputs, 1)
                    
                    # Accumulate batch results
                    accumulated_epoch_preds.extend(predicted.cpu().numpy())
                    accumulated_epoch_labels.extend(labels.cpu().numpy())
                    
                    # Accumulate loss
                    epoch_test_loss += batch_loss.item()
                    num_test_batches += 1
                    
                    # Update progress
                    if num_test_batches > 0:
                        avg_loss_so_far = epoch_test_loss / num_test_batches
                        test_pbar.set_postfix(loss=f"{avg_loss_so_far:.4f}")

            # --- Calculate final metrics after the loop --- 
            log_memory(f"Epoch {epoch} Test Loop End") # <<< Log Point
            avg_test_loss = epoch_test_loss / num_test_batches if num_test_batches > 0 else 0
            test_accuracy = accuracy_score(accumulated_epoch_labels, accumulated_epoch_preds) * 100 if accumulated_epoch_labels else 0

            target_names = ["Background", "Artifact1", "Artifact2"]
            report = "N/A"
            cm = None
            log_memory(f"Epoch {epoch} Test Before Metrics Calc") # <<< Log Point
            if accumulated_epoch_labels: # Check if we accumulated any results
                try:
                    report = classification_report(accumulated_epoch_labels, accumulated_epoch_preds, target_names=target_names, zero_division=0, labels=range(args.num_classes))
                    cm = confusion_matrix(accumulated_epoch_labels, accumulated_epoch_preds, labels=range(args.num_classes))
                except ValueError as e:
                    print(f"Could not generate final classification report/CM: {e}")
            log_memory(f"Epoch {epoch} Test After Metrics Calc") # <<< Log Point

            # Print the final, epoch-wide metrics
            print(f"\n--- Epoch {epoch} Final Test Results ---")
            print(f"Average Test Loss: {avg_test_loss:.4f}")
            print(f"Test Accuracy (based on {len(accumulated_epoch_labels)} samples): {test_accuracy:.2f}%")
            test_metrics["test/epoch_avg_loss"] = avg_test_loss
            test_metrics["test/epoch_accuracy"] = test_accuracy

            if accumulated_epoch_labels and cm is not None:
                print("Final Test Classification Report:")
                print(report)
                print("Final Test Confusion Matrix:")
                print(cm)
            else:
                 print("No samples were included in the final metrics.")
            print("----------------------------------\n")

            # --- WandB Logging for Test (using accumulated results) --- #
            if wandb_enabled:
                log_memory(f"Epoch {epoch} Test Before WandB Log") # <<< Log Point
                try:
                    # Log CM to wandb using accumulated data
                    if accumulated_epoch_labels and cm is not None:
                        wandb_cm = wandb.Table(columns=["Actual", "Predicted", "nPredictions"], 
                                               rows=[[target_names[i], target_names[j], cm[i, j]] 
                                                     for i in range(len(target_names)) for j in range(len(target_names))])
                        test_metrics["test/confusion_matrix"] = wandb_cm
                    # Log other metrics calculated above
                except Exception as report_e:
                    print(f"Warning: Could not format detailed report/cm for WandB: {report_e}")
                wandb.log(test_metrics, step=epoch * MAX_STEPS_PER_EPOCH + MAX_STEPS_PER_EPOCH)
                log_memory(f"Epoch {epoch} Test After WandB Log") # <<< Log Point
            # --- End WandB Logging --- #

            # --- Logging to File (using accumulated results) --- #
            with open(log_file, 'a') as f:
                f.write(f"\n--- Epoch {epoch} Final Test Results ---\n")
                f.write(f"Average Test Loss: {avg_test_loss:.4f}\n")
                f.write(f"Test Accuracy (based on {len(accumulated_epoch_labels)} samples): {test_accuracy:.2f}%\n")
                if accumulated_epoch_labels and cm is not None:
                    f.write("Final Test Classification Report:\n")
                    f.write(f"{report}\n")
                    f.write("Final Test Confusion Matrix:\n")
                    f.write(f"{str(cm)}\n")
                else:
                    f.write("No samples were included in the final metrics.\n")
                f.write("----------------------------------\n")
            # --- End Logging to File --- #
            log_memory(f"Epoch {epoch} Test End & Logged") # <<< Log Point 8

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")

        if wandb_enabled:
             wandb.log(test_metrics, step=epoch * batches_per_epoch + batches_per_epoch)
             moving_avg_loss_epoch_end = np.mean(recent_train_losses) if recent_train_losses else avg_train_loss
             moving_avg_acc_epoch_end = np.mean(recent_train_accuracies) if recent_train_accuracies else train_accuracy
             wandb.log({
                 "train/epoch_loss": avg_train_loss,
                 "train/epoch_accuracy": train_accuracy,
                 "train/epoch_loss_moving_avg": moving_avg_loss_epoch_end,
                 "train/epoch_accuracy_moving_avg": moving_avg_acc_epoch_end,
                 "learning_rate": current_lr,
                 "epoch": epoch
             }, step=epoch * batches_per_epoch + batches_per_epoch)

        # --- Checkpoint saving logic --- Check test_accuracy variable name ---
        current_test_accuracy = test_metrics.get("test/epoch_accuracy", -1) # Use the updated accuracy metric name
        if test_loader and current_test_accuracy > best_metric:
            best_metric = current_test_accuracy
            best_metric_epoch = epoch
            print(f"New best test accuracy (3-class): {best_metric:.2f}% at epoch {epoch}. Saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'best_metric_epoch': best_metric_epoch,
                'args': vars(args)
            }, run_output_dir / "checkpoint_best.pt")

        # Save latest checkpoint
        print(f"Saving latest checkpoint for epoch {epoch}...")
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric, # Save current best metric info
            'best_metric_epoch': best_metric_epoch,
            'args': vars(args)
        }, run_output_dir / "checkpoint_latest.pt")
        # --- End Checkpoint saving logic --- ADD BACK ---

        # --- ADDED: Explicitly delete large vars from test phase --- 
        if 'accumulated_epoch_preds' in locals():
            del accumulated_epoch_preds
        if 'accumulated_epoch_labels' in locals():
            del accumulated_epoch_labels
        if 'test_metrics' in locals():
            del test_metrics
        if 'report' in locals():
            del report
        if 'cm' in locals():
            del cm
        if 'wandb_cm' in locals():
            del wandb_cm
        # --- End ADDED ---

        # --- ADDED: Optional explicit garbage collect ---
        import gc
        gc.collect()
        # --- End ADDED ---

        log_memory(f"Epoch {epoch} End & Cleaned") # <<< Log Point 9

    print(f"Training finished. Best Test Accuracy: {best_metric:.2f}% at epoch {best_metric_epoch}")
    print(f"Logs saved to: {log_file}")

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
    parser.add_argument('--input_size', type=int, nargs=3, default=[64, 192, 192], help='Input size for the network (D, H, W).')
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1, 1, 1], help='Target voxel spacing (x, y, z).')
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
    parser.add_argument('--min_artifact_pixels', type=int, default=3,
                        help='Minimum number of artifact voxels required in transformed mask to assign label 1 or 2.')

    # --- Loss Function Arguments ---
    parser.add_argument('--use_weighted_loss', action=argparse.BooleanOptionalAction, default=True,
                        help='Use weighted CrossEntropyLoss based on class distribution.')
    # Default weights example - ADJUST THESE based on your 3-class distribution!
    parser.add_argument('--class_weights', type=float, nargs=3, default=[1, 1, 1.0],
                        help='Weights for Background (0), Artifact1 (1), and Artifact2 (2) for weighted loss. Provide 3 values. Ignored if --no_use_weighted_loss.')

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
    if args.use_weighted_loss and len(args.class_weights) != args.num_classes:
        raise ValueError(f"--class_weights must provide exactly {args.num_classes} weights, but got {len(args.class_weights)}.")

    main(args) 