# train_segmentation.py
import argparse
import os
import time
import shutil
from pathlib import Path
import warnings
import pickle
import random
import math
import concurrent.futures
import itertools
from typing import Sequence, Union
import sys
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Use MONAI DataLoader for validation
from monai.data import DataLoader as MonaiDataLoader
from sklearn.model_selection import KFold # Keep KFold if used for splitting
from tqdm import tqdm

import torch
import torch.nn.functional as F
from monai.networks.utils import one_hot

import monai
from monai.data import (
    Dataset as MonaiDataset, # Use Alias for clarity
    CacheDataset,
    partition_dataset,
    load_decathlon_datalist,
    list_data_collate,
    pad_list_data_collate,
    decollate_batch,
    DistributedSampler,
    DistributedWeightedRandomSampler,
)
from monai.transforms.transform import MapTransform
from monai.transforms import (
    Compose, # Use Alias
    EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, RandRotate90d, RandGaussianNoised, EnsureTyped,
    ScaleIntensityRanged, RandCropByPosNegLabeld,
    NormalizeIntensityd,
    RandZoomd,
    CenterSpatialCropd,
)
# import pytorch dataloader
from torch.utils.data import DataLoader as PyTorchDataLoader # Alias PyTorch DataLoader

# Import the multi-task SwinUNETR (or a pure segmentation one if available)
try:
    from swin_unetr import SwinUNETR
except ImportError:
    print("ERROR: Cannot import SwinUNETR from swin_unetr.py.")
    sys.exit(1)

from monai.losses import DiceCELoss # Segmentation Loss
from monai.metrics import DiceMetric # Segmentation Metric
from monai.utils import set_determinism, ensure_tuple_rep
from monai.networks.utils import one_hot # Needed for DiceCELoss

# --- BatchGenerators Imports (Essential) ---
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose as BGCompose # Alias BG compose

# --- Augmentations Import ---

from Augmentations import get_augmentations # Should return BGCompose


# --- Other Imports (PIL, psutil, wandb) ---
try: from PIL import Image, ImageDraw, ImageFont
except ImportError: Image = None; print("Warning: PIL not found, debug viz disabled.")
try: import psutil; _process = psutil.Process(os.getpid())
except ImportError: psutil = None; print("Warning: psutil not found, memory logging disabled.")
try: import wandb
except ImportError: wandb = None; print("Warning: wandb not found.")

warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# --- Memory Logging Function ---
def log_memory(stage: str):
    if not psutil: return
    mem_info = _process.memory_info(); rss_gb = mem_info.rss / (1024**3); vms_gb = mem_info.vms / (1024**3)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S"); print(f"MEMLOG [{timestamp}] - {stage}: RSS={rss_gb:.2f} GB, VMS={vms_gb:.2f} GB")

# --- Base Loader Class (Copied *exactly* from classify_artifacts.py) ---
# This loader's generate_train_batch applies monai_transforms sequentially
# and returns a dict of NumPy arrays: {'data', 'seg', 'label', ...}
import torch.nn.functional as F # Add this import

# --- Base Loader Class (Manual Transforms) ---
class ArtifactClassificationDataLoader(DataLoader):
    """
    BatchGenerators compatible DataLoader.
    Loads paired image/segmentation NPZ files.
    Applies manual EnsureChannelFirst, RandCropByPosNegLabeld (logic),
    Resize, and NormalizeIntensity directly using PyTorch/NumPy.
    Derives classification labels and prepares batches for MultiThreadedAugmenter.
    """
    def __init__(self, data_dicts, batch_size, min_pixels_threshold, args, num_threads_in_multithreaded=1):
        super().__init__(data_dicts, batch_size, num_threads_in_multithreaded)
        # self.monai_transforms = monai_transforms # REMOVED
        self.min_pixels_threshold = min_pixels_threshold
        self.indices = list(range(len(data_dicts)))
        self.args = args # Store args to access input_size, normalization params etc.
        self.num_samples = len(data_dicts)

        # Store necessary parameters from args
        self.input_size = ensure_tuple_rep(self.args.input_size, 3) # Ensure tuple (D, H, W)
        self.pos_fraction = 0.99 # From RandCropByPosNegLabeld params used before
        self.norm_subtrahend = 0.412456 # From NormalizeIntensityd
        self.norm_divisor = 0.278396    # From NormalizeIntensityd

    def __len__(self):
        # MONAI Compose returns list of dicts,DataLoader expects list of dicts
        # return self.num_samples
        return len(self._data) # Use length of underlying data list

    def _get_crop_slice(self, image_shape, center, crop_size):
        """Calculates slices for cropping, handling boundaries."""
        slices = []
        # image_shape is C, D, H, W, crop_size is D, H, W, center is D, H, W
        # We operate on spatial dims D, H, W (indices 1, 2, 3 of shape)
        for i in range(3): # D, H, W
            dim_shape = image_shape[i+1]
            dim_crop_size = crop_size[i]
            dim_center = center[i]

            start = max(0, dim_center - dim_crop_size // 2)
            end = start + dim_crop_size

            if end > dim_shape:
                # Shift crop window left if it exceeds right boundary
                end = dim_shape
                start = max(0, end - dim_crop_size)

            slices.append(slice(start, end))
        # Return slices for spatial dimensions only
        return tuple(slices) # (slice_D, slice_H, slice_W)

    def generate_train_batch(self):
        indices = self.get_indices()
        batch_images = []
        batch_segs = []
        batch_filenames = []
        skipped_count = 0

        for idx in indices:
            data_dict_i = self._data[idx]
            img_path_str = data_dict_i.get('image_path', 'unknown_image')
            seg_path_str = data_dict_i.get('seg_path', 'unknown_seg')
            filename = Path(img_path_str).name

            try:
                # 1. Load NumPy arrays directly
                with np.load(img_path_str) as img_npz:
                    if 'arr_0' not in img_npz: raise KeyError("arr_0 not in image")
                    image_np = img_npz['arr_0'].astype(np.float32) # Ensure float32
                with np.load(seg_path_str) as seg_npz:
                    if 'arr_0' not in seg_npz: raise KeyError("arr_0 not in seg")
                    # Keep seg as int/uint8 for label finding, maybe convert later if needed
                    seg_np = seg_npz['arr_0'].astype(np.uint8)

                # Skip volumes containing class 2 by resampling until we get a valid one (maintains batch size)
                while np.any(seg_np == 2):
                    data_dict_i = self._data[random.randint(0, len(self._data) - 1)]
                    img_path_str = data_dict_i.get('image_path', 'unknown_image')
                    seg_path_str = data_dict_i.get('seg_path', 'unknown_seg')
                    filename = Path(img_path_str).name
                    # Reload image and segmentation
                    with np.load(img_path_str) as img_npz:
                        if 'arr_0' not in img_npz: raise KeyError("arr_0 not in image")
                        image_np = img_npz['arr_0'].astype(np.float32)
                    with np.load(seg_path_str) as seg_npz:
                        if 'arr_0' not in seg_npz: raise KeyError("arr_0 not in seg")
                        seg_np = seg_npz['arr_0'].astype(np.uint8)

                seg_np[seg_np == 4] = 0
                seg_np[seg_np == 3] = 0
                # 2. Manual EnsureChannelFirst (Assuming DHW input -> CDHW)
                if image_np.ndim == 3: image_np = image_np[None, ...] # Add channel dim
                if seg_np.ndim == 3: seg_np = seg_np[None, ...]       # Add channel dim
                if image_np.shape[0] != 1 or seg_np.shape[0] != 1:
                     tqdm.write(f"Warning: Unexpected channel count after load/add for {filename}. Img: {image_np.shape}, Seg: {seg_np.shape}. Skipping.")
                     skipped_count += 1; continue

                # 3. Convert to Tensor for Torch operations
                # Move to CPU explicitly, although default for from_numpy
                img_tensor = torch.from_numpy(image_np).cpu()
                seg_tensor = torch.from_numpy(seg_np).cpu() # Keep as uint8 for now

                # 4. Manual RandCropByPosNegLabeld Logic
                crop_size = self.input_size # Target spatial size (D, H, W)
                img_shape_spatial = img_tensor.shape[1:] # Spatial shape (D, H, W)

                # Find foreground centers (pixels == 1 or 2)
                foreground_indices = torch.nonzero((seg_tensor[0] == 1) | (seg_tensor[0] == 2)) # Indices are relative to spatial dims D, H, W

                center = None
                if foreground_indices.shape[0] > 0 and random.random() < self.pos_fraction:
                    # Positive sample: pick a random foreground voxel as center
                    center_idx = random.randint(0, foreground_indices.shape[0] - 1)
                    # Center coords are D, H, W
                    center = foreground_indices[center_idx].tolist() # D, H, W
                else:
                    # Negative sample (or no foreground): pick random center in image
                    center = [random.randint(0, s - 1) for s in img_shape_spatial] # D, H, W

                # Calculate crop slices, handling boundaries
                # Slices need to be applied to spatial dims (1, 2, 3)

                spatial_slices = self._get_crop_slice(img_tensor.shape, center, crop_size)

                # Apply crop - index channel 0, then apply spatial slices
                img_cropped = img_tensor[0, spatial_slices[0], spatial_slices[1], spatial_slices[2]]
                seg_cropped = seg_tensor[0, spatial_slices[0], spatial_slices[1], spatial_slices[2]]

                # Add channel dim back: CDHW -> D H W -> C D H W
                img_cropped = img_cropped.unsqueeze(0)
                seg_cropped = seg_cropped.unsqueeze(0)

                # 5. Manual Resize (Interpolate)
                # F.interpolate needs NCDHW input
                # Current shape is CDHW, add N dim
                img_for_resize = img_cropped.unsqueeze(0).float() # Ensure float for interpolate
                seg_for_resize = seg_cropped.unsqueeze(0).float() # Interpolate needs float

                img_resized = F.interpolate(img_for_resize, size=self.input_size, mode='nearest') # Using 'nearest' as per last example
                seg_resized = F.interpolate(seg_for_resize, size=self.input_size, mode='nearest') # Must use 'nearest' for masks

                # Remove N dim, back to CDHW
                img_resized = img_resized.squeeze(0)
                seg_resized = seg_resized.squeeze(0)

                # Convert seg back to int type after interpolation
                seg_resized = seg_resized.byte() # Or .long() depending on downstream needs, byte/uint8 often used

                # 6. Manual Normalize Intensity
                img_normalized = (img_resized - self.norm_subtrahend) / self.norm_divisor

                # 7. Convert final image tensor to NumPy for batchgenerators
                image_np_final = img_normalized.cpu().numpy()

                # --- Append results ---
                batch_images.append(image_np_final)
                batch_segs.append(seg_resized.cpu().numpy()) # Append final processed seg
                batch_filenames.append(filename)

            except FileNotFoundError as e:
                tqdm.write(f"File not found error: {e}. Skipping index {idx}.")
                skipped_count += 1
                continue
            except KeyError as e:
                tqdm.write(f"Key error loading {filename}: {e}. Skipping index {idx}.")
                skipped_count += 1
                continue
            except Exception as e:
                import traceback
                tqdm.write(f"Generic error processing index {idx} ({filename}): {e}\n{traceback.format_exc()}")
                skipped_count += 1
                continue

        # --- Batch Assembly (Remains mostly the same) ---
        if not batch_images:
            tqdm.write("Warning: generate_train_batch generated an empty batch after processing/filtering.")
            # Use target input_size for placeholder shape
            c, d, h, w = 1, self.input_size[0], self.input_size[1], self.input_size[2]
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.uint8), # Match output type
                'roi': np.empty((0, c, d, h, w), dtype=np.int64)
             }

        try:
            image_batch_np = np.stack(batch_images, axis=0)
            # Ensure seg batch matches expected type (e.g., uint8)
            seg_batch_np = np.stack(batch_segs, axis=0).astype(np.uint8)
        except Exception as stack_e:
            tqdm.write(f"Error stacking batch data: {stack_e}")
            tqdm.write(f"Individual image shapes: {[img.shape for img in batch_images]}")
            tqdm.write(f"Individual seg shapes: {[seg.shape for seg in batch_segs]}")
            c, d, h, w = 1, self.input_size[0], self.input_size[1], self.input_size[2]
            # Return empty batch on stacking error
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.uint8),
                'roi': np.empty((0, c, d, h, w), dtype=np.int64)
             }

        # Return dict required by MTA -> Training Loop
        return {
            'data': image_batch_np,
            'seg': seg_batch_np,
            'roi': np.ones_like(seg_batch_np, dtype=np.int64), # Keep ROI for compatibility maybe?
            'filenames': batch_filenames
        }

# --- Custom Transform for NPZ Loading (Copied) ---
class LoadPairedArr0d(MapTransform):
    """
    Custom transform to load image and segmentation from separate NPZ files under keys 'arr_0'.
    Outputs arrays under 'image' and 'seg'.
    """
    def __init__(self, keys=("image_path", "seg_path"), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img_path = d.get("image_path")
        seg_path = d.get("seg_path")
        if img_path is None or seg_path is None:
            raise KeyError("Input must contain 'image_path' and 'seg_path'.")
        try:
            img_npz = np.load(img_path)
            if 'arr_0' not in img_npz:
                raise KeyError(f"'arr_0' missing in image {img_path}")
            d['image'] = img_npz['arr_0']; img_npz.close()
            seg_npz = np.load(seg_path)
            if 'arr_0' not in seg_npz:
                raise KeyError(f"'arr_0' missing in seg {seg_path}")
            d['seg'] = seg_npz['arr_0']; seg_npz.close()
            del d['image_path']; del d['seg_path']
        except Exception as e:
            print(f"Error loading NPZ: {e}")
            raise e
        return d

def checkpoint(run_output_dir, device, model, optimizer, scheduler):
    # --- Checkpoint Loading (Similar, track Dice) ---
    start_epoch = 0; best_metric = 0.0; best_metric_epoch = -1 # Use Dice score now
    resume_path = Path(args.resume) if args.resume else run_output_dir / "checkpoint_latest.pt"
    if resume_path.is_file():
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', 0.0) # Default best Dice is 0
            best_metric_epoch = checkpoint.get('best_metric_epoch', -1)
            print(f" Resumed from Epoch {start_epoch}. Previous best Dice: {best_metric:.4f}")
        except Exception as e: print(f"Error loading checkpoint state: {e}. Starting scratch.")
    return model, optimizer, scheduler, start_epoch, best_metric, best_metric_epoch

def initialize_model(args, device, run_output_dir):
    # Build model with segmentation head only (num_classification_outputs defaults to 1)
    model = SwinUNETR(
        img_size=args.input_size,
        in_channels=1,
        out_channels=args.num_seg_classes,
        feature_size=args.feature_size,
        use_v2=True,
    ).to(device)
    
    # Simple print of model initialization
    print(f"Model initialized with {args.num_seg_classes} segmentation classes")
    
    gc.collect()
    if args.optimizer.lower() == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd': optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    else: raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    print(f"Using {args.optimizer} optimizer: LR={args.initial_lr}, WD={args.weight_decay}")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.scheduler_T0,
        T_mult=args.scheduler_T_mult,
        eta_min=args.min_lr
    )
    model, optimizer, scheduler, start_epoch, best_metric, best_metric_epoch = checkpoint(run_output_dir, device, model, optimizer, scheduler)
    
    if not start_epoch:
        start_epoch = 0
        # initialize best_metric high for validation loss minimization
        best_metric = float('inf')
        best_metric_epoch = -1
    else:
        print(f"Resumed from Epoch {start_epoch}. Previous best Dice: {best_metric:.4f}")
    return model, optimizer, scheduler, start_epoch, best_metric, best_metric_epoch




# --- Main Function (Adapted for Segmentation) ---
def main(args):
    log_memory("Main Start") # <<< Log Point 1
    set_determinism(seed=args.seed)
    output_dir = Path(args.output_dir)
    if args.run_name:
        run_name = args.run_name
    else:
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
    
    # --- MONAI Transforms ---
    img_key = "image"
    seg_key = "seg" # Keys used internally by transforms

    # Training transforms (pipeline passed to ArtifactClassificationDataLoader)
    # Should output {'image': Tensor, 'seg': Tensor}
    base_transforms = [
        LoadPairedArr0d(keys=("image_path", "seg_path")),
        EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel", allow_missing_keys=True),
        Orientationd(keys=[img_key, seg_key], axcodes="RAS", allow_missing_keys=True),
        Spacingd(keys=[img_key, seg_key], pixdim=args.target_spacing, mode=("nearest", "nearest")),
        RandCropByPosNegLabeld(keys=[img_key, seg_key], label_key=seg_key, spatial_size=args.input_size, pos=0.99, neg=0.01, num_samples=1, allow_smaller=True),
        Resized(keys=[img_key, seg_key], spatial_size=args.input_size, mode=("nearest", "nearest")),
        # Add intensity augs
        NormalizeIntensityd(keys=[img_key], subtrahend=0.412456, divisor=0.278396),
    ]
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
        min_pixels_threshold=args.min_artifact_pixels,
        args=args # ADDED: Pass args
    )
    log_memory("After Train DataLoader Init") # <<< Log Point

    print("Setting up MONAI test data loader...")
    
    test_compose = Compose([
        LoadPairedArr0d(keys=("image_path", "seg_path")),
        EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel", allow_missing_keys=True),
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
    ])

    test_ds = monai.data.Dataset(data=test_files, transform=test_compose) if test_files else None
    
    print("Getting batchgenerators transforms from Augmentations.py...")
    bg_transforms = get_augmentations()
    if bg_transforms is None:
         print("Warning: get_augmentations returned None, training loader will have no batchgenerators transforms.")

    log_memory("Before MultiThreadedAugmenter Init") # <<< Log Point
    print("Initializing MultiThreadedAugmenter for training...")
    train_loader = MultiThreadedAugmenter(
        train_dl,
        transform=augmentations_result,
        num_processes=args.num_workers,
        num_cached_per_queue=2,
        pin_memory=True,
        useroi=False # Disable ROI calculation
    )
    print("MultiThreadedAugmenter initialized.")
    log_memory("After MultiThreadedAugmenter Init") # <<< Log Point

    train_loader.next()
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

    # --- Model Setup (CHANGED to SwinUNETR) ---
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    print(f"Using device: {device}")
    print(f"Initializing SwinUNETR for {args.num_seg_classes} segmentation classes.")
    
    # --- Loss and Optimizer (CHANGED for Binary Segmentation) ---
    print(f"Using DiceCELoss with lambda_dice={args.lambda_dice}, lambda_ce={args.lambda_ce}")
    # DiceCELoss = Dice + lambda_ce * CrossEntropy; scale entire loss by lambda_dice
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, lambda_ce=args.lambda_ce).to(device)

    # --- Metrics ---
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    print(f"Starting segmentation training for {args.epochs} epochs...")
    model, optimizer, scheduler, start_epoch, best_metric, best_metric_epoch = initialize_model(args, device, run_output_dir)
    
    # --- Training Loop (ADAPTED for Single-Class Segmentation) ---
    for epoch in range(start_epoch, args.epochs):
        log_memory(f"Epoch {epoch} Start")
        model.train(); train_loss = 0.0; train_steps = 0
        epoch_start_time = time.time(); print("-" * 10); print(f"Epoch {epoch}/{args.epochs - 1}")
        recent_train_losses = []
        MAX_STEPS_PER_EPOCH = 1000  # cap training steps per epoch
        # update foreground cropping probability via tanh, plateau at 0.5 by epoch 25
        mid_pf = 25.0
        scale_pf = 5.0
        new_pos = 0.5 + 0.5 * math.tanh((mid_pf - epoch) / scale_pf)
        # clamp probability to [0.5, 1.0]
        train_dl.pos_fraction = float(min(1.0, max(0.5, new_pos)))
        print(f"Foreground sampling prob set to: {train_dl.pos_fraction:.3f}")

        # Wrap train_loader with islice and set total for tqdm
        progress_bar = tqdm(
            itertools.islice(train_loader, MAX_STEPS_PER_EPOCH),
            desc=f"Epoch {epoch} Train", unit="batch",
            leave=False, total=MAX_STEPS_PER_EPOCH,
            dynamic_ncols=True
        )
        for batch_data in progress_bar:
            # Expecting 'data', 'seg', 'label' (label might be ignored)
            if not isinstance(batch_data, dict) or 'data' not in batch_data or 'seg' not in batch_data or batch_data['data'].size == 0:
                print(f"Training: Batch data keys: {batch_data.keys()}")
                continue # Skip incomplete batch
            train_steps += 1

            inputs = torch.tensor(batch_data['data']).to(device).float()
            seg_targets = torch.tensor(batch_data['seg']).to(device).long() # shape [B,1,D,H,W]
            
            optimizer.zero_grad()
            try:
                outputs = model(inputs)
                seg_logits = outputs[0] if isinstance(outputs, tuple) else outputs  # [B,1,D,H,W]
            except Exception as e:
                print(f"Forward pass error: {e}")
                continue

            try:
                # compute combined Dice+CE loss; seg_targets has shape [B,1,D,H,W]
                loss = args.lambda_dice * criterion(seg_logits, seg_targets)
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Loss/backward error: {e}")
                continue

            current_loss = loss.item(); train_loss += current_loss
            recent_train_losses.append(current_loss)
            if len(recent_train_losses) > 100: recent_train_losses.pop(0) # Keep last 100
            avg_recent_loss = np.mean(recent_train_losses)
            
            # update progress display with current loss and learning rate
            progress_bar.set_postfix(
                loss=f"{avg_recent_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}"
            )

            # Insert debugging visualization every 10 batches, drop class2 and slice along y-axis
            if False and train_steps % 10 == 0:
                try:
                    import matplotlib.pyplot as plt
                    # compute segmentation probabilities and binary mask for class1
                    pred_prob = torch.sigmoid(seg_logits)  # [B,1,D,H,W]
                    # select first sample
                    img_vol = inputs[0, 0].detach().cpu().numpy()      # [D, H, W]
                    prob_vol = pred_prob[0, 0].detach().cpu().numpy()       # [D, H, W]
                    true_vol = seg_targets[0, 0].detach().cpu().numpy()     # [D, H, W]
                    pred_bin = (prob_vol >= 0.5).astype(np.uint8)
                    # compute Dice between binary pred and true mask
                    def dice_score(pred_mask, true_mask):
                        inter = (pred_mask & true_mask).sum()
                        denom = pred_mask.sum() + true_mask.sum()
                        return (2. * inter / denom) if denom > 0 else 1.0
                    dice1 = dice_score(pred_bin, (true_vol == 1).astype(np.uint8))
                    # choose y-axis slice (axis -2) with most targets
                    pixel_counts = (true_vol > 0).sum(axis=(0,2))
                    if pixel_counts.sum() > 0:
                        slice_idx = int(pixel_counts.argmax())
                    else:
                        slice_idx = img_vol.shape[1] // 2
                    # extract plane along y-axis (D x W)
                    img_slice = img_vol[:, slice_idx, :]
                    true_slice = (true_vol == 1)[:, slice_idx, :]
                    pred_slice = pred_bin[:, slice_idx, :]
                    # plot with axes labels
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    # raw input
                    axes[0].imshow(img_slice, cmap='gray', origin='lower')
                    axes[0].set_title('Input (Y-slice)')
                    axes[0].set_ylabel('Depth (Z)')
                    axes[0].set_xlabel('Width (X)')
                    # overlay true mask class1
                    axes[1].imshow(img_slice, cmap='gray', origin='lower')
                    axes[1].imshow(true_slice == 1, cmap='Reds', alpha=0.5, origin='lower')
                    axes[1].set_title('True Mask (class1)')
                    axes[1].set_ylabel('Depth (Z)')
                    axes[1].set_xlabel('Width (X)')
                    # predicted class1 mask
                    axes[2].imshow(pred_slice, cmap='Reds', origin='lower', alpha=0.5)
                    axes[2].set_title(f'Pred Class1 (dice {dice1:.3f})')
                    axes[2].set_ylabel('Depth (Z)')
                    axes[2].set_xlabel('Width (X)')
                    fig.suptitle(f'Epoch {epoch} Batch {train_steps}')
                    # save figure
                    debug_dir = run_output_dir / 'debug_vis'
                    debug_dir.mkdir(exist_ok=True)
                    fig_path = debug_dir / f'epoch{epoch}_batch{train_steps}.png'
                    plt.savefig(fig_path)
                    plt.close(fig)
                except Exception as e:
                    print(f"Debug visualization error: {e}")

            # Optional: Log step loss to WandB
            if wandb_enabled and train_steps % args.log_freq == 0:
                 wandb_data = {"train/step_loss": current_loss, "train/step_loss_avg100": avg_recent_loss}
                 wandb.log(wandb_data, step=epoch * MAX_STEPS_PER_EPOCH + train_steps)

        # End of Epoch Summary
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f"\n--- Epoch {epoch} Training Summary ---")
        print(f"Avg Train Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} LR: {current_lr:.6f}")

        # --- Validation Phase (ADAPTED for Segmentation) ---
        model.eval()
        val_loss = 0.0; val_steps = 0
        dice_metric.reset()
        
        if test_loader:
            log_memory(f"Epoch {epoch} Val Start")
            print(f"Running Validation for Epoch {epoch}...")
            with torch.no_grad():
                MAX_STEPS_PER_EPOCH = 200
                val_pbar = tqdm(
                    itertools.islice(test_loader, MAX_STEPS_PER_EPOCH),
                    desc=f"Epoch {epoch} Validate", unit="batch",
                    leave=False, total=MAX_STEPS_PER_EPOCH,
                    dynamic_ncols=True
                )
                for batch_data in val_pbar:
                    # Validation loader outputs Tensors: {'image', 'seg'} (or maybe 'seg_target' if not changed)
                    if 'image' not in batch_data or 'seg' not in batch_data: 
                        print(f"Test: keys in batch_data: {batch_data.keys()}")
                        continue # Adjust keys based on val_transforms output

                    inputs = torch.tensor(batch_data['image']).to(device).float()
                    seg_targets = torch.tensor(batch_data['seg']).to(device).long() # shape [B,1,D,H,W]
                    
                    val_steps += 1

                    try:
                        outputs = model(inputs)
                        seg_logits = outputs[0] if isinstance(outputs, tuple) else outputs  # [B,1,D,H,W]
                        
                        # Simple debug to check dimensions
                        if val_steps < 3 or val_steps % 50 == 0:
                            print(f"Batch {val_steps} seg_logits shape: {seg_logits.shape}")
                            
                            # Check segmentation logits for each class
                            if isinstance(seg_logits, torch.Tensor) and seg_logits.dim() > 1:
                                num_channels = seg_logits.shape[1]
                                print(f"Seg logits have {num_channels} channels")
                                for c in range(min(num_channels, 1)):
                                    channel = seg_logits[:, c]
                                    print(f"  Class {c} logits: min={channel.min().item():.4f}, max={channel.max().item():.4f}, mean={channel.mean().item():.4f}")
                                
                                # compute probabilities with sigmoid for binary segmentation
                                probs = torch.sigmoid(seg_logits)
                                channel = probs[:, 0]
                                print(f"  Class 0 probs: min={channel.min().item():.4f}, max={channel.max().item():.4f}, mean={channel.mean().item():.4f}")
                    except Exception as e: print(f"Validation inference error: {e}"); continue

                    try:  # Validation Loss (match training loss formula)
                         # combined Dice+CE loss for validation; seg_targets has channel dimension
                         loss_val = args.lambda_dice * criterion(seg_logits, seg_targets)
                         val_loss += loss_val.item()
                    except Exception as e: print(f"Validation loss error: {e}"); continue

                    try: # Validation Metric
                         # Compute Dice with monai metric
                         try:
                             # Compute dice using MONAI's metric
                             dice_metric(y_pred=seg_logits, y=seg_targets)
                         except Exception as dice_err:
                             print(f"Error computing dice metric: {dice_err}")
                             import traceback
                             print(traceback.format_exc())
                         
                    except Exception as e: 
                        print(f"Validation metrics error: {e}")
                        continue

            # Aggregate Validation Metrics
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            
            # Get dice scores from MONAI metric - with reduction="mean_batch", this returns
            # a tensor with shape [num_classes] where each element is the mean Dice score for that class
            if val_steps > 0:
                # Get aggregated dice scores
                dice_scores = dice_metric.aggregate().cpu().numpy()
                # Reset metric for next epoch
                dice_metric.reset()
                
                # Calculate mean dice across classes (excluding background if desired)
                # For now we include all classes
                mean_dice = np.mean(dice_scores)
                
                # Store for metrics tracking, etc.
                metric_val = float(mean_dice)
            else:
                dice_scores = np.zeros(1)
                metric_val = 0.0
            
            log_memory(f"Epoch {epoch} Val End")
            
            print(f"\n--- Epoch {epoch} Validation Summary ---")
            print(f"Avg Val Loss: {avg_val_loss:.4f}, Mean Dice: {metric_val:.4f}")
            
            # Segmentation metrics report
            print("\n=== SEGMENTATION REPORT ===")
            # Report Dice score for foreground (class 1)
            if len(dice_scores) > 0:
                print(f"Dice score: {dice_scores[0]:.4f}")
            
            # Add class distribution in ground truth segmentation masks
            try:
                # Sample a subset of validation files
                sample_size = min(50, len(test_files))
                sample_files = random.sample(test_files, sample_size)
                
                class_pixel_counts = np.zeros(1)
                total_pixels = 0
                
                for i, sample in enumerate(tqdm(sample_files, desc="Analyzing validation masks")):
                    try:
                        seg_data = np.load(sample['seg_path'])['arr_0']
                        
                        # Remap classes to match training
                        seg_data_remapped = seg_data.copy()
                        seg_data_remapped[seg_data_remapped == 4] = 0
                        seg_data_remapped[seg_data_remapped == 3] = 0
                        
                        # Count pixels per class
                        for c in range(1):
                            class_pixel_counts[c] += np.sum(seg_data_remapped == c)
                        total_pixels += seg_data_remapped.size
                    except Exception as e:
                        print(f"Error analyzing {sample['seg_path']}: {e}")
                
                # Report class distributions
                print(f"\nAnalyzed {sample_size} validation masks")
                for c in range(1):
                    percentage = (class_pixel_counts[c] / total_pixels) * 100 if total_pixels > 0 else 0
                    print(f"  Class {c}: {class_pixel_counts[c]:.0f} pixels ({percentage:.4f}%)")
            except Exception as e:
                print(f"Error analyzing validation masks: {e}")
            
            # Class distribution in validation set (image-level presence)
            print("\nClass Distribution in Validation Set (Image Level):")
            # compute presence counts for background (0) and foreground (1)
            num_samples = seg_targets.shape[0]
            class_presence_counts = {0: 0, 1: 0}
            # seg_targets shape [B,1,D,H,W], squeeze to [B,D,H,W]
            for sample in seg_targets.squeeze(1):
                # find unique classes in this sample
                unique_cls = torch.unique(sample).cpu().tolist()
                for c in unique_cls:
                    if c in class_presence_counts:
                        class_presence_counts[c] += 1

            for cls, count in class_presence_counts.items():
                percentage = (count / num_samples) * 100 if num_samples > 0 else 0
                print(f"Class {cls}: {count} samples ({percentage:.2f}%)")

            if wandb_enabled:
                wandb_data = {
                    "val/epoch_loss": avg_val_loss, 
                    "val/epoch_mean_dice": metric_val, 
                    "epoch": epoch
                }
                
                # Add MONAI Dice scores if available
                if val_steps > 0 and len(dice_scores) > 0:
                    # Log overall mean dice
                    wandb_data["val/mean_dice"] = metric_val
                    
                    # Log per-class dice scores
                    for i in range(min(1, len(dice_scores))):
                        wandb_data[f"val/dice_class_{i}"] = dice_scores[i]
                
                wandb.log(wandb_data, step=(epoch+1)*MAX_STEPS_PER_EPOCH -1)

            # --- Checkpointing (Based on Average Val Loss) ---
            is_best = avg_val_loss < best_metric
            if is_best: best_metric = avg_val_loss; best_metric_epoch = epoch
            print(f" Current Val Loss: {avg_val_loss:.4f}, Best Val Loss: {best_metric:.4f} at Epoch {best_metric_epoch}")
            latest_save_path = run_output_dir / 'checkpoint_latest.pt'
            chkpt = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_metric': best_metric, 'best_metric_epoch': best_metric_epoch}
            torch.save(chkpt, latest_save_path)
            if is_best: torch.save(chkpt, run_output_dir / 'checkpoint_best.pt'); print(" Best Checkpoint Saved.")
        else:
            print("Skipping validation: No validation loader.")

        print(f"Epoch {epoch} completed in {(time.time() - epoch_start_time):.2f} seconds.")
        print("-" * 10 + "\n")
        log_memory(f"Epoch {epoch} End")
        # Optional GC
        # import gc; gc.collect(); torch.cuda.empty_cache()

    print(f"Training finished. Best Validation Mean Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"Logs saved to: {log_file}")
    if wandb_enabled: wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SwinUNETR for Segmentation (Adapted from classify_artifacts)")
    # --- Arguments (Keep relevant ones, adjust defaults/names) ---
    parser.add_argument('dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./results_segmentation') # Changed default
    parser.add_argument('--file_pattern', type=str, default='*.np[yz]')
    parser.add_argument('--num_workers', type=int, default=12, help="Num workers for MTA")
    parser.add_argument('--num_workers_val', type=int, default=4, help="Num workers for MONAI val loader")

    parser.add_argument('--num_seg_classes', type=int,default=1, required=False, help="Number of segmentation output classes (e.g., 3 for Bkg, Art1, Art2).")
    parser.add_argument('--input_size', type=int, nargs=3, default=[64, 192+32, 192+64])
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--feature_size', type=int, default=24, help='SwinUNETR feature size.') # Added if needed

    parser.add_argument('--initial_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scheduler_T0', type=int, default=10) # Added scheduler args
    parser.add_argument('--scheduler_T_mult', type=int, default=2)
    parser.add_argument('--lambda_ce', type=float, default=1.0)
    parser.add_argument('--lambda_dice', type=float, default=1.0)

    # Removed classification specific args like --num_classes, --use_weighted_loss, --class_weights
    parser.add_argument('--min_artifact_pixels', type=int, default=3, help='Min pixels for deriving label in loader (can be ignored by loss).')
    parser.add_argument('--foreground_prob', type=float, default=0.9, help='Prob. RandCrop foreground.') # Kept if RandCrop used

    parser.add_argument('--device', type=str, default='0', help="GPU device ID ('0', 'cpu', etc).") # Allow 'cpu'
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--train_steps_per_epoch', type=int, default=500) # Added

    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='ArtifactSegmentation') # Changed default project
    parser.add_argument('--run_name', type=str, default=None, help="Name of existing run to resume, overrides default run name.")

    args = parser.parse_args()

    # Make args accessible globally if needed by loader/transforms (like classify_artifacts)
    global cli_args
    cli_args = args

    # Basic validation
    # Allow single-class segmentation
    # (args.num_seg_classes may be unused downstream now)

    main(args)