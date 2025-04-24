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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # Can remove later
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

# --- Exponential Moving Average for Class Accuracy ---
class ClassAccuracyEMA:
    """
    Tracks exponential moving average accuracy for specific classes.
    """
    def __init__(self, classes_to_track=[1, 2], alpha=0.95):
        """
        Initialize EMA tracker for specific classes.
        
        Args:
            classes_to_track: List of class indices to track (default: [1, 2])
            alpha: EMA decay factor (default: 0.95, higher means slower updates)
        """
        self.classes_to_track = classes_to_track
        self.alpha = alpha
        self.class_ema = {cls: 0.0 for cls in classes_to_track}
        self.class_counts = {cls: 0 for cls in classes_to_track}
        self.initialized = False
        
    def update(self, logits, targets):
        """
        Update EMA accuracies based on current batch.
        
        Args:
            logits: Tensor of shape [B, C] with class logits
            targets: Tensor of shape [B] with class targets
        
        Returns:
            Dict of current EMA accuracies for tracked classes
        """
        preds = torch.argmax(logits, dim=1)
        
        for cls in self.classes_to_track:
            # Find samples of this class
            cls_mask = (targets == cls)
            cls_count = cls_mask.sum().item()
            
            if cls_count > 0:
                # Calculate accuracy on samples of this class
                cls_correct = (preds[cls_mask] == targets[cls_mask]).float().mean().item()
                
                # Update EMA
                if not self.initialized and self.class_counts[cls] == 0:
                    self.class_ema[cls] = cls_correct
                else:
                    self.class_ema[cls] = self.alpha * self.class_ema[cls] + (1 - self.alpha) * cls_correct
                
                self.class_counts[cls] += cls_count
        
        if not self.initialized and all(self.class_counts[cls] > 0 for cls in self.classes_to_track):
            self.initialized = True
            
        return {cls: self.class_ema[cls] for cls in self.classes_to_track}
    
    def get_accuracy_string(self):
        """Return formatted accuracy string for tqdm"""
        return ", ".join([f"cls{cls}_acc: {self.class_ema[cls]:.4f}" for cls in self.classes_to_track])
    
    def reset(self):
        """Reset all tracked metrics"""
        self.class_ema = {cls: 0.0 for cls in self.classes_to_track}
        self.class_counts = {cls: 0 for cls in self.classes_to_track}
        self.initialized = False

# Define a MaskedDiceLoss class with lambda term and optional background ignore
class MaskedDiceLoss(nn.Module):
    """
    Dice loss averaged over classes, skipping classes without GT voxels.
    Can ignore background (class 0) and scales by lambda_term.
    """
    def __init__(self,
                 num_classes: int,
                 lambda_term: float = 1.0,
                 ignore_background: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_term = lambda_term
        self.ignore_background = ignore_background
        self.eps = eps

    def forward(self, seg_logits: torch.Tensor, seg_targets: torch.Tensor) -> torch.Tensor:
        # seg_logits: [B, C, D, H, W], seg_targets: [B, D, H, W]
        probs = F.softmax(seg_logits, dim=1)  # [B, C, D, H, W]
        gt_onehot = one_hot(seg_targets, num_classes=self.num_classes)  # [B, C, D, H, W]
        losses = []
        B = seg_logits.shape[0]
        # compute per-sample, per-class dice
        for c in range(self.num_classes):
            if c == 0 and self.ignore_background:
                continue
            for b in range(B):
                gt_c = gt_onehot[b, c, ...].float()
                if gt_c.sum() > 0:
                    pred_c = probs[b, c, ...]
                    inter = (pred_c * gt_c).sum()
                    denom = pred_c.sum() + gt_c.sum()
                    dice_c = 1.0 - 2.0 * inter / (denom + self.eps)
                    losses.append(dice_c)
        if losses:
            return self.lambda_term * torch.stack(losses).mean()
        # no classes to include anywhere
        return torch.tensor(0.0, device=seg_logits.device)

# --- get_image_label (Not strictly needed for seg task, but used by loader) ---
# CORRECTED VERSION
def get_image_label(segmentation_mask_data: np.ndarray, min_pixels_threshold: int = 100) -> int:
    """
    Derives a single image-level classification label from a segmentation mask,
    requiring a minimum number of artifact pixels. CORRECTED VERSION.
    """
    # Ensure input is numpy array if it's a tensor (might happen if called elsewhere)
    if isinstance(segmentation_mask_data, torch.Tensor):
        segmentation_mask_data = segmentation_mask_data.cpu().numpy()

    unique_labels = np.unique(segmentation_mask_data)

    # Check for class 1 first
    if 1 in unique_labels:
        count1 = np.count_nonzero(segmentation_mask_data == 1)
        # Only check the threshold if label 1 exists and count1 is calculated
        if count1 >= min_pixels_threshold:
            return 1
    
    # Only check for class 2 if class 1 wasn't present or wasn't dominant enough
    # Note: Changed from elif to if, to correctly handle cases where 1 exists but count1 < threshold
    if 2 in unique_labels:
        count2 = np.count_nonzero(segmentation_mask_data == 2)
        # Only check the threshold if label 2 exists and count2 is calculated
        if count2 >= min_pixels_threshold:
            return 2

    # If neither class 1 nor class 2 met the criteria
    return 0

# --- Visualization Function (Optional) ---
# def save_small_artifact_visualization(...) -> remains the same if needed

# --- Custom Transform for NPZ Loading (Copied) ---
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
    
# --- GetLabelFromSegd (Only needed if validation transforms require it) ---
# We might not need this if val transforms are simplified or label isn't used
class GetLabelFromSegd(MapTransform):
    def __init__(self, keys: str, label_key: str = 'label', min_pixels_threshold: int = 100, allow_missing_keys: bool = False):
        super().__init__(keys=[keys] if isinstance(keys, str) else keys, allow_missing_keys=allow_missing_keys)
        self.seg_input_key = keys; self.label_output_key = label_key; self.min_pixels_threshold = min_pixels_threshold
    def __call__(self, data):
        d = dict(data); seg_data = d.get(self.seg_input_key)
        if seg_data is None:
            if not self.allow_missing_keys: raise KeyError(f"Seg key '{self.seg_input_key}' missing.")
            d[self.label_output_key] = 0; return d
        seg_np = seg_data.cpu().numpy() if isinstance(seg_data, torch.Tensor) else np.asarray(seg_data)
        image_label = get_image_label(seg_np, self.min_pixels_threshold) # Use global func
        d[self.label_output_key] = image_label
        return d

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
        batch_labels = []
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

                # 7. Get Label (from final processed segmentation)
                seg_np_final = seg_resized.cpu().numpy() # Convert final seg to numpy
                image_label = get_image_label(seg_np_final[0], self.min_pixels_threshold) # Pass spatial part

                # 8. Convert final image tensor to NumPy for batchgenerators
                image_np_final = img_normalized.cpu().numpy()

                # --- Append results ---
                batch_images.append(image_np_final)
                batch_segs.append(seg_np_final) # Append final processed seg
                batch_labels.append(image_label)
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
                'label': np.empty((0,), dtype=np.int64),
                'filenames': [],
                'roi': np.empty((0, c, d, h, w), dtype=np.int64)
             }

        try:
            image_batch_np = np.stack(batch_images, axis=0)
            # Ensure seg batch matches expected type (e.g., uint8)
            seg_batch_np = np.stack(batch_segs, axis=0).astype(np.uint8)
            label_batch_np = np.array(batch_labels, dtype=np.int64)
        except Exception as stack_e:
            tqdm.write(f"Error stacking batch data: {stack_e}")
            tqdm.write(f"Individual image shapes: {[img.shape for img in batch_images]}")
            tqdm.write(f"Individual seg shapes: {[seg.shape for seg in batch_segs]}")
            c, d, h, w = 1, self.input_size[0], self.input_size[1], self.input_size[2]
            # Return empty batch on stacking error
            return {
                'data': np.empty((0, c, d, h, w), dtype=np.float32),
                'seg': np.empty((0, c, d, h, w), dtype=np.uint8),
                'label': np.empty((0,), dtype=np.int64),
                'filenames': [],
                'roi': np.empty((0, c, d, h, w), dtype=np.int64)
             }

        # Return dict required by MTA -> Training Loop
        return {
            'data': image_batch_np,
            'seg': seg_batch_np,
            'label': label_batch_np,
            'roi': np.ones_like(seg_batch_np, dtype=np.int64), # Keep ROI for compatibility maybe?
            'filenames': batch_filenames
        }

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
    model = SwinUNETR(
        img_size=args.input_size,
        in_channels=1,
        out_channels=args.num_seg_classes,
        feature_size=args.feature_size,
        use_v2=True,
        num_classification_outputs=3,
    ).to(device)
    
    # Simple print of model initialization
    print(f"Model initialized with {args.num_seg_classes} segmentation classes")
    
    gc.collect()
    if args.optimizer.lower() == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd': optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    else: raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    print(f"Using {args.optimizer} optimizer: LR={args.initial_lr}, WD={args.weight_decay}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    model, optimizer, scheduler, start_epoch, best_metric, best_metric_epoch = checkpoint(run_output_dir, device, model, optimizer, scheduler)
    
    if not start_epoch:
        start_epoch = 0
        best_metric = 0.0
        best_metric_epoch = -1
    else:
        print(f"Resumed from Epoch {start_epoch}. Previous best Dice: {best_metric:.4f}")
    return model, optimizer, scheduler, start_epoch, best_metric, best_metric_epoch




# --- Main Function (Adapted for Segmentation) ---
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
    
    # --- Loss and Optimizer (CHANGED for Segmentation) ---
    print(f"Using MaskedDiceLoss for {args.num_seg_classes} classes.")
    # instantiate masked dice loss module (ignore background by default)
    criterion = MaskedDiceLoss(num_classes=args.num_seg_classes,
                               lambda_term=1.0,
                               ignore_background=True)
    criterion = criterion.to(device)
    seg_ce_criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()

    # --- Metrics ---
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    print(f"Starting segmentation training for {args.epochs} epochs...")
    model, optimizer, scheduler, start_epoch, best_metric, best_metric_epoch = initialize_model(args, device, run_output_dir)
    
    # --- Training Loop (ADAPTED for Segmentation) ---
    for epoch in range(start_epoch, args.epochs):
        log_memory(f"Epoch {epoch} Start")
        model.train(); train_loss = 0.0; train_steps = 0
        epoch_start_time = time.time(); print("-" * 10); print(f"Epoch {epoch}/{args.epochs - 1}")
        recent_train_losses = []
        
        # Initialize class accuracy tracker for this epoch
        train_accuracy_tracker = ClassAccuracyEMA(classes_to_track=[1, 2], alpha=0.98)

        # --- NO RESTART CALL ---

        MAX_STEPS_PER_EPOCH = 500
        # --- End Define ---

        # Wrap train_loader with islice and set total for tqdm
        progress_bar = tqdm(
            itertools.islice(train_loader, MAX_STEPS_PER_EPOCH),
            desc=f"Epoch {epoch} Train", unit="batch",
            leave=False, total=MAX_STEPS_PER_EPOCH,
            dynamic_ncols=True
        )
        # compute dynamic classification weight (shifted tanh ramp, mid at epoch 5)
        mid_epoch = 5.0
        scale = 5.0
        class_weight = 0.2 * (1.0 + math.tanh((epoch - mid_epoch) / scale))
        print(f"Classification loss weight: {class_weight:.3f} at epoch {epoch}")
        for batch_data in progress_bar:
            # Expecting 'data', 'seg', 'label' (label might be ignored)
            if not isinstance(batch_data, dict) or 'data' not in batch_data or 'seg' not in batch_data or batch_data['data'].size == 0:
                print(f"Training: Batch data keys: {batch_data.keys()}")
                continue # Skip incomplete batch
            train_steps += 1

            inputs = torch.tensor(batch_data['data']).to(device).float()
            seg_targets = torch.tensor(batch_data['seg']).to(device).long() # Use 'seg', ensure Long
            
            # for each segmentation in the batch, we need to see which ones have class 1 and which ones have class 2
            # if there is none that image has a class target of 0
            class_targets = torch.zeros(seg_targets.shape[0]).to(device)
            for i in range(seg_targets.shape[0]):
                if torch.any(seg_targets[i] == 1):
                    class_targets[i] = 1
                elif torch.any(seg_targets[i] == 2):
                    class_targets[i] = 2
            class_targets = class_targets.long()



            optimizer.zero_grad()
            try: # Forward pass
                outputs = model(inputs)
                # Handle potential tuple output from multi-task model
                seg_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                class_logits = outputs[1] if isinstance(outputs, tuple) else None
                # derive aggregated logits for classes 1 and 2 (ignore background channel)
                logits_12 = seg_logits[:, 1:3, ...]  # shape [B,2,D,H,W]
                logits_12 = torch.nn.functional.softmax(logits_12, dim=1)
                vol_logits = logits_12.sum(dim=(2,3,4))  # shape [B,2]
                # convert summed logits to probabilities for stable classification loss
                vol_probs = torch.softmax(vol_logits, dim=1)  # shape [B,2]
            except Exception as e: print(f"Forward pass error: {e}"); continue

            try:
                class_loss = class_criterion(class_logits, class_targets)
                # seg-derived classification loss (artifact 1 vs 2)
                mask = class_targets > 0
                if mask.any():
                    seg_bin_targets = (class_targets[mask] == 2).long()
                    class_probs_loss = F.nll_loss(torch.log(class_targets[mask]), seg_bin_targets)
                else:
                    class_probs_loss = torch.tensor(0.0, device=device)
            except Exception as e: print(f"Class loss calculation error: {e}"); continue

            try: # Backward pass
                loss = criterion(seg_logits, seg_targets) # Use MaskedDiceLoss
                seg_ce_loss = seg_ce_criterion(seg_logits, seg_targets.squeeze(1))  # Use CrossEntropyLoss
                loss = args.lambda_dice * loss + args.lambda_ce * seg_ce_loss + class_weight * (class_loss + class_weight * class_probs_loss)
                loss.backward(); optimizer.step()
            except Exception as e: print(f"Backward/step error: {e}"); continue

            current_loss = loss.item(); train_loss += current_loss
            recent_train_losses.append(current_loss)
            if len(recent_train_losses) > 100: recent_train_losses.pop(0) # Keep last 100
            avg_recent_loss = np.mean(recent_train_losses)
            
            # Update class accuracy EMAs
            if class_logits is not None:
                try:
                    class_accs = train_accuracy_tracker.update(class_logits, class_targets)
                    # display a comprehensive set of metrics
                    progress_bar.set_postfix(
                        total_loss=f"{avg_recent_loss:.4f}",
                        class_loss=f"{class_loss.item():.4f}",
                        segderived_loss=f"{seg_ce_loss.item():.4f}",
                        cls1_acc=f"{class_accs.get(1, 0):.4f}",
                        cls2_acc=f"{class_accs.get(2, 0):.4f}",
                        loss_weight=f"{class_weight:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.6f}"
                    )
                except Exception as e:
                    print(f"Error updating class accuracies: {e}")
                    progress_bar.set_postfix(loss=f"{avg_recent_loss:.4f}")
            else:
                progress_bar.set_postfix(loss=f"{avg_recent_loss:.4f}")

            # Insert debugging visualization every 10 batches
            if train_steps % 10 == 0:
                try:
                    import matplotlib.pyplot as plt
                    # compute segmentation predictions
                    seg_preds = torch.argmax(seg_logits, dim=1)  # shape [B, D, H, W]
                    # select first sample
                    img_vol = inputs[0, 0].cpu().numpy()    # [D, H, W]
                    pred_vol = seg_preds[0].cpu().numpy()    # [D, H, W]
                    true_vol = seg_targets.squeeze(1)[0].cpu().numpy()  # [D, H, W]
                    # define dice calculation
                    def dice_score(pred, true, cls):
                        pred_mask = (pred == cls).astype(np.uint8)
                        true_mask = (true == cls).astype(np.uint8)
                        inter = (pred_mask & true_mask).sum()
                        denom = pred_mask.sum() + true_mask.sum()
                        return (2. * inter / denom) if denom > 0 else 1.0
                    # compute dice for classes 1 and 2
                    dice1 = dice_score(pred_vol, true_vol, 1)
                    dice2 = dice_score(pred_vol, true_vol, 2)
                    # choose slice with most target pixels (classes 1 or 2)
                    pixel_counts = (true_vol > 0).sum(axis=(1,2))
                    if pixel_counts.sum() > 0:
                        slice_idx = int(pixel_counts.argmax())
                    else:
                        slice_idx = img_vol.shape[0] // 2
                    img_slice = img_vol[slice_idx]
                    true_slice = true_vol[slice_idx]
                    pred1_slice = (pred_vol == 1)[slice_idx]
                    pred2_slice = (pred_vol == 2)[slice_idx]
                    # plot with overlay
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    # raw input
                    axes[0].imshow(img_slice, cmap='gray'); axes[0].set_title('Input')
                    # overlay true segmentation mask per class (red = class1, blue = class2)
                    axes[1].imshow(img_slice, cmap='gray')
                    axes[1].imshow(true_slice == 1, cmap='Reds', alpha=0.5)
                    axes[1].imshow(true_slice == 2, cmap='Blues', alpha=0.5)
                    axes[1].set_title('True Mask (red=1, blue=2)')
                    # predicted class1 mask
                    axes[2].imshow(pred1_slice, cmap='Reds'); axes[2].set_title(f'Pred Class1 (dice {dice1:.3f})')
                    # predicted class2 mask
                    axes[3].imshow(pred2_slice, cmap='Blues'); axes[3].set_title(f'Pred Class2 (dice {dice2:.3f})')
                    for ax in axes: ax.axis('off')
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
                 # Add class accuracies if available
                 if class_logits is not None:
                     for cls in [1, 2]:
                         if cls in train_accuracy_tracker.class_ema:
                             wandb_data[f"train/cls{cls}_acc_ema"] = train_accuracy_tracker.class_ema[cls]
                 wandb.log(wandb_data, step=epoch * MAX_STEPS_PER_EPOCH + train_steps)

        # End of Epoch Summary
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0
        epoch_duration = time.time() - epoch_start_time
        print(f"\n--- Epoch {epoch} Training Summary ---")
        print(f"Avg Train Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s")
        print(f"Class EMA Accuracies: {train_accuracy_tracker.get_accuracy_string()}")
        
        if wandb_enabled: 
            wandb_data = {"train/epoch_loss": avg_train_loss, "epoch": epoch}
            # Add final EMAs to wandb
            for cls in [1, 2]:
                if cls in train_accuracy_tracker.class_ema:
                    wandb_data[f"train/epoch_cls{cls}_acc_ema"] = train_accuracy_tracker.class_ema[cls]
            wandb.log(wandb_data, step=(epoch+1)*MAX_STEPS_PER_EPOCH -1)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} LR: {current_lr:.6f}")

        # --- Validation Phase (ADAPTED for Segmentation) ---
        model.eval()
        val_loss = 0.0; val_steps = 0
        dice_metric.reset()
        
        # Initialize class accuracy tracker for validation
        val_accuracy_tracker = ClassAccuracyEMA(classes_to_track=[1, 2], alpha=0.95)
        
        # Collectors for comprehensive evaluation
        all_class_preds = []
        all_class_targets = []

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
                    seg_targets = torch.tensor(batch_data['seg']).to(device).long() # Use 'seg', ensure Long
                    
                    # Apply target remapping
                    seg_targets[seg_targets == 4] = 0 # Set class 4 to background
                    seg_targets[seg_targets == 3] = 0 # Set class 3 to background
                    
                    # Derive class targets from segmentation masks
                    class_targets = torch.zeros(seg_targets.shape[0]).to(device)
                    for i in range(seg_targets.shape[0]):
                        if torch.any(seg_targets[i] == 1):
                            class_targets[i] = 1
                        elif torch.any(seg_targets[i] == 2):
                            class_targets[i] = 2
                    class_targets = class_targets.long()
                    
                    val_steps += 1

                    try:
                        # Get classification predictions (direct forward pass)
                        output_tuple = model(inputs)
                        class_logits = output_tuple[1] if isinstance(output_tuple, tuple) else None
                        seg_logits = output_tuple[0] if isinstance(output_tuple, tuple) else output_tuple
                        # derive aggregated probabilities for classes 1 and 2 (ignore background)
                        logits_12 = seg_logits[:, 1:3, ...]     # shape [B,2,D,H,W]
                        vol_logits = logits_12.sum(dim=(2,3,4))  # shape [B,2]
                        vol_probs = torch.softmax(vol_logits, dim=1)  # shape [B,2]
                        class_probs = vol_probs                  # rename for consistency
                        
                        # Simple debug to check dimensions
                        if val_steps < 3 or val_steps % 50 == 0:
                            print(f"Batch {val_steps} seg_logits shape: {seg_logits.shape}")
                            
                            # Check segmentation logits for each class
                            if isinstance(seg_logits, torch.Tensor) and seg_logits.dim() > 1:
                                num_channels = seg_logits.shape[1]
                                print(f"Seg logits have {num_channels} channels")
                                for c in range(min(num_channels, args.num_seg_classes)):
                                    channel = seg_logits[:, c]
                                    print(f"  Class {c} logits: min={channel.min().item():.4f}, max={channel.max().item():.4f}, mean={channel.mean().item():.4f}")
                                
                                # Apply softmax and check probabilities
                                probs = torch.nn.functional.softmax(seg_logits, dim=1)
                                for c in range(min(probs.shape[1], args.num_seg_classes)):
                                    channel = probs[:, c]
                                    print(f"  Class {c} probs: min={channel.min().item():.4f}, max={channel.max().item():.4f}, mean={channel.mean().item():.4f}")
                    except Exception as e: print(f"Validation inference error: {e}"); continue

                    try:  # Validation Loss (match training loss formula)
                         # dice component
                         dice_loss = criterion(seg_logits, seg_targets)
                         # segmentation CE component (squeeze channel dim)
                         seg_ce_loss = seg_ce_criterion(seg_logits, seg_targets.squeeze(1))
                         # combine dice and CE with weights
                         loss_val = args.lambda_dice * dice_loss + args.lambda_ce * seg_ce_loss
                         if class_logits is not None:
                             # classification head CE
                             class_loss = class_criterion(class_logits, class_targets)
                             # seg-derived classification CE
                             mask = class_targets > 0
                             if mask.any():
                                 seg_bin_targets = (class_targets[mask] == 2).long()
                                 class_probs_loss = F.nll_loss(torch.log(class_probs[mask]), seg_bin_targets)
                             else:
                                 class_probs_loss = torch.tensor(0.0, device=device)
                             # mirror train: weighted classification sub-loss
                             loss_val = loss_val + class_weight * (class_loss + class_weight * class_probs_loss)
                         val_loss += loss_val.item()
                    except Exception as e: print(f"Validation loss error: {e}"); continue

                    try: # Validation Metric
                         # Compute Dice with monai metric
                         try:
                             # Convert logits to predicted classes
                             val_outputs_seg_labels = torch.argmax(seg_logits, dim=1, keepdim=True)
                             
                             # Create one-hot encoded tensors for DiceMetric
                             # num_classes should be exactly 3 for our task
                             val_outputs_one_hot = one_hot(val_outputs_seg_labels, num_classes=args.num_seg_classes)
                             seg_targets_onehot = one_hot(seg_targets, num_classes=args.num_seg_classes)
                             
                             # For every 10th batch, print detailed info about class distributions
                             if val_steps % 10 == 0:
                                 # Check classes in predictions and targets
                                 print(f"\nBatch {val_steps} Segmentation Classes:")
                                 print(f"  Predicted classes: {torch.unique(val_outputs_seg_labels).cpu().numpy()}")
                                 print(f"  Target classes: {torch.unique(seg_targets).cpu().numpy()}")
                                 
                                 # Count pixels per class
                                 for c in range(args.num_seg_classes):
                                     pred_count = (val_outputs_seg_labels == c).sum().item()
                                     target_count = (seg_targets == c).sum().item()
                                     print(f"  Class {c}: Pred={pred_count}, Target={target_count}")
                             
                             # Compute dice using MONAI's metric
                             dice_metric(y_pred=val_outputs_one_hot, y=seg_targets_onehot)
                         except Exception as dice_err:
                             print(f"Error computing dice metric: {dice_err}")
                             import traceback
                             print(traceback.format_exc())
                         
                         # Update class accuracy EMAs
                         if class_logits is not None:
                             class_accs = val_accuracy_tracker.update(class_logits, class_targets)
                             val_pbar.set_postfix(
                                 loss=f"{loss_val.item():.4f}",
                                 cls1_acc=f"{class_accs.get(1, 0):.4f}",
                                 cls2_acc=f"{class_accs.get(2, 0):.4f}"
                             )
                             
                             # Store predictions and targets for confusion matrix and classification report
                             class_preds = torch.argmax(class_logits, dim=1).cpu().numpy()
                             class_targets_np = class_targets.cpu().numpy()
                             all_class_preds.extend(class_preds)
                             all_class_targets.extend(class_targets_np)
                             
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
                dice_scores = np.zeros(args.num_seg_classes)
                metric_val = 0.0
            
            log_memory(f"Epoch {epoch} Val End")
            
            print(f"\n--- Epoch {epoch} Validation Summary ---")
            print(f"Avg Val Loss: {avg_val_loss:.4f}, Mean Dice: {metric_val:.4f}")
            print(f"Class EMA Accuracies: {val_accuracy_tracker.get_accuracy_string()}")
            
            # Generate comprehensive reports if we collected data
            if all_class_preds and all_class_targets:
                try:
                    # Classification metrics report
                    cm = confusion_matrix(all_class_targets, all_class_preds)
                    class_report = classification_report(all_class_targets, all_class_preds, digits=4)
                    
                    print("\n=== CLASSIFICATION REPORT ===")
                    print(f"Confusion Matrix (rows=true, cols=pred):")
                    print(cm)
                    print("\nDetailed Classification Report:")
                    print(class_report)
                    
                    # Basic per-class metrics
                    for cls in [1, 2]:
                        if cls in np.unique(all_class_targets) or cls in np.unique(all_class_preds):
                            precision = cm[cls,cls] / np.sum(cm[:,cls]) if np.sum(cm[:,cls]) > 0 else 0
                            recall = cm[cls,cls] / np.sum(cm[cls,:]) if np.sum(cm[cls,:]) > 0 else 0
                            print(f"Class {cls} Precision: {precision:.4f}, Recall: {recall:.4f}")
                except Exception as e:
                    print(f"Error generating classification report: {e}")
            
            # Segmentation metrics report
            print("\n=== SEGMENTATION REPORT ===")
            print("Per-class Dice scores:")
            for i in range(args.num_seg_classes):
                if i < len(dice_scores):
                    print(f"  Class {i}: {dice_scores[i]:.4f}")
                else:
                    print(f"  Class {i}: N/A")
            
            # Add class distribution in ground truth segmentation masks
            try:
                # Sample a subset of validation files
                sample_size = min(50, len(test_files))
                sample_files = random.sample(test_files, sample_size)
                
                class_pixel_counts = np.zeros(args.num_seg_classes)
                total_pixels = 0
                
                for i, sample in enumerate(tqdm(sample_files, desc="Analyzing validation masks")):
                    try:
                        seg_data = np.load(sample['seg_path'])['arr_0']
                        
                        # Remap classes to match training
                        seg_data_remapped = seg_data.copy()
                        seg_data_remapped[seg_data_remapped == 4] = 0
                        seg_data_remapped[seg_data_remapped == 3] = 0
                        
                        # Count pixels per class
                        for c in range(args.num_seg_classes):
                            class_pixel_counts[c] += np.sum(seg_data_remapped == c)
                        total_pixels += seg_data_remapped.size
                    except Exception as e:
                        print(f"Error analyzing {sample['seg_path']}: {e}")
                
                # Report class distributions
                print(f"\nAnalyzed {sample_size} validation masks")
                for c in range(args.num_seg_classes):
                    percentage = (class_pixel_counts[c] / total_pixels) * 100 if total_pixels > 0 else 0
                    print(f"  Class {c}: {class_pixel_counts[c]:.0f} pixels ({percentage:.4f}%)")
            except Exception as e:
                print(f"Error analyzing validation masks: {e}")
            
            # Class distribution in validation set
            print("\nClass Distribution in Validation Set (Image Level):")
            class_presence_counts = {cls: 0 for cls in range(args.num_seg_classes)}
            for cls_target in all_class_targets:
                class_presence_counts[cls_target] += 1
            
            for cls, count in class_presence_counts.items():
                percentage = (count / len(all_class_targets)) * 100 if all_class_targets else 0
                print(f"Class {cls}: {count} samples ({percentage:.2f}%)")

            if wandb_enabled:
                wandb_data = {
                    "val/epoch_loss": avg_val_loss, 
                    "val/epoch_mean_dice": metric_val, 
                    "epoch": epoch
                }
                # Add class accuracies to wandb
                for cls in [1, 2]:
                    if cls in val_accuracy_tracker.class_ema:
                        wandb_data[f"val/epoch_cls{cls}_acc_ema"] = val_accuracy_tracker.class_ema[cls]
                
                # Add MONAI Dice scores if available
                if val_steps > 0 and len(dice_scores) > 0:
                    # Log overall mean dice
                    wandb_data["val/mean_dice"] = metric_val
                    
                    # Log per-class dice scores
                    for i in range(min(args.num_seg_classes, len(dice_scores))):
                        wandb_data[f"val/dice_class_{i}"] = dice_scores[i]
                
                wandb.log(wandb_data, step=(epoch+1)*MAX_STEPS_PER_EPOCH -1)

            # --- Checkpointing (Based on Mean Dice) ---
            is_best = metric_val > best_metric
            if is_best: best_metric = metric_val; best_metric_epoch = epoch
            print(f" Current Val Dice: {metric_val:.4f}, Best Val Dice: {best_metric:.4f} at Epoch {best_metric_epoch}")
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

    parser.add_argument('--num_seg_classes', type=int, required=True, help="Number of segmentation output classes (e.g., 3 for Bkg, Art1, Art2).")
    parser.add_argument('--input_size', type=int, nargs=3, default=[64, 192, 192+32])
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
    parser.add_argument('--lambda_ce', type=float, default=1.8)
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

    args = parser.parse_args()

    # Make args accessible globally if needed by loader/transforms (like classify_artifacts)
    global cli_args
    cli_args = args

    # Basic validation
    if args.num_seg_classes <= 1: raise ValueError("num_seg_classes must be >= 2.")

    main(args)