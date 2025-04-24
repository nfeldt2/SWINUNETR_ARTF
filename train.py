#!/usr/bin/env python3

import torch
import os
import sys
import numpy as np
import time
import warnings
import argparse
import re
import random
import itertools
from collections.abc import Callable, Sequence
from pathlib import Path
import concurrent.futures # For parallel file pairing moved here
from threading import Thread
from tqdm import tqdm

# --- BatchGenerators Imports ---
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader as BGDataLoader # Use BGDataLoader alias
from batchgenerators.transforms.abstract_transforms import Compose as BGCompose, AbstractTransform

# --- PyTorch Imports ---
from torch import nn
from torch.optim import SGD, AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- MONAI Imports ---
try:
    from swin_unetr import SwinUNETR # Ensure this supports num_classification_outputs
except ImportError:
    print("ERROR: Cannot import SwinUNETR from swin_unetr.py.")
    sys.exit(1)
import monai
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms.transform import MapTransform, Transform
from monai.transforms import (
    Compose as MonaiCompose, # Alias to avoid confusion
    EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, RandRotate90d, RandGaussianNoised, EnsureTyped,
    ScaleIntensityRanged, RandCropByPosNegLabeld,
    NormalizeIntensityd,
    RandZoomd,
    CenterSpatialCropd,
    # Lambda needed? Probably not now.
)
# Use MONAI's DataLoader for validation set
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
from monai.data import pad_list_data_collate, list_data_collate
from monai.data.utils import dense_patch_slices
from monai.utils.enums import LossReduction
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot

# --- Scikit-learn Imports ---
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Custom Local Imports ---
try:
    # Import setup functions from the *modified* setup.py (if it still exists)
    # Or define/move them here if setup.py is nuked
    from training_utils.setup import setup_output_directory # Keep this utility
    from training_utils.logging_utils import initialize_wandb # Keep this utility
    # get_augmentations is crucial
    from Augmentations import get_augmentations
except ImportError as e:
     print(f"ERROR: Failed to import necessary components: {e}")
     print("Ensure training_utils/setup.py contains setup_output_directory.")
     print("Ensure training_utils/logging_utils.py contains initialize_wandb.")
     print("Ensure Augmentations.py contains get_augmentations.")
     sys.exit(1)

# --- WandB ---
try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. Logging will be disabled.")
    wandb = None

warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

# --- get_image_label Definition (Copied from classify_artifacts.py) ---
# Using the 3-class version from classify_artifacts, adjust if needed
def get_image_label(segmentation_mask_data: np.ndarray, min_pixels_threshold: int = 100) -> int:
    unique_labels = np.unique(segmentation_mask_data)
    if 1 in unique_labels: # Check for class 1
        count1 = np.count_nonzero(segmentation_mask_data == 1)
        if count1 >= min_pixels_threshold: return 1
    elif 2 in unique_labels: # Check for class 2
        count2 = np.count_nonzero(segmentation_mask_data == 2)
        if count2 >= min_pixels_threshold: return 2
    return 0 # Return 0 if neither is present above threshold


# === Data Loading Section Copied & Adapted from classify_artifacts.py ===

# --- Custom Transform for NPZ Loading (Copied) ---
class LoadPairedArr0d(MapTransform):
    def __init__(self, keys=("image_path", "seg_path"), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    def __call__(self, data):
        d = dict(data); img_path = d.get("image_path"); seg_path = d.get("seg_path")
        if img_path is None or seg_path is None: raise KeyError("Need 'image_path' and 'seg_path'.")
        try:
            img_npz = np.load(img_path); d["image"] = img_npz['arr_0']; img_npz.close()
            seg_npz = np.load(seg_path); d["seg"] = seg_npz['arr_0']; seg_npz.close()
            del d["image_path"]; del d["seg_path"]
        except Exception as e: print(f"Error loading ({img_path}, {seg_path}): {e}"); raise e
        return d

# --- Custom Base Loader Class (Copied *exactly* from classify_artifacts.py) ---
class ArtifactClassificationDataLoader(BGDataLoader): # Inherit from BGDataLoader
    def __init__(self, data_dicts, batch_size, monai_transforms, min_pixels_threshold, args, num_threads_in_multithreaded=1):
        super().__init__(data_dicts, batch_size, num_threads_in_multithreaded)
        self.monai_transforms = monai_transforms
        self.min_pixels_threshold = min_pixels_threshold
        # --- This line was present in classify_artifacts.py ---
        self.indices = list(range(len(data_dicts)))
        # --- End line ---
        self.args = args # Store args if needed (e.g., for debug viz path)
        # Store num_samples explicitly
        self.num_samples = len(data_dicts)

    # Define __len__ explicitly
    def __len__(self):
        return self.num_samples # Use stored value

    def generate_train_batch(self):
        # Get indices from parent BGDataLoader shuffling mechanism
        indices = self.get_indices()
        batch_images = []; batch_segs = []; batch_labels = []; batch_filenames = []
        skipped_count = 0

        # Sequential processing loop
        for idx in indices:
            data_dict_i = self._data[idx]
            img_path_str = data_dict_i.get('image_path', 'unknown_image')
            filename = Path(img_path_str).name
            try:
                # --- Apply MONAI Transforms ---
                # Expects dict {'image': Tensor, 'seg': Tensor} output from pipeline
                try:
                    monai_output = self.monai_transforms(data_dict_i)
                except Exception as transform_exc:
                    import traceback
                    tqdm.write(f"ERROR during MONAI transform idx {idx} ({filename}): {transform_exc}\n{traceback.format_exc()}")
                    skipped_count += 1; continue

                # --- Validate and Extract ---
                transformed_data = None
                if isinstance(monai_output, dict): transformed_data = monai_output
                elif isinstance(monai_output, (list, tuple)) and len(monai_output) == 1 and isinstance(monai_output[0], dict): transformed_data = monai_output[0]
                else: tqdm.write(f"ERROR: MONAI transforms unexpected type: {type(monai_output)} ({filename}). Skip."); skipped_count += 1; continue

                if not (isinstance(transformed_data, dict) and "image" in transformed_data and "seg" in transformed_data):
                    tqdm.write(f"ERROR: MONAI transforms missing keys for {filename}. Keys: {list(transformed_data.keys())}. Skip.")
                    skipped_count += 1; continue

                # --- Convert to NumPy ---
                seg_tensor = transformed_data['seg']
                seg_np = seg_tensor.cpu().numpy() if isinstance(seg_tensor, torch.Tensor) else np.asarray(seg_tensor)
                image_tensor = transformed_data['image']
                image_np = image_tensor.cpu().numpy() if isinstance(image_tensor, torch.Tensor) else np.asarray(image_tensor)

                # --- Derive Label ---
                # Ensure get_image_label is accessible (defined globally above)
                image_label = get_image_label(seg_np, self.min_pixels_threshold)

                # --- Append to Batch ---
                batch_images.append(image_np)
                batch_segs.append(seg_np) # Append the segmentation mask
                batch_labels.append(image_label)
                batch_filenames.append(filename)

            except Exception as e:
                import traceback
                tqdm.write(f"Error processing sample idx {idx} ({filename}): {e}\n{traceback.format_exc()}")
                skipped_count += 1; continue

        # --- Handle Empty Batch ---
        if not batch_images:
            # Define placeholder shape based on expected patch size
            c, d, h, w = 1, self.args.patch_size[0], self.args.patch_size[1], self.args.patch_size[2]
            return { 'data': np.empty((0, c, d, h, w), dtype=np.float32),
                     'seg': np.empty((0, c, d, h, w), dtype=np.float32), # Use 'seg' key
                     'label': np.empty((0,), dtype=np.int64)}

        # --- Stack Batch ---
        try:
            image_batch_np = np.stack(batch_images, axis=0)
            seg_batch_np = np.stack(batch_segs, axis=0) # Stack seg masks
            label_batch_np = np.array(batch_labels, dtype=np.int64)
        except Exception as stack_e:
            tqdm.write(f"Error stacking batch data: {stack_e}")
            # Log shapes for debugging
            tqdm.write(f"Image shapes: {[img.shape for img in batch_images]}")
            tqdm.write(f"Seg shapes: {[seg.shape for seg in batch_segs]}")
            c, d, h, w = 1, self.args.patch_size[0], self.args.patch_size[1], self.args.patch_size[2]
            return { 'data': np.empty((0, c, d, h, w), dtype=np.float32),
                     'seg': np.empty((0, c, d, h, w), dtype=np.float32),
                     'label': np.empty((0,), dtype=np.int64)}

        # --- Return Final Batch Dictionary (NumPy arrays) ---
        # IMPORTANT: Using keys 'data', 'seg', 'label'
        return { 'data': image_batch_np, 'seg': seg_batch_np, 'label': label_batch_np }
        # Removed 'roi' and 'filenames' for simplicity, add back if needed by BG transforms


class RenameKeyd(MapTransform):
    """ Renames a key in the data dictionary. """
    def __init__(self, old_key: str, new_key: str, allow_missing_keys: bool = False):
        super().__init__(keys=[old_key], allow_missing_keys=allow_missing_keys)
        self.old_key = old_key
        self.new_key = new_key
    def __call__(self, data):
        d = dict(data)
        if self.old_key in d:
            d[self.new_key] = d.pop(self.old_key)
        elif not self.allow_missing_keys:
            raise KeyError(f"Key '{self.old_key}' not found: {d.keys()}")
        return d

class GetLabelFromSegd(MapTransform):
    """ Calculates image-level label from segmentation key, stores in 'label_key'. """
    # Use the definition provided in the previous full train.py script rewrite
    def __init__(self, keys: str, label_key: str = 'label', min_pixels_threshold: int = 100, allow_missing_keys: bool = False):
        super().__init__(keys=[keys] if isinstance(keys, str) else keys, allow_missing_keys=allow_missing_keys)
        self.seg_input_key = keys; self.label_output_key = label_key; self.min_pixels_threshold = min_pixels_threshold
    def __call__(self, data):
        d = dict(data); seg_data = d.get(self.seg_input_key)
        if seg_data is None:
            if not self.allow_missing_keys: raise KeyError(f"Seg key '{self.seg_input_key}' missing.")
            d[self.label_output_key] = 0; return d
        seg_np = seg_data.cpu().numpy() if isinstance(seg_data, torch.Tensor) else np.asarray(seg_data)
        image_label = 0
        if seg_np.ndim >= 3:
            seg_for_label = seg_np[0] if seg_np.ndim == 4 else seg_np
            try:
                # Need access to the global get_image_label function
                # Ensure get_image_label is defined globally or imported in train.py
                # Assuming get_image_label IS defined globally based on previous code context
                global get_image_label # Add this line if needed to access global function
                image_label = get_image_label(seg_for_label, self.min_pixels_threshold)
            except NameError: # Define it inline if not global
                def get_image_label(segmentation_mask_data: np.ndarray, min_pixels_threshold: int = 100) -> int:
                    if 1 in np.unique(segmentation_mask_data):
                        count1 = np.count_nonzero(segmentation_mask_data == 1)
                        if count1 >= min_pixels_threshold: return 1
                    return 0
                image_label = get_image_label(seg_for_label, self.min_pixels_threshold)
            except Exception as label_e: print(f"Warning: Error during get_image_label: {label_e}. Label=0."); image_label = 0
        else: print(f"Warning: Unexpected seg shape {seg_np.shape}. Label=0."); image_label = 0
        d[self.label_output_key] = image_label
        return d


# --- File Pairing Functions (Moved from setup.py to train.py) ---
def check_and_pair_file_worker(img_path_str, img_suffix, seg_suffix):
    img_path = Path(img_path_str)
    expected_seg_name = img_path.name.replace(img_suffix, seg_suffix, 1)
    seg_p = img_path.with_name(expected_seg_name)
    # Check common suffixes if needed
    if not seg_p.is_file(): seg_p = seg_p.with_suffix('.npy')
    if not seg_p.is_file(): seg_p = seg_p.with_suffix('.npz')
    if seg_p.is_file():
        return {"image_path": img_path_str, "seg_path": str(seg_p), "status": "paired"}
    else:
        # tqdm.write(f"Debug: No seg found for {img_path.name} (tried {expected_seg_name})")
        return {"status": "missing_label", "image_name": img_path.name}

def find_and_pair_files_parallel(data_dir, pattern, img_suff, seg_suff, num_threads=None):
    if num_threads is None: num_threads = max(1, os.cpu_count() // 2)
    paired = []; missing_count = 0
    print(f" Searching {data_dir} for '{pattern}' using up to {num_threads} threads...")
    potential_images = sorted([str(f) for f in data_dir.glob(pattern) if img_suff in f.name])
    print(f" Found {len(potential_images)} potential image files. Pairing...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_path = {executor.submit(check_and_pair_file_worker, img_path, img_suff, seg_suff): img_path for img_path in potential_images}
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(potential_images), desc=f"Pairing {data_dir.name}", leave=False):
            try:
                result = future.result()
                if result["status"] == "paired": paired.append({"image_path": result["image_path"], "seg_path": result["seg_path"]})
                else: missing_count += 1 #; print(f"Missing label for {result['image_name']}") # Debug
            except Exception as exc:
                missing_count += 1; print(f"Error processing path via worker: {exc}")
    print(f" Paired {len(paired)} files in {data_dir.name}. {missing_count} missing/error.")
    return paired

# === End Copied Data Loading Section ===


# --- Trainer Class ---
class MultiTaskTrainer(object):
    def __init__(self,
                 # ... (init arguments mostly same, remove ones now handled directly in main flow) ...
                 pretrained_weights=None, device='0', continue_tr = False, fold = '',
                 dataset_dir = '', batch_size = 2, # use_roi removed
                 initial_lr=1e-4, min_lr=1e-6,
                 warmup_epochs=5, scheduler_T0=10, scheduler_T_mult=2, weight_decay=1e-5,
                 num_epochs=200, num_train_iterations=500, num_val_iterations=50,
                 num_seg_classes=3, num_class_labels=1, lambda_seg=1.0, lambda_class=0.1,
                 patch_size=(64, 128, 160), target_spacing=(1.0, 1.0, 1.0),
                 feature_size=24, min_artifact_pixels=100, foreground_prob=0.6,
                 optimizer_type='adamw', momentum=0.9, nesterov=True,
                 verbose=True, output_dir=None, num_workers_train=None,
                 num_workers_val=None, wandb_project_name='ArtifactMTL', no_wandb=False,
                 # Pass args object directly
                 args=None):

        # --- Store configuration ---
        self.args = args # Store the full args object
        self.verbose = verbose; self.initial_lr = initial_lr; self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs; self.T_0 = scheduler_T0; self.T_mult = scheduler_T_mult
        self.weight_decay = weight_decay; self.num_iterations_per_epoch = num_train_iterations
        self.num_val_iterations_per_epoch = num_val_iterations; self.num_epochs = num_epochs
        self.current_epoch = 0; self.device = f"cuda:{device}" if torch.cuda.is_available() and device != 'cpu' else "cpu"; self.best_metric = 0.0
        self.loss = 1000.0; self.dataset_dir = dataset_dir; self.fold = fold; self.continue_tr = continue_tr
        self.batch_size = batch_size; self.default_patch_size = patch_size
        self.num_seg_classes = num_seg_classes; self.num_class_labels_config = num_class_labels
        self.lambda_seg = lambda_seg; self.lambda_class = lambda_class
        self.target_spacing = target_spacing; self.feature_size = feature_size
        self.min_artifact_pixels = min_artifact_pixels; self.foreground_prob = foreground_prob
        self.optimizer_type = optimizer_type.lower(); self.momentum = momentum; self.nesterov = nesterov
        self.output_dir = output_dir; self.fold_dir = None; self.num_workers_train = num_workers_train; self.num_workers_val = num_workers_val
        self.wandb_project_name = wandb_project_name; self.no_wandb = no_wandb
        self.num_classification_outputs = 1 if self.num_class_labels_config == 1 else self.num_class_labels_config
        print(f"Using device: {self.device}")

        # --- Define MONAI Transforms Directly Here ---
        # These are needed for both the custom train loader and the MONAI val loader
        self.monai_train_transforms = self._build_train_transforms()
        self.monai_val_transforms = self._build_val_transforms()

        # --- Initialize Model (Same as before) ---
        print(f"Initializing Modified SwinUNETR for {self.num_seg_classes}-class Seg + {self.num_classification_outputs}-output Class.")
        self.model = SwinUNETR(img_size=self.default_patch_size, in_channels=1, out_channels=self.num_seg_classes, feature_size=self.feature_size, use_v2=True, num_classification_outputs=self.num_classification_outputs).to(self.device)
        if pretrained_weights: self._load_pretrained_weights(self.model, pretrained_weights)

        # --- Optimizer, Loss, Scheduler (Same as before) ---
        if self.optimizer_type == 'adamw': self.optimizer = AdamW(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd': self.optimizer = SGD(self.model.parameters(), lr=self.initial_lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
        else: raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        self.criterion_seg = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
        if self.num_classification_outputs == 1: self.criterion_class = nn.BCEWithLogitsLoss()
        else: self.criterion_class = nn.CrossEntropyLoss()
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)
        self.wandb_initialized = False


    # --- Training Transform Definition (Copied & Adjusted from classify_artifacts.py's structure) ---
    def _build_train_transforms(self):
        """
        Builds MONAI transform pipeline for TRAINING data.
        OUTPUT: Dict {'image': Tensor, 'seg': Tensor}
        """
        img_key = "image"; seg_key = "seg"
        print("--- Building MONAI Training Transforms (Mimicking classify_artifacts.py) ---")
        # Using the 'base_transforms' structure from classify_artifacts.py
        transforms_list = [
            LoadPairedArr0d(keys=("image_path", "seg_path")),
            EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel"),
            Orientationd(keys=[img_key, seg_key], axcodes="RAS"),
            Spacingd(keys=[img_key, seg_key], pixdim=self.target_spacing, mode=("bilinear", "nearest")), # Bilinear for image now
            # NOTE: classify_artifacts used RandCropByPosNegLabeld AND Resized
            # Let's replicate that order. Label derivation happens *after* these in the loader.
            RandCropByPosNegLabeld(
                keys=[img_key, seg_key], label_key=seg_key, spatial_size=self.default_patch_size,
                pos=self.foreground_prob, neg=1.0 - self.foreground_prob, num_samples=1, allow_smaller=True
            ),
            Resized(keys=[img_key, seg_key], spatial_size=self.default_patch_size, mode=("area", "nearest")), # area/nearest
            NormalizeIntensityd(keys=[img_key], subtrahend=0.412456, divisor=0.278396), # Normalize last before output
            EnsureTyped(keys=[img_key, seg_key], dtype=(torch.float32, torch.float32)), # Output float tensors
        ]
        # NOTE: NO GetLabelFromSegd or RenameKeyd here - handled by ArtifactClassificationDataLoader
        print(f" Built {len(transforms_list)} MONAI training transforms.")
        return MonaiCompose(transforms_list)

    # --- Validation Transform Definition (Similar but includes Label/Rename for MONAI Loader) ---
    def _build_val_transforms(self):
        img_key = "image"; seg_key = "seg"
        data_key = "data"; label_key = "label"; seg_target_key = "seg_target"
        print("--- Building MONAI Validation Transforms (Output: 'data', 'seg_target', 'label') ---")
        transforms_list = [
            LoadPairedArr0d(keys=("image_path", "seg_path")),
            EnsureChannelFirstd(keys=[img_key, seg_key], channel_dim="no_channel"),
            Orientationd(keys=[img_key, seg_key], axcodes="RAS"),
            Spacingd(keys=[img_key, seg_key], pixdim=self.target_spacing, mode=("bilinear", "nearest")),
            GetLabelFromSegd(keys=seg_key, label_key=label_key, min_pixels_threshold=self.min_artifact_pixels), # Derive label
            CenterSpatialCropd(keys=[img_key, seg_key], roi_size=self.default_patch_size), # Crop
            Resized(keys=[img_key, seg_key], spatial_size=self.default_patch_size, mode=("area", "nearest")), # Resize
            NormalizeIntensityd(keys=[img_key], subtrahend=0.412456, divisor=0.278396), # Normalize
            EnsureTyped(keys=[seg_key], dtype=torch.long), # Seg to long before rename
            RenameKeyd(old_key=seg_key, new_key=seg_target_key), # Rename seg -> seg_target
            RenameKeyd(old_key=img_key, new_key=data_key),       # Rename image -> data
            EnsureTyped(keys=[data_key, seg_target_key, label_key], dtype=(torch.float32, torch.long, torch.long)), # Final types
        ]
        print(f" Built {len(transforms_list)} MONAI validation transforms.")
        return MonaiCompose(transforms_list)

    # --- _load_pretrained_weights (Same as before) ---
    def _load_pretrained_weights(self, model, weights_path):
        try:
            # ... (implementation unchanged) ...
            print(f"Attempting to load pretrained weights from: {weights_path}")
            if not Path(weights_path).is_file(): print(f"  ERROR: File not found: {weights_path}"); return
            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            if state_dict is None: print(f"  ERROR: Could not find state_dict in {weights_path}"); return
            load_msg = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights. Load message:\n{load_msg}")
        except Exception as e: print(f"Error loading pretrained weights: {e}"); import traceback; traceback.print_exc()


    # --- Setup Data Loaders (Now done inside run_training) ---
    def _setup_dataloaders_internal(self):
        """Sets up train/val dataloaders directly using classify_artifacts pattern."""
        print("--- Setting up Internal DataLoaders (classify_artifacts pattern) ---")
        dataset_path = Path(self.dataset_dir); train_path = dataset_path / 'train'
        val_path = dataset_path / 'validate'; test_path = dataset_path / 'test'
        if not all([p.exists() for p in [train_path, val_path, test_path]]): raise FileNotFoundError(f"Dataset structure incomplete in {self.dataset_dir}.")

        # --- File Pairing ---
        num_pairing_threads = max(1, os.cpu_count() // 2)
        img_suffix="_image_"; seg_suffix="_maskArtifact_"; file_pattern="*.np[yz]"
        train_files_pairs = find_and_pair_files_parallel(train_path, file_pattern, img_suffix, seg_suffix, num_pairing_threads)
        val_files_pairs = find_and_pair_files_parallel(val_path, file_pattern, img_suffix, seg_suffix, num_pairing_threads)
        test_files_pairs = find_and_pair_files_parallel(test_path, file_pattern, img_suffix, seg_suffix, num_pairing_threads)
        if not test_files_pairs: print("Warning: No test data found for validation.")

        # --- KFold Split ---
        all_train_val_pairs = np.array(train_files_pairs + val_files_pairs, dtype=object)
        if not (self.fold.isdigit() and 0 <= int(self.fold) < 5): raise ValueError(f"Invalid fold: {self.fold}")
        fold_idx = int(self.fold); folds = KFold(n_splits=5, shuffle=True, random_state=42)
        train_indices, _ = list(folds.split(all_train_val_pairs))[fold_idx]
        train_files_fold_pairs = all_train_val_pairs[train_indices].tolist()
        val_files_fold_pairs = test_files_pairs # Use test set for validation

        self.val_files_fold = [d['image_path'] for d in val_files_fold_pairs] # Store val filenames
        np.save(self.fold_dir / f'val_files_{self.fold}.npy', np.array(self.val_files_fold))
        print(f"Using Fold {self.fold}: {len(train_files_fold_pairs)} train samples, {len(val_files_fold_pairs)} validation samples (from test dir).")

        # --- Training Loader Setup (Exact classify_artifacts pattern) ---
        print("Setting up BatchGenerators training data loader (classify_artifacts pattern)...")
        # Use the MONAI transforms defined in __init__
        train_dl_base = ArtifactClassificationDataLoader(
            data_dicts=train_files_fold_pairs,
            batch_size=self.batch_size,
            monai_transforms=self.monai_train_transforms, # Pass the train pipeline
            min_pixels_threshold=self.min_artifact_pixels,
            args=self.args # Pass command line args needed by loader
        )
        print(f" Base training loader (ArtifactClassificationDataLoader) initialized. Length: {len(train_dl_base)}")

        print("Getting batchgenerators transforms from Augmentations.py...")
        bg_transforms = get_augmentations() # From Augmentations.py
        if isinstance(bg_transforms, BGCompose): print(f"Loaded {len(bg_transforms.transforms)} BatchGenerators transforms.")
        else: print("Warning: get_augmentations() did not return BGCompose. No BG transforms applied.")

        print("Initializing MultiThreadedAugmenter for training...")
        self.train_loader = MultiThreadedAugmenter(
            data_loader=train_dl_base,      # Use the ArtifactClassificationDataLoader instance
            transform=bg_transforms,        # BG transforms
            num_processes=self.num_workers_train,
            num_cached_per_queue=2,
            pin_memory=True,
            seeds=None # Optional: Can provide seeds for workers
            # NO 'indices' argument here
        )
        print(" Training Loader (MultiThreadedAugmenter) initialized.")

        # --- Validation Loader Setup (Standard MONAI) ---
        print(f" Initializing MONAI Dataset/DataLoader for {len(val_files_fold_pairs)} validation files...")
        # Use the MONAI transforms defined in __init__
        val_ds = MonaiDataset(data=val_files_fold_pairs, transform=self.monai_val_transforms) if val_files_fold_pairs else None
        self.test_loader = MonaiDataLoader( # Assign to self.test_loader
            val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers_val,
            pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate
        ) if val_ds else None
        print(" Validation Loader (MONAI) initialized." if self.test_loader else " Validation Loader is None.")
        print("--- Internal DataLoaders Setup Complete ---")


    # --- Training Loop (MODIFIED to use correct keys) ---
    def train_model(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Multi-Task Training...")
        for epoch in range(self.current_epoch, self.num_epochs):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Epoch {epoch} Training...")
            self.model.train()
            epoch_loss_total = 0.0; epoch_loss_seg = 0.0; epoch_loss_class = 0.0
            epoch_class_correct = 0; epoch_class_total = 0; step = 0

            self.lr_scheduler.step(epoch)
            lr = self.optimizer.param_groups[0]['lr']
            if self.verbose and wandb and self.wandb_initialized: wandb.log({'train/learning_rate': lr, 'epoch': epoch}, step=epoch * self.num_iterations_per_epoch)
            print(f"Epoch {epoch}: LR={lr:.6f}")

            # --- NO train_data_loader.restart() ---

            epoch_start_time = time.time()
            recent_total_losses = []
            pbar = tqdm(itertools.islice(self.train_loader, self.num_iterations_per_epoch),
                        desc=f"Epoch {epoch} Train", unit="batch",
                        leave=False, total=self.num_iterations_per_epoch)

            for batch_data in pbar:
                # Expecting 'data', 'seg', 'label' NumPy arrays from copied loader
                required_keys = ['data', 'seg', 'label']
                if not isinstance(batch_data, dict) or not all(k in batch_data for k in required_keys) or batch_data['data'].size == 0:
                    continue # Skip bad batch
                step += 1

                try:
                    # Convert NumPy arrays to Tensors
                    data_tensor = torch.from_numpy(batch_data['data']).to(self.device).float()
                    seg_target = torch.from_numpy(batch_data['seg']).to(self.device).long() # Use 'seg' key
                    class_target = torch.from_numpy(batch_data['label']).to(self.device).long()
                except Exception as convert_e: print(f"Error converting batch data: {convert_e}"); continue

                self.optimizer.zero_grad(set_to_none=True)

                try: # Forward pass
                    outputs = self.model(data_tensor)
                    if isinstance(outputs, tuple) and len(outputs) == 2: seg_output_logits, class_output_logits = outputs
                    else: print(f"ERROR: Bad model output. Skip."); continue
                except Exception as forward_e: print(f"ERROR forward pass: {forward_e}"); continue

                try: # Seg Loss
                    seg_target_one_hot = one_hot(seg_target, num_classes=self.num_seg_classes)
                    if isinstance(seg_output_logits, (list, tuple)): loss_seg = sum(self.criterion_seg(l, seg_target_one_hot) for l in seg_output_logits) / len(seg_output_logits)
                    else: loss_seg = self.criterion_seg(seg_output_logits, seg_target_one_hot)
                except Exception as seg_loss_e: print(f"ERROR seg loss: {seg_loss_e}"); continue

                try: # Class Loss
                    if isinstance(self.criterion_class, nn.BCEWithLogitsLoss): loss_class = self.criterion_class(class_output_logits.squeeze(-1), class_target.float())
                    else: loss_class = self.criterion_class(class_output_logits, class_target)
                except Exception as class_loss_e: print(f"ERROR class loss: {class_loss_e}"); continue

                total_loss = self.lambda_seg * loss_seg + self.lambda_class * loss_class
                try: total_loss.backward(); self.optimizer.step()
                except Exception as backward_e: print(f"ERROR backward/step: {backward_e}"); continue

                # --- Track Metrics ---
                # ... (metrics tracking identical to previous version) ...
                current_total_loss = total_loss.item()
                epoch_loss_total += current_total_loss; epoch_loss_seg += loss_seg.item(); epoch_loss_class += loss_class.item()
                recent_total_losses.append(current_total_loss)
                with torch.no_grad():
                    batch_total = class_target.size(0)
                    if isinstance(self.criterion_class, nn.BCEWithLogitsLoss): predicted_class = (torch.sigmoid(class_output_logits.squeeze(-1)) > 0.5).long()
                    else: _, predicted_class = torch.max(class_output_logits.data, 1)
                    batch_correct = (predicted_class == class_target).sum().item()
                    epoch_class_correct += batch_correct; epoch_class_total += batch_total
                avg_recent_loss = np.mean(recent_total_losses[-min(50, len(recent_total_losses)):]) if recent_total_losses else 0.0
                batch_acc = (batch_correct / batch_total) * 100 if batch_total > 0 else 0.0
                pbar.set_postfix(Loss=f"{avg_recent_loss:.4f}", ClsAcc=f"{batch_acc:.1f}%")

                # --- Logging ---
                if self.verbose and (step % 100 == 0 or step == 1 or step == self.num_iterations_per_epoch):
                    if self.wandb_initialized and wandb:
                        wandb.log({ 'train/step_loss_total': current_total_loss, 'train/step_loss_seg': loss_seg.item(),
                                    'train/step_loss_class': loss_class.item(), 'train/step_loss_avg100': avg_recent_loss,
                                    'train/step_class_accuracy': batch_acc }, step=epoch * self.num_iterations_per_epoch + step)

            # --- End of Epoch Summary ---
            # ... (summary print identical to previous version) ...
            avg_epoch_loss = epoch_loss_total / step if step > 0 else 0.0; avg_epoch_loss_seg = epoch_loss_seg / step if step > 0 else 0.0; avg_epoch_loss_class = epoch_loss_class / step if step > 0 else 0.0
            avg_epoch_class_acc = (epoch_class_correct / epoch_class_total) * 100 if epoch_class_total > 0 else 0.0
            epoch_duration = time.time() - epoch_start_time
            print(f"\n--- Epoch {epoch} Training Summary ---")
            print(f"Avg Loss: {avg_epoch_loss:.4f} (Seg: {avg_epoch_loss_seg:.4f}, Class: {avg_epoch_loss_class:.4f})")
            print(f"Avg Class Accuracy: {avg_epoch_class_acc:.2f}% ({epoch_class_correct}/{epoch_class_total})"); print(f"Duration: {epoch_duration:.2f} seconds")

            # --- Validation ---
            # ... (validation call identical) ...
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting validation for Epoch {epoch}...")
            val_start_time = time.time(); val_metrics = self.validate(epoch); val_duration = time.time() - val_start_time
            print(f"Avg Val Seg Dice (C1/C2): {val_metrics.get('val/epoch_dice_c1', 0.0):.4f} / {val_metrics.get('val/epoch_dice_c2', 0.0):.4f} (Mean FG: {val_metrics.get('val/epoch_dice_mean_fg', 0.0):.4f})")
            print(f"Avg Val Class Accuracy: {val_metrics.get('val/epoch_class_accuracy', 0.0):.2f}%")
            print(f"Avg Val Loss (Total/Seg/Class): {val_metrics.get('val/epoch_loss_total', 0.0):.4f} / {val_metrics.get('val/epoch_loss_seg', 0.0):.4f} / {val_metrics.get('val/epoch_loss_class', 0.0):.4f}")
            print(f"Validation Duration: {val_duration:.2f} seconds")

            # --- Checkpointing & WandB ---
            # ... (checkpointing/wandb logging identical) ...
            primary_metric_val = val_metrics.get('val/epoch_dice_mean_fg', 0.0)
            is_best = primary_metric_val > self.best_metric
            if is_best: self.best_metric = primary_metric_val
            print(f" Current Val Metric (Avg Fg Dice): {primary_metric_val:.4f}, Best: {self.best_metric:.4f}")
            latest_save_path = self.fold_dir / 'checkpoint_latest.pt'
            checkpoint_latest = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.lr_scheduler.state_dict(), 'best_metric': self.best_metric}
            torch.save(checkpoint_latest, latest_save_path); # print(f" Latest Checkpoint saved.") # Reduce noise
            if is_best: best_save_path = self.fold_dir / 'checkpoint_best.pt'; torch.save(checkpoint_latest.copy(), best_save_path); print(f"  Best Checkpoint saved (Metric: {primary_metric_val:.4f}).")
            if self.wandb_initialized and wandb:
                 log_dict_epoch = { 'train/epoch_loss_total': avg_epoch_loss, 'train/epoch_loss_seg': avg_epoch_loss_seg,
                                    'train/epoch_loss_class': avg_epoch_loss_class, 'train/epoch_class_accuracy': avg_epoch_class_acc,
                                    'epoch': epoch, 'val/best_metric_AvgFgDice': self.best_metric, 'train/learning_rate': lr }
                 log_dict_epoch.update(val_metrics)
                 if 'val/confusion_matrix' in val_metrics and val_metrics['val/confusion_matrix'] is not None: log_dict_epoch['val/confusion_matrix'] = val_metrics['val/confusion_matrix']
                 elif 'val/confusion_matrix' in log_dict_epoch: del log_dict_epoch['val/confusion_matrix']
                 wandb.log(log_dict_epoch, step=(epoch + 1) * self.num_iterations_per_epoch)
            print(f"---------------------------\n")

        print("--- Multi-Task Training Finished ---"); return self.best_metric

    # --- Validation Loop (validate method - unchanged from previous rewrite) ---
    def validate(self, epoch):
        # ... (implementation identical to the previous full train.py rewrite) ...
        # It correctly uses self.test_loader and self.monai_val_transforms
        # and expects keys 'data', 'seg_target', 'label'.
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ---> Entering validate() for Multi-Task, Epoch {epoch}")
        model_to_eval = self.model; model_to_eval.eval()
        seg_dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)
        accumulated_class_preds = []; accumulated_class_labels = []
        total_val_loss_seg = 0.0; total_val_loss_class = 0.0
        val_steps = 0; num_val_samples_processed = 0
        val_loader = self.test_loader
        if val_loader is None: print("Warning: Validation loader is None."); return {}
        with torch.no_grad():
            val_pbar = tqdm(itertools.islice(val_loader, self.num_val_iterations_per_epoch), desc=f"Epoch {epoch} Validate", unit="batch", leave=False, total=self.num_val_iterations_per_epoch)
            for batch_data in val_pbar:
                required_keys = ['data', 'seg_target', 'label']; # Keys from MONAI val loader
                if not isinstance(batch_data, dict) or not all(k in batch_data for k in required_keys) or batch_data['data'].shape[0] == 0: continue
                inputs = batch_data['data'].to(self.device).float()
                seg_target_val = batch_data['seg_target'].to(self.device).long() # Use seg_target key
                class_target_val = batch_data['label'].to(self.device).long()
                try:
                    def seg_predictor(x): output_seg, _ = model_to_eval(x); return output_seg[0] if isinstance(output_seg, (list, tuple)) else output_seg
                    final_seg_logits = monai.inferers.sliding_window_inference(inputs=inputs, roi_size=self.default_patch_size, sw_batch_size=self.batch_size, predictor=seg_predictor, overlap=0.5, mode="gaussian", padding_mode="constant", device=self.device, progress=False)
                    _, class_output_logits = model_to_eval(inputs)
                except Exception as val_infer_e: print(f"Validation inference error: {val_infer_e}"); continue
                try:
                    seg_target_one_hot = one_hot(seg_target_val, num_classes=self.num_seg_classes)
                    if final_seg_logits.shape[-3:] != seg_target_one_hot.shape[-3:]: loss_seg = torch.tensor(0.0) # Or print warning
                    else: loss_seg = self.criterion_seg(final_seg_logits, seg_target_one_hot)
                    total_val_loss_seg += loss_seg.item() * inputs.size(0)
                    if isinstance(self.criterion_class, nn.BCEWithLogitsLoss): loss_class = self.criterion_class(class_output_logits.squeeze(-1), class_target_val.float())
                    else: loss_class = self.criterion_class(class_output_logits, class_target_val)
                    total_val_loss_class += loss_class.item() * inputs.size(0)
                except Exception as val_loss_e: print(f"ERROR calculating val loss: {val_loss_e}"); continue
                # Metrics...
                if isinstance(self.criterion_class, nn.BCEWithLogitsLoss): predicted_class = (torch.sigmoid(class_output_logits.squeeze(-1)) > 0.5).long()
                else: _, predicted_class = torch.max(class_output_logits.data, 1)
                accumulated_class_preds.extend(predicted_class.cpu().numpy())
                accumulated_class_labels.extend(class_target_val.cpu().numpy())
                if final_seg_logits.shape[-3:] == seg_target_val.shape[-3:]:
                     seg_pred_probs = torch.softmax(final_seg_logits, dim=1); seg_pred_labels = torch.argmax(seg_pred_probs, dim=1, keepdim=True)
                     seg_target_one_hot_dice = one_hot(seg_target_val, num_classes=self.num_seg_classes)
                     seg_dice_metric(y_pred=seg_pred_labels, y=seg_target_one_hot_dice)
                num_val_samples_processed += inputs.size(0); val_steps += 1
        # Aggregate... (identical aggregation logic as before)
        avg_val_loss_seg = total_val_loss_seg/num_val_samples_processed if num_val_samples_processed > 0 else 0; avg_val_loss_class = total_val_loss_class/num_val_samples_processed if num_val_samples_processed > 0 else 0
        avg_val_loss_total = self.lambda_seg*avg_val_loss_seg + self.lambda_class*avg_val_loss_class; report_str=None; cm=None; wandb_cm=None; avg_val_class_acc=0.0
        if accumulated_class_labels:
            avg_val_class_acc = accuracy_score(accumulated_class_labels, accumulated_class_preds) * 100
            try:
                num_labels=self.num_classification_outputs if self.num_classification_outputs > 1 else 2; target_names=[f"C_{i}" for i in range(num_labels)]
                report_str=classification_report(accumulated_class_labels, accumulated_class_preds, target_names=target_names, zero_division=0, labels=range(num_labels))
                cm = confusion_matrix(accumulated_class_labels, accumulated_class_preds, labels=range(num_labels))
                if wandb and self.wandb_initialized: wandb_cm = wandb.Table(columns=["A", "P", "N"], rows=[[target_names[r], target_names[p], cm[r, p]] for r in range(num_labels) for p in range(num_labels)])
            except Exception as report_e: print(f"Warn: Report/CM err: {report_e}")
        dice_c1, dice_c2, dice_mean_fg = 0.0, 0.0, 0.0
        try:
            dice_scores=seg_dice_metric.aggregate(); seg_dice_metric.reset()
            if isinstance(dice_scores, torch.Tensor) and dice_scores.numel() >= (self.num_seg_classes - 1):
                 dice_c1=dice_scores[0].item(); dice_c2=dice_scores[1].item(); dice_mean_fg=(dice_c1 + dice_c2) / 2.0
        except Exception as dice_agg_e: print(f"Warn: Dice agg err: {dice_agg_e}")
        metrics = { "val/epoch_loss_total": avg_val_loss_total, "val/epoch_loss_seg": avg_val_loss_seg, "val/epoch_loss_class": avg_val_loss_class,
                    "val/epoch_class_accuracy": avg_val_class_acc, "val/epoch_dice_c1": dice_c1, "val/epoch_dice_c2": dice_c2,
                    "val/epoch_dice_mean_fg": dice_mean_fg, "val/confusion_matrix": wandb_cm }
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] <--- Exiting validate() for Multi-Task, Epoch {epoch}")
        model_to_eval.train(); return metrics


    # --- run_training (Uses internal setup method) ---
    def run_training(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Entering run_training (Multi-Task - Copied Data Loading)...")
        parent_dir = Path(self.dataset_dir).parent; dataset_name = parent_dir.name if parent_dir else Path(self.dataset_dir).name
        output_base_name = f"{dataset_name}_MTL"
        self.fold_dir, log_file_handle, _ = setup_output_directory(self.output_dir, str(self.fold), output_base_name, self.continue_tr)
        run_identifier = f"{output_base_name}_Fold{self.fold}_LSeg{self.lambda_seg}_LCls{self.lambda_class}"
        original_stdout, original_stderr = sys.stdout, sys.stderr
        if log_file_handle != sys.stdout: sys.stdout, sys.stderr = log_file_handle, log_file_handle

        self.current_epoch = 0; self.best_metric = 0.0; load_successful = False
        if self.continue_tr: # Checkpoint loading logic...
             latest_path = self.fold_dir / 'checkpoint_latest.pt'
             if latest_path.exists(): load_successful = self._load_checkpoint(latest_path, self.model, self.optimizer, self.lr_scheduler)
             if not load_successful: print("Checkpoint loading failed."); self._reset_optimizer_scheduler()
        else: print("Starting training from scratch.")

        if not self.no_wandb: initialize_wandb(self, run_identifier, self.continue_tr, project_name=self.wandb_project_name)
        else: print("WandB disabled."); self.wandb_initialized = False

        # --- Setup DataLoaders using the internal method ---
        self._setup_dataloaders_internal()

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Multi-Task Training Loop...")
        try: final_best_metric = self.train_model() # Calls the main training loop
        finally: # Cleanup
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Exiting training loop.")
            if log_file_handle != sys.stdout: sys.stdout, sys.stderr = original_stdout, original_stderr; log_file_handle.close()
            if wandb and wandb.run: wandb.finish()

    # --- _load_checkpoint and _reset_optimizer_scheduler (Unchanged) ---
    def _load_checkpoint(self, path, model, optimizer, scheduler):
        if not Path(path).exists(): print(f"Checkpoint file not found: {path}"); return False
        try: # ... (implementation identical to previous version) ...
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model_state = checkpoint.get('state_dict') or checkpoint.get('model_state_dict'); model.load_state_dict(model_state, strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']); scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint.get('epoch', -1) + 1; self.best_metric = checkpoint.get('best_metric', self.best_metric)
            print(f"Loaded checkpoint {os.path.basename(path)}. Resuming epoch {self.current_epoch}. Best Metric: {self.best_metric:.4f}"); return True
        except Exception as e: print(f"Error loading checkpoint {path}: {e}"); return False

    def _reset_optimizer_scheduler(self):
        print("Resetting optimizer and scheduler.")
        if self.optimizer_type == 'adamw': self.optimizer = AdamW(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd': self.optimizer = SGD(self.model.parameters(), lr=self.initial_lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)
        self.current_epoch = 0; self.best_metric = 0.0

# --- run_training_entry (Instantiates Trainer, passes args) ---
def run_training_entry():
    parser = argparse.ArgumentParser(description="Train Multi-Task SwinUNETR (Seg + Class) - COPIED DATALOADING")
    # ... (All arguments definitions identical to previous version) ...
    parser.add_argument('dataset_dir', type=str); parser.add_argument('fold', type=str)
    parser.add_argument('--output_dir', type=str, default='./results_mtl')
    parser.add_argument('--c', action='store_true'); parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=200); parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--initial_lr', type=float, default=1e-4); parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-5); parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--scheduler_T0', type=int, default=10); parser.add_argument('--scheduler_T_mult', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--momentum', type=float, default=0.9); parser.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num_seg_classes', type=int, default=3); parser.add_argument('--num_class_labels', type=int, default=1)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 128, 160])
    parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument('--feature_size', type=int, default=24); parser.add_argument('--min_artifact_pixels', type=int, default=100)
    parser.add_argument('--foreground_prob', type=float, default=0.6); parser.add_argument('--no_use_roi', action='store_true')
    parser.add_argument('--lambda_seg', type=float, default=1.0); parser.add_argument('--lambda_class', type=float, default=0.1)
    parser.add_argument('--train_steps_per_epoch', type=int, default=500); parser.add_argument('--val_steps_per_epoch', type=int, default=50)
    parser.add_argument('--num_workers_train', type=int, default=max(1, os.cpu_count() // 2)) # Default changed slightly
    parser.add_argument('--num_workers_val', type=int, default=2); parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no_wandb', action='store_true'); parser.add_argument('--wandb_project', type=str, default='ArtifactMTL')

    args = parser.parse_args()

    # --- Validate arguments ---
    if not (args.fold.isdigit() and 0 <= int(args.fold) < 5): raise ValueError("Fold must be 0-4.")
    # ... (other validations) ...

    # --- Create Trainer ---
    trainer = MultiTaskTrainer(
        # Pass all relevant args from parser
        args=args, # Pass the args object itself
        pretrained_weights=args.pretrained_weights, fold=args.fold, dataset_dir=args.dataset_dir, device=args.device,
        continue_tr=args.c, batch_size=args.batch_size, initial_lr=args.initial_lr,
        min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, scheduler_T0=args.scheduler_T0, scheduler_T_mult=args.scheduler_T_mult,
        weight_decay=args.weight_decay, num_epochs=args.epochs, num_train_iterations=args.train_steps_per_epoch,
        num_val_iterations=args.val_steps_per_epoch, num_seg_classes=args.num_seg_classes, num_class_labels=args.num_class_labels,
        lambda_seg=args.lambda_seg, lambda_class=args.lambda_class, patch_size=tuple(args.patch_size), target_spacing=tuple(args.target_spacing),
        feature_size=args.feature_size, min_artifact_pixels=args.min_artifact_pixels, foreground_prob=args.foreground_prob,
        optimizer_type=args.optimizer, momentum=args.momentum, nesterov=args.nesterov, verbose=not args.quiet,
        output_dir=args.output_dir, num_workers_train=args.num_workers_train, num_workers_val=args.num_workers_val,
        wandb_project_name=args.wandb_project, no_wandb=args.no_wandb
        # use_roi is not needed by trainer init anymore
    )
    trainer.run_training()

if __name__ == '__main__':
    run_training_entry()