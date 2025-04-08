import torch
import os
from Dataloaders import CustomDataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from torch.optim import SGD
from torch import nn
from Augmentations import get_augmentations
from monai.losses import DiceFocalLoss, DiceLoss, DiceCELoss, AsymmetricUnifiedFocalLoss, GeneralizedDiceFocalLoss, GeneralizedWassersteinDiceLoss
from swin_unetr import SwinUNETR
import sys
import numpy as np
from lrScheduler import PolyLRScheduler
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel
import re
import random
from collections.abc import Callable, Sequence
from skimage.transform import resize
import time
# 5 fold cross validation library
from sklearn.model_selection import KFold
from threading import Thread
import wandb
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.modules.loss import _Loss
from monai.losses.tversky import TverskyLoss
from monai.losses.focal_loss import FocalLoss
# from monai.losses.dice import DiceLoss # Already imported above
from monai.inferers import sliding_window_inference
import argparse
from batchgenerators.transforms.abstract_transforms import Compose
from pathlib import Path # Use pathlib for robust path handling
from mpl_toolkits.axes_grid1 import make_axes_locatable # For colorbar

try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. Logging will be disabled.")
    wandb = None


# --- Helper Functions ---

def setup_output_directory(output_dir: str, fold: str, dataset_name: str, continue_tr: bool):
    """Creates fold-specific output directory and sets up logging.

    Args:
        output_dir: Base directory for results.
        fold: Current fold number (string).
        dataset_name: Name of the dataset.
        continue_tr: Flag indicating if training is being continued.

    Returns:
        Path: Path to the fold-specific output directory.
        TextIOWrapper: Open log file handle.
        bool: Flag indicating if a checkpoint might exist.
    """
    base_path = Path(output_dir)
    dataset_path = base_path / dataset_name
    fold_path = dataset_path / f"fold_{fold}"

    fold_path.mkdir(parents=True, exist_ok=True)

    checkpoint_exists = (fold_path / 'checkpoint.pt').exists()

    if checkpoint_exists and not continue_tr:
        print(f"WARNING: Checkpoint found in {fold_path} but --c flag not used. Potential to overwrite.")
        # Decide if overwriting is allowed or if an error should be raised.
        # For now, we'll allow overwriting but warn.
        # raise ValueError(f"Fold {fold} checkpoint exists at {fold_path / 'checkpoint.pt'}. Use --c to continue.")

    # Setup logging to file
    log_file_name = "log0.txt"
    i = 0
    while (fold_path / log_file_name).exists():
        # If continuing training and a log file exists, append to the latest one
        if continue_tr:
            break 
        i += 1
        log_file_name = f"log{i}.txt"
    
    log_file_path = fold_path / log_file_name
    log_mode = 'a' if continue_tr and (fold_path / log_file_name).exists() else 'w'
    # Ensure log file handle is managed correctly
    try:
        log_file = open(log_file_path, mode=log_mode)
    except IOError as e:
        print(f"Error opening log file {log_file_path}: {e}")
        # Fallback to standard output if log file fails
        log_file = sys.stdout 

    if log_file != sys.stdout:
        sys.stdout = log_file # Redirect stdout only if file opened successfully
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Output directory: {fold_path}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Logging to {log_file_path} (mode: {log_mode})")

    return fold_path, log_file, checkpoint_exists

def setup_dataloaders(dataset_dir: str, fold: str, fold_dir: Path, batch_size: int, use_roi: bool, num_workers_train: int, num_workers_val: int):
    """Sets up KFold splits, DataLoaders, and Augmenters.

    Args:
        dataset_dir: Path to the root dataset directory.
        fold: Current fold number (string).
        fold_dir: Path to the fold-specific output directory.
        batch_size: Training batch size.
        use_roi: Whether to use ROI cropping/padding.
        num_workers_train: Number of workers for training augmenter.
        num_workers_val: Number of workers for validation augmenter.

    Returns:
        MultiThreadedAugmenter: Training data loader.
        MultiThreadedAugmenter: Test data loader.
        list: List of file paths used for validation in this fold.
        Compose: Training augmentations.
    """
    dataset_path = Path(dataset_dir)
    # Assume dataset_dir points to the directory containing 'train', 'validate', 'test'
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'validate'
    test_path = dataset_path / 'test'

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Dataset directory structure incomplete in {dataset_dir}. Expected 'train', 'validate', 'test' subdirectories.")

    train_files = [str(f) for f in train_path.glob("*.np[yz]")] # Find .npy or .npz
    val_files_orig = [str(f) for f in val_path.glob("*.np[yz]")]
    test_files = [str(f) for f in test_path.glob("*.np[yz]")]

    if not train_files or not val_files_orig:
        raise FileNotFoundError(f"No training/validation data found in {train_path} / {val_path}")
    if not test_files:
        print(f"Warning: No test data found in {test_path}")

    # Combine train and original validation files for KFold splitting
    all_train_val_files = np.array(train_files + val_files_orig, dtype=str)

    if not (fold.isdigit() and 0 <= int(fold) < 5):
         raise ValueError(f"Fold must be an integer between 0 and 4, got {fold}")
    fold_idx = int(fold)

    # Perform KFold split
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = list(folds.split(all_train_val_files))[fold_idx]

    train_files_fold = all_train_val_files[train_indices]
    val_files_fold = all_train_val_files[val_indices]

    # Save the validation file list for this fold
    np.save(fold_dir / f'val_files_{fold}.npy', val_files_fold)
    print(f"Training on fold {fold} with {len(train_files_fold)} train and {len(val_files_fold)} validation files (split from combined train+validate dirs).")

    # --- Create DataLoaders ---
    # Note: The original code adds val_files to train_files. Replicating this behavior.
    # Consider if validation set should be separate or used for training.
    train_files_for_loader = list(train_files_fold) + list(val_files_fold)
    print(f"Creating training loader with {len(train_files_for_loader)} files (train_fold + val_fold)...")

    train_loader_base = CustomDataLoader(train_files_for_loader, batch_size=batch_size, LR=False) # Assuming LR flag is deprecated/related to roi?
    test_loader_base = CustomDataLoader(test_files, batch_size=1, val=True, LR=False)

    # --- Setup Augmentations ---
    transforms = get_augmentations()

    # --- Create MultiThreadedAugmenters ---
    train_loader = MultiThreadedAugmenter(
        train_loader_base, 
        transforms, 
        num_processes=num_workers_train, 
        num_cached_per_queue=8, # Keep original queue size? Or make configurable?
        pin_memory=True, 
        useroi=use_roi
    )
    test_loader = MultiThreadedAugmenter(
        test_loader_base, 
        None, # No augmentation for test set
        num_processes=num_workers_val, 
        num_cached_per_queue=3, 
        pin_memory=True, 
        useroi=use_roi, 
        val=True
    )

    return train_loader, test_loader, val_files_fold, transforms


# --- Custom Loss Function ---

class ArtifactCorrectionLoss(_Loss):
    """
    Calculates loss based on the single artifact class present in the target.
    Assumes target contains 0 for background, 1 for artifact class 1, 2 for class 2.
    Assumes model output has 2 channels (index 0 for class 1, index 1 for class 2).
    """
    def __init__(self, 
                 loss_type: str = 'dice', 
                 class_weights = [10.0, 1.0], # Corrected weights: Class 1 = 10.0, Class 2 = 1.0
                 sigmoid: bool = True, 
                 **loss_kwargs):
        super().__init__()
        # Ensure the passed weights are used, not overwritten later
        self.class_weights = class_weights 
        self.sigmoid = sigmoid
        
        # Instantiate the underlying binary loss function
        if loss_type.lower() == 'dice':
            self.binary_loss = DiceLoss(sigmoid=False, **loss_kwargs) # Sigmoid applied manually
        elif loss_type.lower() == 'tversky':
             self.binary_loss = TverskyLoss(sigmoid=False, **loss_kwargs)
        elif loss_type.lower() == 'focal':
             self.binary_loss = FocalLoss(to_onehot_y=False, **loss_kwargs) # Focal expects probabilities
             self.sigmoid = True # Force sigmoid for FocalLoss
        elif loss_type.lower() == 'dicefocal':
             self.binary_loss = DiceFocalLoss(sigmoid=False, **loss_kwargs)
             self.sigmoid = True # Force sigmoid for DiceFocal
        elif loss_type.lower() == 'dicece': # <-- Add DiceCE option
             # Pass lambda_dice and lambda_ce from loss_kwargs if provided
             dice_lambda = loss_kwargs.pop('lambda_dice', 1.0) # Default to 1.0 if not passed
             ce_lambda = loss_kwargs.pop('lambda_ce', 1.0)   # Default to 1.0 if not passed
             # Enable squared_pred for potentially smoother gradients
             self.binary_loss = DiceCELoss(sigmoid=False, lambda_dice=dice_lambda, lambda_ce=ce_lambda, squared_pred=True, **loss_kwargs)
             self.sigmoid = True # Force sigmoid for DiceCE
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
            
        print(f"Initialized ArtifactCorrectionLoss with {loss_type.upper()} loss.")

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: Model output tensor (B, C=2, D, H, W). Channel 0 for class 1, Channel 1 for class 2.
            target: Ground truth tensor (B, 1, D, H, W) with integer labels (0, 1, 2).
        """
        batch_size = target.shape[0]
        total_loss = 0.0
        samples_with_artifact = 0

        for b in range(batch_size):
            # Find the ground truth artifact class for this sample
            # Use unique instead of max to handle potential edge cases/errors
            unique_labels = torch.unique(target[b])
            gt_classes = unique_labels[unique_labels > 0] # Get non-background labels (e.g., tensor([1, 2]))

            if len(gt_classes) == 0:
                # No artifact present in this sample, loss is 0
                continue
            #elif len(gt_classes) > 1: # No longer needed to warn specifically for > 1
                # Should not happen based on problem description (only one artifact class per image)
                # Take the numerically higher class if error occurs, or handle differently
                #warnings.warn(f"Sample {b} contains multiple artifact classes: {gt_classes}. Using max: {torch.max(gt_classes)}.")
                #gt_class = torch.max(gt_classes).item() # OLD LOGIC - REMOVE
            #else:
            #    gt_class = gt_classes[0].item() # OLD LOGIC - REMOVE

            # --- NEW LOGIC: Iterate through all present GT classes ---
            sample_loss = 0.0
            num_artifact_classes_in_sample = 0
            for gt_class_val in gt_classes: # Iterate through [1], [2], or [1, 2]
                 gt_class = gt_class_val.item()

                 if gt_class not in [1, 2]:
                      warnings.warn(f"Unexpected gt_class {gt_class} found in sample {b}. Skipping this class.")
                      continue

                 num_artifact_classes_in_sample += 1

                 # Select the corresponding output channel (0 for class 1, 1 for class 2)
                 output_channel = output[b:b+1, gt_class - 1 : gt_class]

                 # Create the binary target mask for the specific class
                 binary_target = (target[b:b+1] == gt_class).float()

                 # Apply sigmoid if required by the loss type
                 if self.sigmoid:
                     output_channel = torch.sigmoid(output_channel)

                 # Calculate the binary loss for this specific class
                 loss_for_class = self.binary_loss(output_channel, binary_target)

                 # Apply class weight
                 loss_for_class *= self.class_weights[gt_class - 1]

                 sample_loss += loss_for_class
            # --- End NEW LOGIC ---


            # Sum loss over artifact classes present *in this sample*
            if num_artifact_classes_in_sample > 0:
                # OLD: total_loss += (sample_loss / num_artifact_classes_in_sample) 
                total_loss += sample_loss # NEW: Sum the losses directly
                samples_with_artifact += 1


        # Average loss over samples that actually contained an artifact
        if samples_with_artifact > 0:
             avg_loss = total_loss / samples_with_artifact
        else:
             # Return zero loss, ensuring it's connected to the graph
             # Multiply by output sum ensures requires_grad=True if output does
             avg_loss = 0.0 * output.sum()
             # avg_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype) # Original problematic line

        return avg_loss


class SWINUNETRTrainer(object):
    def __init__(self, 
                 model=None, 
                 optimizer=None, 
                 weights=None, 
                 device='0', 
                 continue_tr = False, 
                 fold = '', 
                 dataset_dir = '', 
                 batch_size = 3, 
                 use_roi = True, 
                 initial_lr=0.001,
                 min_lr=1e-7,
                 warmup_epochs=10,
                 scheduler_T0=30,
                 scheduler_T_mult=2,
                 weight_decay=5e-4,
                 num_epochs=500,
                 num_train_iterations=500, 
                 num_val_iterations=250, 
                 enable_deep_supervision=True,
                 patch_size=(64, 160, 256), 
                 img_size=(64, 160, 256), 
                 feature_size=24,
                 loss_type='dicefocal', 
                 loss_weight_class1=1.0,
                 loss_weight_class2=1.0,
                 focal_gamma=2.0,
                 tversky_alpha=0.5,
                 tversky_beta=0.5,
                 foreground_prob=0.5, # Add foreground_prob here
                 lambda_dice=1.0, # Add lambda_dice
                 lambda_ce=1.0, # Add lambda_ce
                 verbose=True):
        
        # Store configuration parameters
        self.verbose = verbose
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.T_0 = scheduler_T0
        self.T_mult = scheduler_T_mult
        self.weight_decay = weight_decay
        self.num_iterations_per_epoch = num_train_iterations 
        self.num_val_iterations_per_epoch = num_val_iterations
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.enable_deep_supervision = enable_deep_supervision
        self.device = device
        self.best_loss = 1000 # Use best validation dice instead?
        self.best_dice = 0.0 
        self.loss = 1000 # Track training loss
        self.dataset_dir = dataset_dir
        self.fold = fold
        self.continue_tr = continue_tr
        self.batch_size = batch_size
        self.use_roi = use_roi
        self.default_patch_size = patch_size 
        self.img_size = img_size 
        self.feature_size = feature_size 
        self.loss_type = loss_type
        self.loss_weight_class1 = loss_weight_class1
        self.loss_weight_class2 = loss_weight_class2
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.foreground_prob = foreground_prob # Store foreground_prob
        self.lambda_dice = lambda_dice # Store lambda_dice
        self.lambda_ce = lambda_ce # Store lambda_ce

        # Store paths and worker counts (passed from run_training_entry)
        self.output_dir = None
        self.fold_dir = None # Will be set in run_training
        self.num_workers_train = None
        self.num_workers_val = None

        torch.device(f"cuda:{self.device}") 

        if continue_tr:
            # Checkpoint loading happens *after* directory setup in run_training
            pass
        else:
            # Model setup
            if model is not None:
                self.model = model.cuda(device)
            else:
                self.model = SwinUNETR(
                    img_size=self.img_size, 
                    in_channels=1, 
                    out_channels=2, # 2 output channels for class 1 and class 2
                    feature_size=self.feature_size, 
                    deep_supervision=self.enable_deep_supervision, 
                    use_v2=True
                ).cuda(self.device)
            
            # Optimizer setup
            if optimizer is not None:
                self.optimizer = optimizer
            else:
                from torch.optim import AdamW
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.initial_lr,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )

            # Load pretrained weights if provided
            if weights is not None:
                 try:
                     # Use map_location for flexibility
                     state_dict = torch.load(weights, map_location=lambda storage, loc: storage.cuda(self.device))
                     # Handle potential mismatch (e.g., different output layer size)
                     model_dict = self.model.state_dict()
                     # Filter out unnecessary keys or mismatched layers
                     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                     model_dict.update(pretrained_dict) 
                     self.model.load_state_dict(model_dict)
                     print(f"Loaded {len(pretrained_dict)} matching keys from pretrained weights: {weights}")
                 except Exception as e:
                     print(f"Error loading pretrained weights from {weights}: {e}. Training from scratch.")


        # --- Loss Function Initialization ---
        loss_weights = [self.loss_weight_class1, self.loss_weight_class2]
        loss_kwargs = {}
        if self.loss_type.lower() == 'tversky':
            loss_kwargs['alpha'] = self.tversky_alpha
            loss_kwargs['beta'] = self.tversky_beta
        elif self.loss_type.lower() == 'focal' or self.loss_type.lower() == 'dicefocal':
            loss_kwargs['gamma'] = self.focal_gamma
        
        # Pass lambda weights if applicable for the chosen loss type
        if self.loss_type.lower() in ['dicece', 'dicefocal']:
            loss_kwargs['lambda_dice'] = self.lambda_dice
            loss_kwargs['lambda_ce'] = self.lambda_ce

        self.criterion = ArtifactCorrectionLoss(
            loss_type=self.loss_type, 
            class_weights=loss_weights,
            sigmoid=(self.loss_type.lower() in ['focal', 'dicefocal', 'tversky']),
            **loss_kwargs
        )
        
        # Deep supervision losses: Use the same loss type for consistency for now
        # Could be made configurable per layer later if needed
        if self.enable_deep_supervision:
            # Pass lambda weights to deep supervision losses too
            deep_loss_kwargs = loss_kwargs.copy()
            self.deep_supervision_losses = nn.ModuleList([
                 ArtifactCorrectionLoss(
                     loss_type=self.loss_type, 
                     class_weights=loss_weights, 
                     sigmoid=(self.loss_type.lower() in ['focal', 'dicefocal', 'tversky', 'dicece']), # Add dicece here too
                     **deep_loss_kwargs)
                 for _ in range(5) # SwinUNETR default deep supervision levels
            ])
            # Adjust deep supervision weights 
            self.deep_supervision_weights = np.array([1 / (2**i) for i in range(len(self.deep_supervision_losses))]) # More standard weighting
            self.deep_supervision_weights = self.deep_supervision_weights / self.deep_supervision_weights.sum()
        else:
             self.deep_supervision_losses = None
             self.deep_supervision_weights = None


        # --- Scheduler Initialization ---
        self.lr = self.optimizer.param_groups[0]['lr']

        self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.min_lr
        )
        
        # Defer wandb init until fold_dir is known in run_training
        self.wandb_initialized = False

    def initialize_wandb(self):
         """Initializes Weights & Biases logging."""
         if wandb is None:
             print("wandb not installed, logging disabled.")
             self.verbose = False 
             return

         try:
            config = {
                "batch_size": self.batch_size,
                "patch_size": self.default_patch_size,
                "img_size": self.img_size,
                "feature_size": self.feature_size,
                "max_epochs": self.num_epochs,
                "initial_lr": self.initial_lr,
                "min_lr": self.min_lr,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
                "scheduler_T0": self.T_0,
                "scheduler_T_mult": self.T_mult,
                "fold": self.fold,
                "model": "SwinUNETR-artf", 
                "dataset": Path(self.dataset_dir).parent.name if Path(self.dataset_dir).parent else Path(self.dataset_dir).name, # Get dataset name
                "deep_supervision": self.enable_deep_supervision,
                "use_roi": self.use_roi,
                "loss_type": self.criterion.binary_loss.__class__.__name__, # Log the specific loss used
                "loss_weights": self.criterion.class_weights,
                # Add loss kwargs if needed: **self.criterion.binary_loss_kwargs 
            }
            wandb.init(project="swinunetr-artifact-correction", 
                       config=config, 
                       dir=str(self.fold_dir), # Log wandb runs within the fold directory
                       name=f"fold_{self.fold}", # Name the run based on the fold
                       resume="allow", # Allow resuming if run exists
                       id=f"fold_{self.fold}_{Path(self.output_dir).name}") # Unique ID for resuming
            self.wandb_initialized = True
            print("wandb initialized successfully.")
         except ImportError:
             print("wandb not installed, disabling logging.")
             self.verbose = False
         except Exception as e:
             print(f"Error initializing wandb: {e}. Disabling logging.")
             self.verbose = False
            
    def load_most_recent_checkpoint(self, fold_dir):
        #check for pt file 

        #if pt file exists, load the model and optimizer state
        checkpoint_path = fold_dir / 'checkpoint.pt'
        if checkpoint_path.exists():
            # Load checkpoint onto the correct device
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.device))
            
            # Load model state - handle potential architecture changes carefully
            try:
                # Direct load if architecture matches
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                 print(f"Warning: Error loading model state dict, likely architecture mismatch: {e}")
                 # Attempt partial load (load matching keys)
                 model_dict = self.model.state_dict()
                 pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
                 model_dict.update(pretrained_dict) 
                 self.model.load_state_dict(model_dict)
                 print(f"Partially loaded {len(pretrained_dict)} matching keys from checkpoint model state.")

            # Load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}. Optimizer will start from scratch.")

            self.current_epoch = checkpoint.get('epoch', 0) + 1 # Start from next epoch
            self.loss = checkpoint.get('loss', 1000) # Use training loss?
            self.best_dice = checkpoint.get('best_dice', 0.0) # Use best validation dice
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and hasattr(self, 'lr_scheduler'):
                 try:
                     self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 except Exception as e:
                     print(f"Warning: Could not load scheduler state: {e}. Scheduler will start from scratch.")

            print(f"Checkpoint found in {fold_dir}. Continuing training from epoch {self.current_epoch} with best validation Dice {self.best_dice:.4f}")
        else:
            print(f"WARNING: No checkpoint found at {checkpoint_path}. Training from scratch.")
            # Ensure variables are initialized correctly for starting fresh
            self.current_epoch = 0
            self.loss = 1000
            self.best_dice = 0.0
            # Re-initialize optimizer and scheduler states if needed? Usually done in __init__
    
    def sample_foreground_patch(self, data, seg, roi, patch_size=(32, 160, 256)):
        """
        DEPRECATED - Was: Sample a patch centered on a foreground voxel with higher priority for class 2
        and areas containing multiple classes. 
        Kept for reference, but should not be used if oversampling is disabled.
        
        Parameters:
        - data: Image data array (B, C, X, Y, Z)
        - seg: Segmentation mask array
        - roi: Region of interest mask
        - patch_size: Size of patch to extract
        
        Returns:
        - data_patch, seg_patch, roi_patch: Patches extracted
        """
        warnings.warn("sample_foreground_patch is deprecated and should not be used.", DeprecationWarning)
        # Fallback to random sampling if called accidentally
        return self.sample_random_patch(data, seg, roi, patch_size)
    
    def sample_random_patch(self, data, seg, roi, patch_size=(32, 160, 256)):
        """Sample random patches from the data batch"""
        batch_size = data.shape[0]
        img_channels = data.shape[1]
        seg_channels = seg.shape[1] # Should be 1 for integer masks
        roi_channels = roi.shape[1]
        
        # Initialize arrays to store patches
        data_patches = np.zeros((batch_size, img_channels, *patch_size), dtype=data.dtype)
        # Ensure segmentation patches are integer type if target is integer
        seg_patches = np.zeros((batch_size, seg_channels, *patch_size), dtype=seg.dtype) 
        roi_patches = np.zeros((batch_size, roi_channels, *patch_size), dtype=roi.dtype)
        
        for b in range(batch_size):
            data_shape = data.shape[2:] # X, Y, Z
            
            # Check if the data is smaller than the patch size in any dimension
            if any(ds < ps for ds, ps in zip(data_shape, patch_size)):
                # Calculate necessary padding
                padding = []
                for i in range(3):
                    pad_needed = max(0, patch_size[i] - data_shape[i])
                    # Distribute padding (mostly) evenly before and after
                    pad_before = pad_needed // 2
                    pad_after = pad_needed - pad_before
                    padding.append((pad_before, pad_after))
                
                # Pad the image, segmentation, and ROI
                # Use mode='constant' with default constant_values=0
                padded_data = np.pad(data[b], ((0,0), *padding), mode='constant')
                padded_seg = np.pad(seg[b], ((0,0), *padding), mode='constant')
                padded_roi = np.pad(roi[b], ((0,0), *padding), mode='constant')

                # Now data is large enough, sample a random patch from the padded data
                padded_shape = padded_data.shape[1:] # C, X, Y, Z -> X, Y, Z
                x_start = np.random.randint(0, padded_shape[0] - patch_size[0] + 1)
                y_start = np.random.randint(0, padded_shape[1] - patch_size[1] + 1)
                z_start = np.random.randint(0, padded_shape[2] - patch_size[2] + 1)

                data_patches[b] = padded_data[:, x_start:x_start+patch_size[0], y_start:y_start+patch_size[1], z_start:z_start+patch_size[2]]
                seg_patches[b] = padded_seg[:, x_start:x_start+patch_size[0], y_start:y_start+patch_size[1], z_start:z_start+patch_size[2]]
                roi_patches[b] = padded_roi[:, x_start:x_start+patch_size[0], y_start:y_start+patch_size[1], z_start:z_start+patch_size[2]]

            else:
                # Data is large enough, sample directly
                x_start = np.random.randint(0, data_shape[0] - patch_size[0] + 1)
                y_start = np.random.randint(0, data_shape[1] - patch_size[1] + 1)
                z_start = np.random.randint(0, data_shape[2] - patch_size[2] + 1)
                
                data_patches[b] = data[b, :, x_start:x_start+patch_size[0], 
                                     y_start:y_start+patch_size[1], 
                                     z_start:z_start+patch_size[2]]
                seg_patches[b] = seg[b, :, x_start:x_start+patch_size[0], 
                                   y_start:y_start+patch_size[1], 
                                   z_start:z_start+patch_size[2]]
                roi_patches[b] = roi[b, :, x_start:x_start+patch_size[0], 
                                   y_start:y_start+patch_size[1], 
                                   z_start:z_start+patch_size[2]]
                
        return data_patches, seg_patches, roi_patches
    
    def extract_training_patches(self, data_dict):
        """Extracts patches for training, prioritizing foreground sampling."""
        # Use attributes stored in self
        data_dict_sampled = self.sample_patch_foreground_based(
            data_dict, 
            patch_size=self.default_patch_size, # Use stored patch size
            num_samples=self.batch_size, 
            foreground_classes=[1, 2], 
            foreground_prob=self.foreground_prob, # Use stored foreground prob
            allow_empty=True
        )
        return data_dict_sampled
        
    def run_training(self):

        # --- Setup Output Directory & Logging ---
        # Use parent of dataset_dir if it follows structure .../DatasetName/train
        parent_dir = Path(self.dataset_dir).parent
        dataset_name = parent_dir.name if parent_dir and parent_dir.name else Path(self.dataset_dir).name
        
        self.fold_dir, log_file, checkpoint_exists = setup_output_directory(
            self.output_dir, self.fold, dataset_name, self.continue_tr
        )

        # --- Initialize WandB (now that fold_dir is known) ---
        if self.verbose:
            self.initialize_wandb()

        # --- Load Checkpoint if Continuing ---
        if self.continue_tr:
            # Pass fold_dir directly
            self.load_most_recent_checkpoint(self.fold_dir) 
            # Note: load_most_recent_checkpoint now handles the case where no checkpoint exists

        # --- Setup DataLoaders ---
        # Pass dataset_dir which points to the root containing train/validate/test
        self.train_loader, self.test_loader, self.val_files_fold, self.transforms = setup_dataloaders(
            dataset_dir=self.dataset_dir, 
            fold=self.fold, 
            fold_dir=self.fold_dir, 
            batch_size=self.batch_size, 
            use_roi=self.use_roi, 
            num_workers_train=self.num_workers_train, 
            num_workers_val=self.num_workers_val
        )

        # --- Start Training ---
        try:
            self.train_model()
        finally:
            # Ensure log file is closed and stdout restored
            if log_file != sys.stdout:
                 sys.stdout = sys.__stdout__ # Restore original stdout
                 log_file.close()
            if wandb and wandb.run:
                 wandb.finish()


    def calculate_roi(self, data, mask, roi, val=False):
        # Check if ROI mask is valid and contains foreground
        if roi is None or roi.sum() == 0:
            if self.verbose: print("ROI mask is empty or invalid, skipping ROI calculation.")
            return data, mask # Return original data if ROI is unusable

        coords = np.where(roi == 1)
        
        # Ensure coordinates were found
        if len(coords[0]) == 0:
             if self.verbose: print("ROI mask provided but no ROI voxels found, skipping ROI calculation.")
             return data, mask

        try:
            # Calculate bounding box (indices are C, D, H, W or B, C, D, H, W)
            # Assuming roi shape is (B, 1, D, H, W) or similar
            min_coords = [np.min(coords[i]) for i in range(len(coords)) if i > 1] # Skip Batch and Channel dims
            max_coords = [np.max(coords[i]) for i in range(len(coords)) if i > 1]

            # Ensure min/max calculation worked (list should have 3 elements)
            if len(min_coords) != 3 or len(max_coords) != 3:
                 print(f"Error: Could not determine 3D min/max coordinates from ROI shape {roi.shape}. Coords found: {coords}")
                 return data, mask

            # Add padding/adjustments based on patch size or multiples of 32 (SwinUNETR requirement)
            # This part needs careful review based on model/patching strategy
            
            # Example: Pad to be multiple of 32 for SwinUNETR compatibility during validation
            if val:
                current_dims = [max_c - min_c for min_c, max_c in zip(min_coords, max_coords)]
                padded_dims = [( (d + 31) // 32) * 32 for d in current_dims] # Round up to nearest multiple of 32

                pad_needed = [pd - cd for pd, cd in zip(padded_dims, current_dims)]
                
                # Distribute padding
                pad_before = [p // 2 for p in pad_needed]
                pad_after = [p - pb for p, pb in zip(pad_needed, pad_before)]
                
                final_min_coords = [max(0, mc - pb) for mc, pb in zip(min_coords, pad_before)]
                # Adjust max coords based on padding added before, ensuring we don't exceed original image bounds initially
                final_max_coords = [min(data.shape[i+2], mc + pa + (mc - fmc)) 
                                   for i, (mc, pa, fmc) in enumerate(zip(max_coords, pad_after, final_min_coords))]

                # Ensure the final crop size matches padded_dims
                final_dims = [fmax - fmin for fmin, fmax in zip(final_min_coords, final_max_coords)]
                
                # Apply final padding if needed after boundary adjustments
                final_padding = []
                for i in range(3):
                    f_pad_needed = max(0, padded_dims[i] - final_dims[i])
                    f_pad_before = f_pad_needed // 2
                    f_pad_after = f_pad_needed - f_pad_before
                    final_padding.append((f_pad_before, f_pad_after))
                
                # Crop first
                data = data[:, :, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]
                mask = mask[:, :, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]
                
                # Then pad if necessary
                if any(p[0] > 0 or p[1] > 0 for p in final_padding):
                    data = np.pad(data, ((0,0), (0,0), *final_padding), mode='constant')
                    mask = np.pad(mask, ((0,0), (0,0), *final_padding), mode='constant')

            else: # Training ROI calculation (simpler crop, patching handles size)
                 # Add a small margin around the ROI?
                 margin = 15 # Increased margin from 5 to 15
                 final_min_coords = [max(0, c - margin) for c in min_coords]
                 final_max_coords = [min(data.shape[i+2], c + margin) for i, c in enumerate(max_coords)]
                 
                 data = data[:, :, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]
                 mask = mask[:, :, final_min_coords[0]:final_max_coords[0], final_min_coords[1]:final_max_coords[1], final_min_coords[2]:final_max_coords[2]]
                
        except Exception as e:
            print(f"Error in calculating ROI for shape {data.shape} with ROI sum {roi.sum()}: {e}")
            # Return original data if error occurs
            return data, mask

        return data, mask

    
    def generate_patch_size(self, data_shape):
        """
        Generates a patch size for training, ensuring it's compatible with SwinUNETR 
        (multiple of 32) and fits within the data_shape.
        Currently uses the fixed default patch size and ensures it fits.
        Could be adapted for variable patch sizes later.

        Args:
            data_shape: The shape of the input data (after ROI selection) [D, H, W].
        
        Returns:
            tuple: The patch size (pd, ph, pw) to be used for sampling.
        """
        # Use the default patch size specified during initialization
        patch_size = self.default_patch_size

        # Ensure the patch size is not larger than the data dimensions
        adjusted_patch_size = [min(ds, ps) for ds, ps in zip(data_shape, patch_size)]

        # Ensure patch dimensions are multiples of 32 (SwinUNETR constraint)
        # Adjust *down* to the nearest multiple of 32 if necessary
        final_patch_size = [(ps // 32) * 32 for ps in adjusted_patch_size]
        
        # Handle cases where adjusted size becomes 0
        final_patch_size = [max(32, ps) for ps in final_patch_size] 

        return tuple(final_patch_size)


    def create_new_dataloader(self):
        # This seems redundant now that setup_dataloaders handles initialization
        # If needed for re-initializing during training, it should use the fold files
        warnings.warn("create_new_dataloader seems redundant and may be removed.", DeprecationWarning)
        
        # Re-use setup logic if absolutely necessary, but likely indicates design issue
        # train_loader, _, _, _ = setup_dataloaders(
        #     dataset_dir=self.dataset_dir, 
        #     fold=self.fold, 
        #     fold_dir=self.fold_dir, # Needs self.fold_dir to be set
        #     batch_size=self.batch_size, 
        #     use_roi=self.use_roi, 
        #     num_workers_train=self.num_workers_train, 
        #     num_workers_val=self.num_workers_val 
        # )
        # return train_loader
        return self.train_loader # Just return the existing one

    def adjust_learning_rate(self, epoch):
        """Adjusts learning rate based on warmup and scheduler."""
        current_lr = self.optimizer.param_groups[0]['lr']
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            new_lr = lr
        else:
            # Step the scheduler after warmup phase
            # Check if scheduler requires epoch or step based updates
            # CosineAnnealingWarmRestarts steps based on epoch by default
            self.lr_scheduler.step(epoch - self.warmup_epochs) # Pass relative epoch for scheduler
            new_lr = self.optimizer.param_groups[0]['lr'] # Get LR after scheduler step

        if self.verbose and new_lr != current_lr:
             print(f"Epoch {epoch}: LR adjusted from {current_lr:.8f} to {new_lr:.8f}")
        
        # Log LR to wandb if used
        if self.wandb_initialized and wandb:
            wandb.log({"train/learning_rate": new_lr}, step=epoch)


    def train_model(self):
        """Train the SwinUNETR model"""
        
        # Optimizer and scheduler are already initialized in __init__
        # Remove redundant optimizer/scheduler creation here
        
        # Initialize metrics tracking
        train_loss_history = []
        # Validation dice will be tracked epoch-wise
        
        print(f"\n--- Starting Training ---")
        print(f"Epochs: {self.num_epochs}, Train steps/epoch: {self.num_iterations_per_epoch}, Val steps/epoch: {self.num_val_iterations_per_epoch}")
        print(f"Initial LR: {self.initial_lr}, Weight Decay: {self.weight_decay}")
        print(f"Patch Size: {self.default_patch_size}, Batch Size: {self.batch_size}")
        print(f"Device: cuda:{self.device}")
        print(f"Output Directory: {self.fold_dir}")
        print(f"-------------------------\n")

        # --- Training Loop ---
        for epoch in range(self.current_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = 0
            step = 0
            
            # Adjust LR at the beginning of the epoch
            self.adjust_learning_rate(epoch)

            # Use the already initialized self.train_loader
            train_data_loader = self.train_loader 
            train_data_loader.restart() # Restart augmenter threads
            # train_data_loader.generator.shuffle_indices() # Done internally by dataloader? Check CustomDataLoader

            epoch_start_time = time.time()
            
            for i, batch_data in enumerate(train_data_loader):
                # Limit steps per epoch
                if i >= self.num_iterations_per_epoch:
                    break 
                
                step += 1
                data, seg = batch_data["data"], batch_data["seg"]
                roi = batch_data.get("roi", None)

                # Apply ROI cropping if enabled
                if self.use_roi and roi is not None:
                     data, seg = self.calculate_roi(data, seg, roi, val=False)
                
                # Create data dictionary to pass to extract_training_patches
                current_data_dict = {
                    'data': data, # Use potentially cropped data
                    'seg': seg,   # Use potentially cropped seg
                    'roi': roi    # Pass roi along (might be None)
                }
                
                # Generate appropriate patch size based on (potentially cropped) data
                # Note: patch size logic is now within extract_training_patches/sample_patch_foreground_based
                # current_data_shape = data.shape[2:] 
                # patch_size = self.generate_patch_size(current_data_shape)
                
                # Call extract_training_patches with the dictionary
                # It returns a dictionary containing sampled patches
                sampled_dict = self.extract_training_patches(current_data_dict)
                
                # Extract patches from the returned dictionary
                data_patch = sampled_dict['data']
                seg_patch = sampled_dict['seg']
                # roi_patch = sampled_dict.get('roi') # If ROI patches are also needed
                
                # Convert to tensors and move to device
                # Ensure correct types (float for data, long for seg)
                data_tensor = torch.from_numpy(data_patch).to(self.device).float()
                seg_tensor = torch.from_numpy(seg_patch).to(self.device).long()

                # --- Apply Input Normalization (Z-score per patch) ---
                mean = torch.mean(data_tensor, dim=[1, 2, 3, 4], keepdim=True)
                std = torch.std(data_tensor, dim=[1, 2, 3, 4], keepdim=True)
                data_tensor = (data_tensor - mean) / (std + 1e-6) # Add epsilon for stability

                # Sanity check target labels
                if not torch.all((seg_tensor >= 0) & (seg_tensor <= 2)):
                     print(f"Warning: Target tensor contains values outside [0, 1, 2] at step {step}, epoch {epoch}. Clamping.")
                     seg_tensor = torch.clamp(seg_tensor, 0, 2)
                
                self.optimizer.zero_grad()
                
                # --- Forward Pass ---
                outputs = self.model(data_tensor) # outputs is list if deep_supervision, else tensor
                
                # --- Loss Calculation ---
                if self.enable_deep_supervision and isinstance(outputs, (list, tuple)):
                    # Deep supervision enabled
                    total_loss = 0.0
                    final_output = outputs[0] # Main output for metric calculation
                    
                    for ds_idx, ds_output in enumerate(outputs):
                         # Resize target to match deep supervision output size
                         resized_target = F.interpolate(seg_tensor.float(), size=ds_output.shape[2:], mode='nearest').long()
                         
                         # Calculate loss for this level using the main criterion
                         # Or use self.deep_supervision_losses[ds_idx] if they are different
                         loss_level = self.criterion(ds_output, resized_target) 
                         
                         # Apply weight for this supervision level
                         weight = self.deep_supervision_weights[ds_idx]
                         total_loss += weight * loss_level
                         
                         # Store main output loss separately if needed for logging
                         if ds_idx == 0:
                             main_loss = loss_level.item()

                    loss = total_loss

                else:
                    # No deep supervision or model returns single tensor
                    final_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    # Resize target to match final output size if necessary
                    if final_output.shape[2:] != seg_tensor.shape[2:]:
                        resized_target = F.interpolate(seg_tensor.float(), size=final_output.shape[2:], mode='nearest').long()

                    else:
                        resized_target = seg_tensor
                    
                    loss = self.criterion(final_output, resized_target)
                    main_loss = loss.item()

                # Backward pass and optimizer step
                loss.backward()
                
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Track gradient norms (optional, for debugging)
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                self.optimizer.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                train_loss_history.append(current_loss)
                
                # Calculate per-class dice for logging (using non-resized target if shapes match)
                log_target = seg_tensor if final_output.shape[2:] == seg_tensor.shape[2:] else resized_target
                with torch.no_grad(): # No need to track gradients for dice calculation
                    batch_avg_dice, batch_class_dice_dict = self.calculate_dice(
                        final_output, log_target, per_class=True
                    )
                    # Extract average dice for class 1 and 2 for this batch
                    batch_dice_c1 = np.mean(batch_class_dice_dict['class_1']) # Avg over items in batch
                    batch_dice_c2 = np.mean(batch_class_dice_dict['class_2'])

                # --- Logging ---
                if self.verbose and (step % 50 == 0 or step == 1): # Log every 50 steps and first step
                    avg_recent_loss = np.mean(train_loss_history[-min(50, len(train_loss_history)):])
                    print(f"Epoch {epoch}/{self.num_epochs}, Step {step}/{self.num_iterations_per_epoch}: "
                          f"Loss: {current_loss:.4f} (Avg@50: {avg_recent_loss:.4f}), Grad Norm: {total_norm:.4f}, "
                          f"Batch Dice: {batch_avg_dice:.4f} (C1: {batch_dice_c1:.4f}, C2: {batch_dice_c2:.4f})")

                    # Log training loss and dice to wandb
                    if self.wandb_initialized and wandb:
                        wandb.log({
                            'train/step_loss': current_loss, 
                            'train/step_loss_avg50': avg_recent_loss,
                            'train/gradient_norm': total_norm,
                            'train/step_dice_overall': batch_avg_dice,
                            'train/step_dice_class1': batch_dice_c1,
                            'train/step_dice_class2': batch_dice_c2
                        }, step=epoch * self.num_iterations_per_epoch + step) # Global step

                # --- Calculate Metrics for Potential Image Saving ---
                sample_mae_overlap = np.nan # Default if error occurs
                try:
                    # Calculate MAE overlap specifically for the first sample in the batch (index 0)
                    with torch.no_grad():
                        probs_sample = torch.sigmoid(final_output[0:1]) # Get first sample's output, keep batch dim
                        prob1_sample = probs_sample[:, 0]
                        prob2_sample = probs_sample[:, 1]
                        fg_mask_sample = (prob1_sample > 0.1) | (prob2_sample > 0.1)
                        if fg_mask_sample.sum() > 0:
                            sample_mae_overlap = torch.abs(prob1_sample[fg_mask_sample] - prob2_sample[fg_mask_sample]).mean().item()
                        else:
                            sample_mae_overlap = 0.0
                except Exception as mae_e:
                    print(f"Warning: Failed to calculate sample MAE for debug image: {mae_e}")

                # --- Calculate Dice Metrics Specifically for Sample 0 (for debug image title) ---
                sample_dice_overall, sample_dice_c1, sample_dice_c2 = np.nan, np.nan, np.nan # Defaults
                try:
                    with torch.no_grad():
                        # Use the calculate_dice method on the first sample only
                        first_output = final_output[0:1] # Keep batch dimension
                        first_target = log_target[0:1]   # Keep batch dimension
                        _, sample_class_dice_dict = self.calculate_dice(first_output, first_target, per_class=True)
                        # Result is a dict with lists of length 1
                        sample_dice_overall = np.mean(sample_class_dice_dict['overall'])
                        sample_dice_c1 = np.mean(sample_class_dice_dict['class_1'])
                        sample_dice_c2 = np.mean(sample_class_dice_dict['class_2'])
                except Exception as sample_dice_e:
                    print(f"Warning: Failed to calculate sample Dice for debug image: {sample_dice_e}")

                # --- Save Intermediate Images (Conditional & Periodic) ---
                # Save every 50 steps OR if batch dice is high
                # Note: Condition still uses batch_avg_dice
                if (step % 50 == 0) or (batch_avg_dice > 0.9):
                     print(f"--- Saving debug image (Step {step}, Batch Dice {batch_avg_dice:.4f}, Sample Dice {sample_dice_overall:.4f}) ---")
                     # --- Get Filename for Sample 0 ---
                     filename_sample0 = "UnknownFile"
                     try:
                         # Assuming batch_data['keys'] holds filenames or identifiers
                         if 'keys' in batch_data and isinstance(batch_data['keys'], (list, tuple)) and len(batch_data['keys']) > 0:
                             filename_sample0 = Path(batch_data['keys'][0]).name # Get just the filename
                         elif 'data_name' in batch_data: # Fallback if 'keys' isn't there
                             filename_sample0 = Path(batch_data['data_name']).name 
                     except Exception as fname_e:
                         print(f"Warning: Could not get filename for debug image: {fname_e}")

                     self.save_debug_images(
                         epoch, step, data_tensor, final_output, seg_tensor,
                         sample_dice_overall, sample_dice_c1, sample_dice_c2, sample_mae_overlap, # Pass SAMPLE metrics
                         filename_sample0 # Pass filename
                     )

            # --- End of Epoch ---
            avg_epoch_loss = epoch_loss / step
            epoch_duration = time.time() - epoch_start_time
            print(f"--- Epoch {epoch} Summary ---")
            print(f"Average Training Loss: {avg_epoch_loss:.4f}")
            print(f"Duration: {epoch_duration:.2f} seconds")

            # --- Validation ---
            val_start_time = time.time()
            avg_val_dice, avg_val_loss = self.validate(epoch) # Pass epoch for logging
            val_duration = time.time() - val_start_time
            print(f"Average Validation Dice: {avg_val_dice:.4f}")
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Duration: {val_duration:.2f} seconds")

            # --- Checkpointing ---
            is_best = avg_val_dice > self.best_dice
            if is_best:
                self.best_dice = avg_val_dice
                print(f"** New best validation Dice: {self.best_dice:.4f} at epoch {epoch} **")
                save_path = self.fold_dir / 'model_best.pt'
            else:
                save_path = self.fold_dir / 'checkpoint.pt' # Save latest checkpoint

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'loss': avg_epoch_loss, # Save average epoch training loss
                'best_dice': self.best_dice, # Save best validation dice
                # Add any other info: args, validation file list etc.
            }
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")
            # Save latest checkpoint separately as well
            torch.save(checkpoint, self.fold_dir / 'checkpoint.pt')


            # --- WandB Logging (Epoch Level) ---
            if self.wandb_initialized and wandb:
                 # Log epoch metrics against the global step at the end of the epoch's training steps
                 global_step_end_epoch = (epoch + 1) * self.num_iterations_per_epoch
                 wandb.log({
                     'train/epoch_loss': avg_epoch_loss,
                     'val/epoch_dice': avg_val_dice,
                     'val/epoch_loss': avg_val_loss,
                     'epoch': epoch,
                     'val/best_dice': self.best_dice
                 }, step=global_step_end_epoch)

            print(f"---------------------------\n")

        print("--- Training Finished ---")


    def save_debug_images(self, epoch, step, image_tensor, output_tensor, target_tensor, sample_dice_overall, sample_dice_c1, sample_dice_c2, sample_mae_overlap, filename_sample0):
        """Saves slices of the input, output, and target for debugging, generating separate files for axial and coronal views with detailed labels, metrics, and filename."""
        image_dir = None
        fig_axial, fig_coronal = None, None # Define figs outside try block
        try:
            print(f"--- DEBUG: Entering save_debug_images for epoch {epoch}, step {step} ---")
            image_dir = self.fold_dir / "debug_images" / f"epoch_{epoch}"
            image_dir.mkdir(parents=True, exist_ok=True)
            print("--- DEBUG: Directory created/exists.")

            # --- Start: Modified Plotting Logic ---

            # Data Preparation
            try:
                input_img_np = image_tensor[0].squeeze().detach().cpu().numpy()
                target_label_np = target_tensor[0].squeeze().detach().cpu().numpy().astype(int)
                output_prob_np = torch.sigmoid(output_tensor[0]).squeeze().detach().cpu().numpy()
                print(f"--- DEBUG: Data prepared. Shapes: Input {input_img_np.shape}, Target {target_label_np.shape}, Output {output_prob_np.shape}")
                if np.isnan(output_prob_np).any() or np.isinf(output_prob_np).any():
                    print(f"--- WARNING: NaN or Inf detected in output probabilities! ---")
                    output_prob_np = np.nan_to_num(output_prob_np) # Replace NaN/Inf
            except Exception as data_prep_e:
                print(f"--- ERROR during data preparation: {data_prep_e} ---")
                return

            # --- Refined Slice Selection ---
            try:
                labels_in_vol = np.unique(target_label_np)
                artifact_labels = labels_in_vol[labels_in_vol > 0]

                if len(artifact_labels) == 0:
                    # No artifact present, use geometric center
                    mid_slice_d = input_img_np.shape[0] // 2
                    mid_slice_h = input_img_np.shape[1] // 2
                    artifact_label_str = "Label 0"
                else:
                    # Determine primary target class (if multiple, pick one, e.g., the first or dominant)
                    if len(artifact_labels) > 1:
                         artifact_label_str = f"Labels {list(artifact_labels)}"
                         # Optional: Choose dominant class based on pixel count
                         counts = [(lbl, np.sum(target_label_np == lbl)) for lbl in artifact_labels]
                         primary_target_class = max(counts, key=lambda item: item[1])[0]
                         print(f"--- DEBUG: Multiple labels detected. Focusing slice selection on dominant class: {primary_target_class}")
                    else:
                         artifact_label_str = f"Label {artifact_labels[0]}"
                         primary_target_class = artifact_labels[0]

                    # Find slice with max pixels for the primary target class
                    target_mask = (target_label_np == primary_target_class)
                    sum_pixels_d = np.sum(target_mask, axis=(1, 2)) # Sum over H, W for each D slice
                    sum_pixels_h = np.sum(target_mask, axis=(0, 2)) # Sum over D, W for each H slice

                    # Check if the target class actually exists before argmax
                    if np.max(sum_pixels_d) > 0:
                        mid_slice_d = np.argmax(sum_pixels_d)
                    else: # Fallback if primary class somehow has 0 pixels despite being detected
                         artifact_indices_d = np.where(np.any(target_label_np > 0, axis=(1, 2)))[0]
                         mid_slice_d = artifact_indices_d[len(artifact_indices_d) // 2] if len(artifact_indices_d) > 0 else input_img_np.shape[0] // 2
                         print(f"--- WARNING: Primary target class {primary_target_class} had no pixels? Using fallback slice D.")

                    if np.max(sum_pixels_h) > 0:
                        mid_slice_h = np.argmax(sum_pixels_h)
                    else: # Fallback
                         artifact_indices_h = np.where(np.any(target_label_np > 0, axis=(0, 2)))[0]
                         mid_slice_h = artifact_indices_h[len(artifact_indices_h) // 2] if len(artifact_indices_h) > 0 else input_img_np.shape[1] // 2
                         print(f"--- WARNING: Primary target class {primary_target_class} had no pixels? Using fallback slice H.")

                print(f"--- DEBUG: Slices selected (Max Pixels Target). Axial: {mid_slice_d}, Coronal: {mid_slice_h}, Target Label(s): {artifact_label_str}")
            except Exception as slice_sel_e:
                 print(f"--- ERROR during slice selection: {slice_sel_e} ---")
                 # Fallback to geometric center on error
                 mid_slice_d = input_img_np.shape[0] // 2
                 mid_slice_h = input_img_np.shape[1] // 2
                 artifact_label_str = "ErrorInSliceSelection"
                 print(f"--- DEBUG: Using geometric center slices due to error.")
            # --- End Refined Slice Selection ---

            # --- Format Metrics for Titles ---
            # Use SAMPLE metrics passed to the function
            metrics_str = f"Dice: {sample_dice_overall:.3f} (C1:{sample_dice_c1:.3f}, C2:{sample_dice_c2:.3f}) MAE: {sample_mae_overlap:.3f}"

            # Prepare Slice Data & Separate Target Masks
            try:
                target_class1_np = (target_label_np == 1).astype(float)
                target_class2_np = (target_label_np == 2).astype(float)

                slice_axial = input_img_np[mid_slice_d, :, :]
                target1_axial = target_class1_np[mid_slice_d, :, :]
                target2_axial = target_class2_np[mid_slice_d, :, :]
                prob1_axial = output_prob_np[0, mid_slice_d, :, :]
                prob2_axial = output_prob_np[1, mid_slice_d, :, :]

                slice_coronal = input_img_np[:, mid_slice_h, :]
                target1_coronal = target_class1_np[:, mid_slice_h, :]
                target2_coronal = target_class2_np[:, mid_slice_h, :]
                prob1_coronal = output_prob_np[0, :, mid_slice_h, :]
                prob2_coronal = output_prob_np[1, :, mid_slice_h, :]
            except Exception as slice_prep_e:
                print(f"--- ERROR preparing slice data: {slice_prep_e} ---")
                return

            # Calculate Argmax and Difference Maps
            try:
                argmax_output = np.argmax(output_prob_np, axis=0) + 1
                prob_sum_threshold = 0.1
                background_mask = output_prob_np.sum(axis=0) < prob_sum_threshold
                argmax_output[background_mask] = 0
                prob_diff = np.abs(output_prob_np[0] - output_prob_np[1])

                argmax_axial = argmax_output[mid_slice_d, :, :]
                diff_axial = prob_diff[mid_slice_d, :, :]
                argmax_coronal = argmax_output[:, mid_slice_h, :]
                diff_coronal = prob_diff[:, mid_slice_h, :]
                print("--- DEBUG: Argmax and Diff maps calculated.")
            except Exception as calc_e:
                 print(f"--- ERROR during argmax/diff calculation: {calc_e} ---")
                 return

            # --- Plotting Axial View ---
            try:
                fig_axial, axes_axial = plt.subplots(1, 7, figsize=(30, 5)) # 1 row, 7 columns
                # Add filename_sample0 to the title
                fig_axial.suptitle(f"Axial View - Step: {step}, Epoch: {epoch}, Slice: {mid_slice_d}, Target: {artifact_label_str}\nFile: {filename_sample0} | {metrics_str}", fontsize=11) # Reduced fontsize slightly
                print("--- DEBUG: Axial figure created.")

                # Panel 0: Input
                axes_axial[0].imshow(slice_axial, cmap='gray', origin='lower'); axes_axial[0].set_title('Input'); axes_axial[0].axis('off')
                # Panel 1: Target Class 1
                im_t1a = axes_axial[1].imshow(target1_axial, cmap='Reds', vmin=0, vmax=1, origin='lower'); axes_axial[1].set_title('Target Class 1'); axes_axial[1].axis('off')
                # Panel 2: Target Class 2
                im_t2a = axes_axial[2].imshow(target2_axial, cmap='Blues', vmin=0, vmax=1, origin='lower'); axes_axial[2].set_title('Target Class 2'); axes_axial[2].axis('off')
                # Panel 3: Prob Class 1
                im_p1a = axes_axial[3].imshow(prob1_axial, cmap='viridis', vmin=0, vmax=1, origin='lower'); axes_axial[3].set_title('Prob Class 1'); axes_axial[3].axis('off')
                # Panel 4: Prob Class 2
                im_p2a = axes_axial[4].imshow(prob2_axial, cmap='viridis', vmin=0, vmax=1, origin='lower'); axes_axial[4].set_title('Prob Class 2'); axes_axial[4].axis('off')
                # Panel 5: Argmax Output
                im_arga = axes_axial[5].imshow(argmax_axial, cmap='jet', vmin=0, vmax=2, origin='lower'); axes_axial[5].set_title('Argmax Output'); axes_axial[5].axis('off')
                # Panel 6: Difference
                im_diffa = axes_axial[6].imshow(diff_axial, cmap='magma', vmin=0, vmax=1, origin='lower'); axes_axial[6].set_title('Abs(Prob1-Prob2)'); axes_axial[6].axis('off')

                # Add colorbars
                fig_axial.colorbar(im_p1a, ax=axes_axial[3], shrink=0.8)
                fig_axial.colorbar(im_p2a, ax=axes_axial[4], shrink=0.8)
                fig_axial.colorbar(im_arga, ax=axes_axial[5], shrink=0.8, ticks=[0, 1, 2])
                fig_axial.colorbar(im_diffa, ax=axes_axial[6], shrink=0.8)

                plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout for longer title

                # Save Axial Figure
                # Add metrics to filename if it was a high-dice save?
                filename_tag = "_highDice" if sample_dice_overall > 0.9 else "" # Condition uses SAMPLE dice
                save_path_axial = image_dir / f"train_epoch_{epoch}_step_{step}{filename_tag}_axial.png"
                plt.savefig(save_path_axial)
                print(f"--- DEBUG: Successfully saved axial image to {save_path_axial} ---")

            except Exception as plot_axial_e:
                print(f"--- ERROR plotting/saving axial view: {plot_axial_e} ---")
            finally:
                if fig_axial: plt.close(fig_axial)
                print("--- DEBUG: Axial figure closed.")


            # --- Plotting Coronal View ---
            try:
                fig_coronal, axes_coronal = plt.subplots(1, 7, figsize=(30, 5)) # 1 row, 7 columns
                # Add filename_sample0 to the title
                fig_coronal.suptitle(f"Coronal View - Step: {step}, Epoch: {epoch}, Slice: {mid_slice_h}, Target: {artifact_label_str}\nFile: {filename_sample0} | {metrics_str}", fontsize=11) # Reduced fontsize slightly
                print("--- DEBUG: Coronal figure created.")

                # Panel 0: Input
                axes_coronal[0].imshow(slice_coronal, cmap='gray', origin='lower'); axes_coronal[0].set_title('Input'); axes_coronal[0].axis('off')
                # Panel 1: Target Class 1
                im_t1c = axes_coronal[1].imshow(target1_coronal, cmap='Reds', vmin=0, vmax=1, origin='lower'); axes_coronal[1].set_title('Target Class 1'); axes_coronal[1].axis('off')
                # Panel 2: Target Class 2
                im_t2c = axes_coronal[2].imshow(target2_coronal, cmap='Blues', vmin=0, vmax=1, origin='lower'); axes_coronal[2].set_title('Target Class 2'); axes_coronal[2].axis('off')
                # Panel 3: Prob Class 1
                im_p1c = axes_coronal[3].imshow(prob1_coronal, cmap='viridis', vmin=0, vmax=1, origin='lower'); axes_coronal[3].set_title('Prob Class 1'); axes_coronal[3].axis('off')
                # Panel 4: Prob Class 2
                im_p2c = axes_coronal[4].imshow(prob2_coronal, cmap='viridis', vmin=0, vmax=1, origin='lower'); axes_coronal[4].set_title('Prob Class 2'); axes_coronal[4].axis('off')
                # Panel 5: Argmax Output
                im_argc = axes_coronal[5].imshow(argmax_coronal, cmap='jet', vmin=0, vmax=2, origin='lower'); axes_coronal[5].set_title('Argmax Output'); axes_coronal[5].axis('off')
                # Panel 6: Difference
                im_diffc = axes_coronal[6].imshow(diff_coronal, cmap='magma', vmin=0, vmax=1, origin='lower'); axes_coronal[6].set_title('Abs(Prob1-Prob2)'); axes_coronal[6].axis('off')

                # Add colorbars
                fig_coronal.colorbar(im_p1c, ax=axes_coronal[3], shrink=0.8)
                fig_coronal.colorbar(im_p2c, ax=axes_coronal[4], shrink=0.8)
                fig_coronal.colorbar(im_argc, ax=axes_coronal[5], shrink=0.8, ticks=[0, 1, 2])
                fig_coronal.colorbar(im_diffc, ax=axes_coronal[6], shrink=0.8)

                plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout for longer title

                # Save Coronal Figure
                # Use same filename tag
                save_path_coronal = image_dir / f"train_epoch_{epoch}_step_{step}{filename_tag}_coronal.png"
                plt.savefig(save_path_coronal)
                print(f"--- DEBUG: Successfully saved coronal image to {save_path_coronal} ---")

            except Exception as plot_coronal_e:
                print(f"--- ERROR plotting/saving coronal view: {plot_coronal_e} ---")
            finally:
                if fig_coronal: plt.close(fig_coronal)
                print("--- DEBUG: Coronal figure closed.")

            # --- End: Modified Plotting Logic ---

        except Exception as outer_e:
            print(f"--- UNHANDLED ERROR in save_debug_images: {outer_e} ---")
            if fig_axial: plt.close(fig_axial)
            if fig_coronal: plt.close(fig_coronal)


    def training_metrics_report(self, epoch, step, loss_history, dice_history, 
                               class_presence_history, high_dice_low_pixels, 
                               loss_by_class, prediction_distribution, 
                               optimizer, class_pixel_percent):
        """Generate a comprehensive report on training metrics to evaluate model stability."""
        # This function needs significant simplification as many metrics were tied
        # to the old complex loss structure.
        
        # Calculate stability metrics
        recent_losses = loss_history[-min(50, len(loss_history)):]
        loss_std = np.std(recent_losses) if len(recent_losses) > 1 else 0
        
        # Get gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Basic Dice Score Tracking (using simplified history)
        avg_dice_report = {}
        dice_std_report = {}
        for class_idx in range(1, 3): # Classes 1 and 2
             history = dice_history.get(f'class_{class_idx}', []) # Use .get for safety
             recent_dice = history[-min(50, len(history)):]
             if recent_dice:
                 avg_dice_report[f'class_{class_idx}'] = np.mean(recent_dice)
                 dice_std_report[f'class_{class_idx}'] = np.std(recent_dice)
             else:
                  avg_dice_report[f'class_{class_idx}'] = 0.0
                  dice_std_report[f'class_{class_idx}'] = 0.0


        print("\n--- Training Metrics Report (Simplified) ---")
        print(f"Epoch {epoch}, Step {step}")
        print(f"- Loss (current): {recent_losses[-1]:.4f}" if recent_losses else "- Loss (current): N/A")
        print(f"- Loss std (last 50 steps): {loss_std:.4f}")
        print(f"- Gradient norm: {total_norm:.4f}")
        print(f"- Avg Dice @50: {avg_dice_report}")
        print(f"- Dice std @50: {dice_std_report}")
        
        # Log simplified metrics to wandb
        if self.wandb_initialized and wandb:
            log_dict = {
                f'train/report/loss_std': loss_std,
                f'train/report/gradient_norm': total_norm,
            }
            for cls, avg_d in avg_dice_report.items():
                log_dict[f'train/report/avg_dice_{cls}'] = avg_d
            for cls, std_d in dice_std_report.items():
                log_dict[f'train/report/dice_std_{cls}'] = std_d

            # Log report metrics against the current global training step
            wandb.log(log_dict, step=epoch * self.num_iterations_per_epoch + step) # Global step
        
        print("---------------------------------------")

    def calculate_dice(self, output, target, per_class=False):
        """Calculate Dice coefficient between prediction and target.
        Assumes output is (B, C=2, D, H, W) logits for class 1 and 2.
        Assumes target is (B, 1, D, H, W) integer labels (0, 1, 2).
        
        Args:
            output: Model output tensor (logits).
            target: Ground truth tensor (integers).
            per_class: If True, returns dice scores for each class (1 and 2) separately.
        """
        batch_size = target.shape[0]
        
        # Apply sigmoid to get probabilities per class channel
        output_prob = torch.sigmoid(output) # Shape (B, 2, D, H, W)
        
        # Threshold probabilities to get binary predictions for each class
        pred_binary_c1 = (output_prob[:, 0:1] > 0.5).float() # Predictions for class 1
        pred_binary_c2 = (output_prob[:, 1:2] > 0.5).float() # Predictions for class 2

        # Create binary targets for each class
        target_binary_c1 = (target == 1).float()
        target_binary_c2 = (target == 2).float()

        # Combine predictions and targets for overall foreground dice
        pred_fg = (pred_binary_c1 + pred_binary_c2) > 0 # Combine predictions
        target_fg = (target_binary_c1 + target_binary_c2) > 0 # Combine targets

        # Calculate overall foreground dice
        intersection_fg = torch.sum(pred_fg * target_fg, dim=[1, 2, 3, 4]).float()
        union_fg = torch.sum(pred_fg, dim=[1, 2, 3, 4]) + torch.sum(target_fg, dim=[1, 2, 3, 4])
        overall_dice = (2.0 * intersection_fg + 1e-6) / (union_fg + 1e-6) # Shape (B,)
        
        if per_class:
            dice_scores = {}
            # Class 1 Dice
            intersection_c1 = torch.sum(pred_binary_c1 * target_binary_c1, dim=[1, 2, 3, 4]).float()
            union_c1 = torch.sum(pred_binary_c1, dim=[1, 2, 3, 4]) + torch.sum(target_binary_c1, dim=[1, 2, 3, 4])
            dice_c1 = (2.0 * intersection_c1 + 1e-6) / (union_c1 + 1e-6) # Shape (B,)
            
            # Class 2 Dice
            intersection_c2 = torch.sum(pred_binary_c2 * target_binary_c2, dim=[1, 2, 3, 4]).float()
            union_c2 = torch.sum(pred_binary_c2, dim=[1, 2, 3, 4]) + torch.sum(target_binary_c2, dim=[1, 2, 3, 4])
            dice_c2 = (2.0 * intersection_c2 + 1e-6) / (union_c2 + 1e-6) # Shape (B,)

            # Store batch-wise scores
            dice_scores['overall'] = overall_dice.tolist()
            dice_scores['class_1'] = dice_c1.tolist()
            dice_scores['class_2'] = dice_c2.tolist()
            
            # Return the average overall dice and the dictionary of per-class batch scores
            return torch.mean(overall_dice).item(), dice_scores 
        else:
            # Return average overall dice across the batch
            return torch.mean(overall_dice).item()
            
            
    def validate(self, epoch):
        """Validate the model on the test dataset (as validation set is used for training)."""

        print("Running validation on Test Set...")
        self.model.eval()
        total_val_dice = 0.0
        total_val_loss = 0.0
        steps = 0
        
        # Store per-class dice results for averaging
        all_dice_c1 = []
        all_dice_c2 = []
        all_prec_c1 = [] # Precision Class 1
        all_rec_c1 = []  # Recall Class 1
        all_prec_c2 = [] # Precision Class 2
        all_rec_c2 = []  # Recall Class 2
        all_mae_overlap = []
        all_loss_dice_comp = [] # Dice component of loss
        all_loss_ce_comp = []   # CE component of loss

        # Use the test loader for validation
        val_loader = self.test_loader 
        val_loader.restart()
        
        with torch.no_grad():
            # Define a wrapper for the model to handle deep supervision output during inference
            def inference_predictor(x):
                model_output = self.model(x)
                # Return only the main output (index 0) if deep supervision is enabled
                if self.enable_deep_supervision and isinstance(model_output, (list, tuple)):
                    return model_output[0]
                else:
                    return model_output

            for step, data in enumerate(val_loader):
                if step >= self.num_val_iterations_per_epoch:
                    break

                # Get data
                inputs, labels = data['data'], data['seg'] # Match keys from CustomDataLoader
                roi = data.get('roi', None)

                # Apply ROI cropping and ensure data is tensor on correct device
                if self.use_roi and roi is not None:
                     roi_np = roi # Assuming roi is already numpy from dataloader if not tensor
                     # Convert tensors to numpy for calculate_roi
                     inputs_np = inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs
                     labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

                     inputs_np, labels_np = self.calculate_roi(inputs_np, labels_np, roi_np, val=True)

                     # Convert back to tensor on the correct device
                     inputs = torch.from_numpy(inputs_np).to(self.device).float()
                     labels = torch.from_numpy(labels_np).to(self.device).long()
                else:
                    # Ensure data is tensor on correct device if ROI not used
                    # Check if it's already a tensor (e.g., from pin_memory)
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.from_numpy(inputs).to(self.device).float()
                    else:
                        inputs = inputs.to(self.device) # Ensure it's on the right device

                    if not isinstance(labels, torch.Tensor):
                        labels = torch.from_numpy(labels).to(self.device).long()
                    else:
                        labels = labels.to(self.device) # Ensure it's on the right device

                # --- Sliding Window Inference for Validation ---
                # Use sliding window inference for potentially larger validation images
                outputs = sliding_window_inference(
                     inputs=inputs,
                     roi_size=self.default_patch_size, # Use training patch size for window
                     sw_batch_size=self.batch_size, # Process multiple windows per batch
                     predictor=inference_predictor, # Use the wrapper predictor
                     overlap=0.5, # Overlap ratio
                     mode="gaussian", # Blending mode
                     padding_mode="constant",
                     device=self.device,
                     progress=False # Disable progress bar for less verbose logging
                 ) # Output shape (B, 2, D, H, W)

                # --- Ensure label shape matches output shape ---
                if outputs.shape[2:] != labels.shape[2:]:
                     resized_labels = F.interpolate(labels.float(), size=outputs.shape[2:], mode='nearest').long()
                else:
                     resized_labels = labels

                # --- Calculate Loss ---
                # (Ensure outputs and resized_labels are available)
                loss = self.criterion(outputs, resized_labels)
                total_val_loss += loss.item()

                # --- Log Loss Components (if DiceCELoss) ---
                if isinstance(self.criterion.binary_loss, DiceCELoss):
                    try:
                        # Need to re-calculate loss components manually or access them if stored internally
                        # Easiest: Recalculate components using binary targets
                        # Note: Assumes loss uses sigmoid internally if needed
                        output_sig = torch.sigmoid(outputs)
                        target_c1_bin = (resized_labels == 1).float()
                        target_c2_bin = (resized_labels == 2).float()
                        
                        # Get lambda weights used
                        lambda_d = getattr(self.criterion.binary_loss, 'lambda_dice', 1.0)
                        lambda_c = getattr(self.criterion.binary_loss, 'lambda_ce', 1.0)

                        # Calculate Dice and CE for class 1
                        dice_loss_c1 = DiceLoss(sigmoid=False)(output_sig[:, 0:1], target_c1_bin)
                        ce_loss_c1 = nn.BCELoss()(output_sig[:, 0], target_c1_bin.squeeze(1)) # BCE expects (B, *) and (B, *)
                        
                        # Calculate Dice and CE for class 2
                        dice_loss_c2 = DiceLoss(sigmoid=False)(output_sig[:, 1:2], target_c2_bin)
                        ce_loss_c2 = nn.BCELoss()(output_sig[:, 1], target_c2_bin.squeeze(1))

                        # Average component losses based on presence (approximate how main loss does it)
                        # This is an approximation; a cleaner way would be to modify ArtifactCorrectionLoss
                        # to return components.
                        num_c1 = torch.sum(target_c1_bin).item()
                        num_c2 = torch.sum(target_c2_bin).item()
                        
                        avg_dice_comp = 0.0
                        avg_ce_comp = 0.0
                        count = 0
                        if num_c1 > 0:
                            avg_dice_comp += dice_loss_c1.item() * lambda_d * self.criterion.class_weights[0]
                            avg_ce_comp += ce_loss_c1.item() * lambda_c * self.criterion.class_weights[0]
                            count += 1
                        if num_c2 > 0:
                            avg_dice_comp += dice_loss_c2.item() * lambda_d * self.criterion.class_weights[1]
                            avg_ce_comp += ce_loss_c2.item() * lambda_c * self.criterion.class_weights[1]
                            count += 1
                        
                        if count > 0:
                            all_loss_dice_comp.append(avg_dice_comp / count)
                            all_loss_ce_comp.append(avg_ce_comp / count)
                        else: # Handle case where sample has no foreground
                            all_loss_dice_comp.append(0.0)
                            all_loss_ce_comp.append(0.0)

                    except Exception as loss_comp_e:
                        print(f"Warning: Failed to log loss components: {loss_comp_e}")
                        all_loss_dice_comp.append(np.nan)
                        all_loss_ce_comp.append(np.nan)
                
                # --- Calculate Dice, Precision, Recall Metrics ---
                # Get binary predictions and targets (can reuse from Dice calc?)
                with torch.no_grad():
                    output_prob = torch.sigmoid(outputs) # Sigmoid once
                    pred_binary_c1 = (output_prob[:, 0:1] > 0.5).float()
                    pred_binary_c2 = (output_prob[:, 1:2] > 0.5).float()
                    target_binary_c1 = (resized_labels == 1).float()
                    target_binary_c2 = (resized_labels == 2).float()
                    
                    # Overall Dice (already done in self.calculate_dice)
                    avg_batch_dice, batch_class_dice_dict = self.calculate_dice(outputs, resized_labels, per_class=True)
                    total_val_dice += avg_batch_dice
                    all_dice_c1.extend(batch_class_dice_dict['class_1'])
                    all_dice_c2.extend(batch_class_dice_dict['class_2'])
                    
                    # Calculate TP, FP, FN per class for Precision/Recall
                    eps = 1e-6
                    # Class 1
                    tp_c1 = torch.sum(pred_binary_c1 * target_binary_c1, dim=[1, 2, 3, 4]).float()
                    fp_c1 = torch.sum(pred_binary_c1 * (1 - target_binary_c1), dim=[1, 2, 3, 4]).float()
                    fn_c1 = torch.sum((1 - pred_binary_c1) * target_binary_c1, dim=[1, 2, 3, 4]).float()
                    prec_c1 = (tp_c1 + eps) / (tp_c1 + fp_c1 + eps)
                    rec_c1 = (tp_c1 + eps) / (tp_c1 + fn_c1 + eps)
                    all_prec_c1.extend(prec_c1.tolist())
                    all_rec_c1.extend(rec_c1.tolist())

                    # Class 2
                    tp_c2 = torch.sum(pred_binary_c2 * target_binary_c2, dim=[1, 2, 3, 4]).float()
                    fp_c2 = torch.sum(pred_binary_c2 * (1 - target_binary_c2), dim=[1, 2, 3, 4]).float()
                    fn_c2 = torch.sum((1 - pred_binary_c2) * target_binary_c2, dim=[1, 2, 3, 4]).float()
                    prec_c2 = (tp_c2 + eps) / (tp_c2 + fp_c2 + eps)
                    rec_c2 = (tp_c2 + eps) / (tp_c2 + fn_c2 + eps)
                    all_prec_c2.extend(prec_c2.tolist())
                    all_rec_c2.extend(rec_c2.tolist())

                steps += 1
                
                if self.verbose and (step % 10 == 0 or step == 1):
                    print(f"Validation step {step}/{self.num_val_iterations_per_epoch}: Batch Dice: {avg_batch_dice:.4f}, Batch Loss: {loss.item():.4f}")


        # Calculate final average metrics
        avg_val_dice = total_val_dice / steps if steps > 0 else 0.0
        avg_val_loss = total_val_loss / steps if steps > 0 else 0.0
        avg_dice_c1 = np.mean(all_dice_c1) if all_dice_c1 else 0.0
        avg_dice_c2 = np.mean(all_dice_c2) if all_dice_c2 else 0.0
        avg_prec_c1 = np.mean(all_prec_c1) if all_prec_c1 else 0.0 # Avg Precision C1
        avg_rec_c1 = np.mean(all_rec_c1) if all_rec_c1 else 0.0   # Avg Recall C1
        avg_prec_c2 = np.mean(all_prec_c2) if all_prec_c2 else 0.0 # Avg Precision C2
        avg_rec_c2 = np.mean(all_rec_c2) if all_rec_c2 else 0.0   # Avg Recall C2
        avg_mae_overlap = np.nanmean(all_mae_overlap) if 'all_mae_overlap' in locals() and all_mae_overlap else 0.0 # Use nanmean
        avg_loss_dice_comp = np.nanmean(all_loss_dice_comp) if all_loss_dice_comp else 0.0 # Avg Dice Loss Comp
        avg_loss_ce_comp = np.nanmean(all_loss_ce_comp) if all_loss_ce_comp else 0.0     # Avg CE Loss Comp
        
        print("\n--- Validation Summary ---")
        print(f"Average Dice (Overall): {avg_val_dice:.4f}")
        print(f"Average Dice (Class 1): {avg_dice_c1:.4f}")
        print(f"Average Dice (Class 2): {avg_dice_c2:.4f}")
        print(f"Average Loss: {avg_val_loss:.4f}")
        print(f"Average Prec/Rec (C1): {avg_prec_c1:.4f} / {avg_rec_c1:.4f}") # Print Prec/Rec C1
        print(f"Average Prec/Rec (C2): {avg_prec_c2:.4f} / {avg_rec_c2:.4f}") # Print Prec/Rec C2
        print(f"Average Prob MAE Overlap: {avg_mae_overlap:.4f}")
        if avg_loss_dice_comp > 0 or avg_loss_ce_comp > 0: # Only print if calculated
            print(f"Average Loss Components (Dice/CE): {avg_loss_dice_comp:.4f} / {avg_loss_ce_comp:.4f}")
        print("--------------------------")

        # Log detailed validation metrics to wandb
        if self.wandb_initialized and wandb:
            log_dict = {
                f'val/epoch_dice_overall': avg_val_dice,
                f'val/epoch_dice_class1': avg_dice_c1,
                f'val/epoch_dice_class2': avg_dice_c2,
                f'val/epoch_precision_c1': avg_prec_c1, # Log Prec C1
                f'val/epoch_recall_c1': avg_rec_c1,     # Log Rec C1
                f'val/epoch_precision_c2': avg_prec_c2, # Log Prec C2
                f'val/epoch_recall_c2': avg_rec_c2,     # Log Rec C2
                f'val/epoch_loss': avg_val_loss,
                f'val/prob_mae_overlap': avg_mae_overlap,
                f'val/epoch_loss_dice_comp': avg_loss_dice_comp, # Log Dice Loss Comp
                f'val/epoch_loss_ce_comp': avg_loss_ce_comp,   # Log CE Loss Comp
                'epoch': epoch
            }
            # Log validation metrics against the global step at the end of the corresponding training epoch
            global_step_end_epoch = (epoch + 1) * self.num_iterations_per_epoch
            wandb.log(log_dict, step=global_step_end_epoch)

        self.model.train() # Set model back to training mode
        return avg_val_dice, avg_val_loss # Return overall dice and loss
        
    def analyze_weighting_effectiveness(self, class_pixel_percent, prediction_distribution, loss_by_class):
        """
        DEPRECATED - Analysis based on old complex loss structure.
        Kept for reference.
        """
        warnings.warn("analyze_weighting_effectiveness is deprecated.", DeprecationWarning)
        return {} # Return empty dict

    def sample_patch_foreground_based(self, data_dict, patch_size, num_samples, foreground_classes=None,
                                      foreground_prob=0.5, allow_empty=False, margin=None):
        """Samples patches, prioritizing foreground regions based on foreground_prob."""
        data = data_dict['data']
        seg = data_dict['seg']
        roi = data_dict.get('roi') # Optional ROI

        batch_size_actual = data.shape[0]
        img_channels = data.shape[1]
        seg_channels = seg.shape[1] # Should be 1
        roi_channels = roi.shape[1] if roi is not None else 0

        data_patches = np.zeros((num_samples, img_channels, *patch_size), dtype=data.dtype)
        seg_patches = np.zeros((num_samples, seg_channels, *patch_size), dtype=seg.dtype)
        roi_patches = np.zeros((num_samples, roi_channels, *patch_size), dtype=roi.dtype) if roi is not None else None

        for i in range(num_samples):
            b = i % batch_size_actual 
            data_orig = data[b]
            seg_orig = seg[b, 0] 
            roi_orig = roi[b] if roi is not None else None
            data_shape = data_orig.shape[1:] 
            
            coords = None
            attempts = 0
            max_attempts = 100 

            sample_foreground = random.random() < foreground_prob

            while coords is None and attempts < max_attempts:
                attempts += 1
                if sample_foreground:
                    coords = self.sample_foreground_coordinate(seg_orig, patch_size, foreground_classes)
                    if coords is None and not allow_empty:
                         continue 
                    elif coords is None and allow_empty:
                         sample_foreground = False 

                if not sample_foreground: 
                    coords = self.sample_random_coordinate(data_shape, patch_size)
                
                if coords is None: 
                    if any(ds < ps for ds, ps in zip(data_shape, patch_size)):
                        coords = [0] * len(patch_size) 
                    else:
                        print(f"Warning: Failed to get sampling coordinates for shape {data_shape} and patch {patch_size}. Retrying.")
                        continue 
            
            if coords is None:
                 print(f"ERROR: Could not determine sampling coordinates after {max_attempts} attempts. Skipping sample {i}.")
                 continue

            starts = [c - ps // 2 for c, ps in zip(coords, patch_size)]
            ends = [st + ps for st, ps in zip(starts, patch_size)]

            pad_before = [max(0, -st) for st in starts]
            pad_after = [max(0, end - ds) for end, ds in zip(ends, data_shape)]
            padding = list(zip(pad_before, pad_after))

            crop_starts = [st + pb for st, pb in zip(starts, pad_before)]
            crop_ends = [end - pa for end, pa in zip(ends, pad_after)]

            data_slice = data_orig[:, crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1], crop_starts[2]:crop_ends[2]]
            seg_slice = seg_orig[crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1], crop_starts[2]:crop_ends[2]]
            roi_slice = roi_orig[:, crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1], crop_starts[2]:crop_ends[2]] if roi_orig is not None else None

            if any(p > 0 for p_pair in padding for p in p_pair):
                data_patches[i] = np.pad(data_slice, ((0, 0), *padding), mode='constant', constant_values=np.min(data_orig)) 
                seg_patches[i] = np.pad(seg_slice[np.newaxis, ...], ((0, 0), *padding), mode='constant', constant_values=0)
                if roi_patches is not None and roi_slice is not None:
                    roi_patches[i] = np.pad(roi_slice, ((0, 0), *padding), mode='constant', constant_values=0)
            else:
                data_patches[i] = data_slice
                seg_patches[i, 0] = seg_slice 
                if roi_patches is not None and roi_slice is not None:
                    roi_patches[i] = roi_slice

        return {'data': data_patches, 'seg': seg_patches, 'roi': roi_patches}

    def sample_foreground_coordinate(self, seg_mask, patch_size, foreground_classes):
        """Samples a random coordinate centered on a foreground voxel."""
        if foreground_classes is None:
            foreground_mask = seg_mask > 0
        else:
            foreground_mask = np.isin(seg_mask, foreground_classes)
            
        foreground_coords = np.argwhere(foreground_mask)
        if len(foreground_coords) == 0:
            return None
            
        center_idx = np.random.randint(len(foreground_coords))
        center_coords = foreground_coords[center_idx] 
        return center_coords

    def sample_random_coordinate(self, data_shape, patch_size):
        """Samples a random coordinate ensuring the patch fits (mostly)."""
        coords = [np.random.randint(0, ds) for ds in data_shape]
        return coords
    
    # ... rest of SWINUNETRTrainer class ...


    
def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser(description="Train SwinUNETR for 4DCT Artifact Correction")
    
    # --- Essential Arguments ---
    parser.add_argument('dataset_dir', type=str, 
                        help="Root directory containing the 'train', 'validate', 'test' subfolders with data files.")
    parser.add_argument('fold', type=str, 
                        help="Fold number (0-4) for 5-fold cross-validation.")
    parser.add_argument('--output_dir', type=str, default='./results', required=False,
                        help="Directory to save checkpoints, logs, and validation files.")
    parser.add_argument('--c', action='store_true', required=False, 
                        help="Continue training from the latest checkpoint in the fold's output directory.")
    parser.add_argument('--pretrained_weights', type=str, default=None, required=False, 
                        help="Path to pretrained model weights (.pt file) to load before training.")
    parser.add_argument('--device', type=int, required=False, default=0, 
                        help="GPU device ID to train on.")    
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size.') 
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for scheduler.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of linear warmup epochs.')
    parser.add_argument('--scheduler_T0', type=int, default=30, help='T_0 for CosineAnnealingWarmRestarts scheduler.')
    parser.add_argument('--scheduler_T_mult', type=int, default=2, help='T_mult for CosineAnnealingWarmRestarts scheduler.')
    
    # --- Model & Data Parameters ---
    parser.add_argument('--img_size', type=int, nargs=3, default=[64, 160, 256], help='Input image size (depth, height, width) for the model.')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 160, 256], help='Patch size for training.')
    parser.add_argument('--feature_size', type=int, default=24, help='Feature size for SwinUNETR.')
    parser.add_argument('--no_deep_supervision', action='store_false', dest='enable_deep_supervision', 
                        help='Disable deep supervision.')
    parser.add_argument('--use_roi', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable ROI cropping/padding during loading.')

    # --- Loss Parameters ---
    parser.add_argument('--loss_type', type=str, default='dicefocal', choices=['dice', 'tversky', 'focal', 'dicefocal', 'dicece'], help='Type of loss function for artifacts.')
    parser.add_argument('--loss_weight_c1', type=float, default=1.0, help='Weight for artifact class 1 in loss')
    parser.add_argument('--loss_weight_c2', type=float, default=1.0, help='Weight for artifact class 2 in loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for Focal/DiceFocal loss.')
    parser.add_argument('--tversky_alpha', type=float, default=0.5, help='Alpha for Tversky loss (weights FP).')
    parser.add_argument('--tversky_beta', type=float, default=0.5, help='Beta for Tversky loss (weights FN).')
    parser.add_argument('--lambda_dice', type=float, default=1.0, help='Weight for Dice component in DiceCE/DiceFocal loss.')
    parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for CE/Focal component in DiceCE/DiceFocal loss.')

    # --- Performance & Logging ---\
    parser.add_argument('--train_steps_per_epoch', type=int, default=500, help='Number of training steps per epoch.')
    parser.add_argument('--val_steps_per_epoch', type=int, default=250, help='Number of validation steps per epoch.')
    parser.add_argument('--num_workers_train', type=int, default=12, help='Number of workers for training data augmentation.')
    parser.add_argument('--num_workers_val', type=int, default=4, help='Number of workers for validation data augmentation.')
    parser.add_argument('--quiet', action='store_true', required=False,
                        help="Reduce verbosity and disable wandb logging.")
    parser.add_argument('--foreground_prob', type=float, default=0.8, help='Probability to sample foreground patches')

    args = parser.parse_args()

    # --- Validate arguments ---
    if not (args.fold.isdigit() and 0 <= int(args.fold) < 5):
         raise ValueError(f"Fold must be an integer between 0 and 4, got {args.fold}")
    args.img_size = tuple(args.img_size)
    args.patch_size = tuple(args.patch_size)
    if not Path(args.dataset_dir).is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {args.dataset_dir}")
    # Add more validation as needed

    # --- Create Trainer ---
    trainer = SWINUNETRTrainer(
        weights=args.pretrained_weights, 
        fold=args.fold, 
        dataset_dir=args.dataset_dir, 
        device=args.device, 
        continue_tr=args.c, 
        batch_size=args.batch_size,
        use_roi=args.use_roi,
        initial_lr=args.initial_lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        scheduler_T0=args.scheduler_T0,
        scheduler_T_mult=args.scheduler_T_mult,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        num_train_iterations=args.train_steps_per_epoch,
        num_val_iterations=args.val_steps_per_epoch,
        enable_deep_supervision=args.enable_deep_supervision,
        patch_size=args.patch_size,
        img_size=args.img_size,
        feature_size=args.feature_size,
        loss_type=args.loss_type,
        loss_weight_class1=args.loss_weight_c1,
        loss_weight_class2=args.loss_weight_c2,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        foreground_prob=args.foreground_prob, # Pass foreground_prob here
        lambda_dice=args.lambda_dice, # Pass lambda_dice
        lambda_ce=args.lambda_ce, # Pass lambda_ce
        verbose=not args.quiet
    )
    
    # Pass output_dir and worker counts to the trainer instance
    trainer.output_dir = args.output_dir 
    trainer.num_workers_train = args.num_workers_train
    trainer.num_workers_val = args.num_workers_val

    trainer.run_training()

if __name__ == '__main__':
    # Consider adding set_determinism for reproducibility if desired
    # set_determinism(seed=42) 
    run_training_entry()

