import torch
import os
from Dataloaders import CustomDataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from torch.optim import SGD
from torch import nn
from Augmentations import get_augmentations
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
from monai.losses import DiceLoss, TverskyLoss, FocalLoss, DiceFocalLoss, DiceCELoss, AsymmetricUnifiedFocalLoss
from training_utils.setup import setup_output_directory, setup_dataloaders
import argparse
from batchgenerators.transforms.abstract_transforms import Compose
from pathlib import Path # Use pathlib for robust path handling
from mpl_toolkits.axes_grid1 import make_axes_locatable # For colorbar
from monai.inferers import sliding_window_inference # Re-added for validate method
from monai.networks.utils import one_hot # Needed for AUFL workaround
from monai.utils.enums import LossReduction # Import LossReduction

# --- Import refactored components ---
from training_utils.patching import (
    calculate_roi,
    generate_patch_size,
    extract_training_patches,
    sample_patch_foreground_based
)
from training_utils.metrics import calculate_dice # Added import
from training_utils.logging_utils import (
    initialize_wandb,
    save_debug_images,
    training_metrics_report
) # Added logging imports
from training_utils.checkpointing import load_most_recent_checkpoint # Added checkpointing import
from training_utils.lr_scheduling import adjust_learning_rate # Added LR scheduling import
# --- End Import refactored components ---

try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. Logging will be disabled.")
    wandb = None

# Import the new loss
from losses.custom_losses import TverskyCELoss 

# Need BCEWithLogitsLoss for penalty term calculation potentially
from torch.nn import BCEWithLogitsLoss 

# --- Custom Tversky Focal Loss Definition ---
class TverskyFocalLoss(nn.Module):
    """
    Combines Tversky index with a focal scaling term.
    Loss = (1 - TverskyIndex)^gamma
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 2.0, # Focal exponent
        include_background: bool = True, # MONAI compatibility, but less relevant here
        sigmoid: bool = True, # Expect logits as input
        reduction: str = LossReduction.MEAN,
        smooth_nr: float = 1e-5, # Numerator smoothing
        smooth_dr: float = 1e-5, # Denominator smoothing
    ):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("Tversky alpha and beta must be non-negative.")
        # Relaxing the alpha + beta = 1 constraint, though it's common
        # if alpha + beta != 1.0:
        #      print(f"Warning: Tversky alpha ({alpha}) + beta ({beta}) != 1.0.")
        if gamma < 0:
            raise ValueError("Focal gamma must be non-negative.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.reduction = LossReduction(reduction)
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predictions from network (logits expected if sigmoid=True).
                    Shape [B, 1, D, H, W] for binary.
            y_true: Ground truth. Shape [B, 1, D, H, W] (binary 0/1).
        """
        if self.sigmoid:
            probs = torch.sigmoid(y_pred)
        else:
            probs = y_pred # Assume input is already probabilities

        # Ensure target is float and has same shape as probs
        y_true = y_true.float()
        if y_true.shape != probs.shape:
            # This might happen with deep supervision, though loss calculation handles resizing
            # For direct use, ensure shapes match or handle resizing externally.
            raise ValueError(f"Target shape {y_true.shape} does not match prediction shape {probs.shape}")

        # Flatten spatial/channel dimensions: B, N = B, C*D*H*W
        # Keep batch dimension separate for per-sample Tversky index
        probs_flat = probs.view(probs.shape[0], -1)
        y_true_flat = y_true.view(y_true.shape[0], -1)

        # Calculate TP, FP, FN per batch element
        tp = torch.sum(probs_flat * y_true_flat, dim=1)
        fp = torch.sum(probs_flat * (1 - y_true_flat), dim=1)
        fn = torch.sum((1 - probs_flat) * y_true_flat, dim=1)

        # Calculate Tversky Index per batch element
        # TI = (TP + smooth_nr) / (TP + alpha*FP + beta*FN + smooth_dr)
        tversky_index = (tp + self.smooth_nr) / (tp + self.alpha * fp + self.beta * fn + self.smooth_dr)

        # Calculate Tversky Focal Loss per batch element
        # Loss = (1 - TI)^gamma
        # Add clamp to prevent NaN gradients if tversky_index is exactly 1
        tversky_index_clamped = torch.clamp(tversky_index, max=1.0 - 1e-7)
        loss_per_sample = torch.pow(1.0 - tversky_index_clamped, self.gamma)

        # Apply reduction across batch
        if self.reduction == LossReduction.MEAN:
            return torch.mean(loss_per_sample)
        elif self.reduction == LossReduction.SUM:
            return torch.sum(loss_per_sample)
        elif self.reduction == LossReduction.NONE:
            return loss_per_sample
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class SWINUNETRTrainer(object):
    def __init__(self, 
                 weights_c1=None, # Separate weights for each model
                 weights_c2=None, 
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
                 feature_size=12,
                 optimizer_type='adamw', # Added optimizer type
                 momentum=0.99,         # Added SGD momentum
                 nesterov=True,         # Added SGD Nesterov flag
                 loss_type_c1='tverskyce', # Separate loss types per class
                 loss_type_c2='dicece',   
                 focal_gamma_c1=2.0, # Per-class gamma
                 focal_gamma_c2=2.0,
                 tversky_alpha_c1=0.3, # Default alpha for Class 1 Tversky
                 tversky_beta_c1=0.7,  # Default beta for Class 1 Tversky
                 tversky_alpha_c2=0.7, # Default alpha for Class 2 Tversky (higher FP penalty)
                 tversky_beta_c2=0.3,  # Default beta for Class 2 Tversky
                 foreground_prob=0.5, 
                 lambda_dice=1.0, # Reuse for Tversky/Dice weight in combined losses
                 lambda_focal_c1=1.0, # Per-class focal weight
                 lambda_focal_c2=1.0,
                 lambda_penalty=0.1, # Add penalty weight
                 aufl_delta=0.6,
                 aufl_weight=0.5,
                 lambda_hausdorff=0.0, # Add hausdorff weight
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
        self.best_avg_dice = 0.0 
        self.best_dice_c1 = 0.0
        self.best_dice_c2 = 0.0
        self.loss = 1000 # Track training loss
        self.dataset_dir = dataset_dir
        self.fold = fold
        self.continue_tr = continue_tr
        self.batch_size = batch_size
        self.use_roi = use_roi
        self.default_patch_size = patch_size 
        self.img_size = img_size 
        self.feature_size = feature_size 
        self.loss_type_c1 = loss_type_c1
        self.loss_type_c2 = loss_type_c2
        self.focal_gamma_c1 = focal_gamma_c1
        self.focal_gamma_c2 = focal_gamma_c2
        self.tversky_alpha_c1 = tversky_alpha_c1
        self.tversky_beta_c1 = tversky_beta_c1
        self.tversky_alpha_c2 = tversky_alpha_c2
        self.tversky_beta_c2 = tversky_beta_c2
        self.foreground_prob = foreground_prob 
        self.lambda_dice = lambda_dice # Shared weight for Dice/Tversky part (can be separated later if needed)
        self.lambda_focal_c1 = lambda_focal_c1 # Weight for Focal/CE part C1
        self.lambda_focal_c2 = lambda_focal_c2 # Weight for Focal/CE part C2
        self.lambda_penalty = lambda_penalty # Weight for exclusivity penalty
        self.aufl_delta = aufl_delta
        self.aufl_weight = aufl_weight
        self.lambda_hausdorff = lambda_hausdorff # Add hausdorff weight
        self.optimizer_type = optimizer_type.lower()
        self.momentum = momentum
        self.nesterov = nesterov

        # Store paths and worker counts (passed from run_training_entry)
        self.output_dir = None
        self.fold_dir = None # Will be set in run_training
        self.num_workers_train = None
        self.num_workers_val = None

        torch.device(f"cuda:{self.device}") 

        # --- Initialize Models (C1 and C2) ---
        self.model_c1 = SwinUNETR(
                    img_size=self.img_size, 
                    in_channels=1, 
            out_channels=1, # Binary output
                    feature_size=self.feature_size, 
                    deep_supervision=self.enable_deep_supervision, 
                    use_v2=True
                ).cuda(self.device)
            
        self.model_c2 = SwinUNETR(
            img_size=self.img_size, 
            in_channels=1, 
            out_channels=1, # Binary output
            feature_size=self.feature_size, 
            deep_supervision=self.enable_deep_supervision, 
            use_v2=True
        ).cuda(self.device)

        # --- Initialize Optimizers (Separate for now) --- 
        from torch.optim import AdamW, SGD # Import both optimizers
        if self.optimizer_type == 'adamw':
            self.optimizer_c1 = AdamW(
                self.model_c1.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            self.optimizer_c2 = AdamW(
                self.model_c2.parameters(),
                        lr=self.initial_lr,
                        weight_decay=self.weight_decay,
                        betas=(0.9, 0.999),
                        eps=1e-8
                    )
            print(f"Using AdamW optimizer with initial LR={self.initial_lr}, weight_decay={self.weight_decay}")
        elif self.optimizer_type == 'sgd':
            self.optimizer_c1 = SGD(
                self.model_c1.parameters(),
                lr=self.initial_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
            )
            self.optimizer_c2 = SGD(
                self.model_c2.parameters(),
                lr=self.initial_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
            )
            print(f"Using SGD optimizer with initial LR={self.initial_lr}, momentum={self.momentum}, nesterov={self.nesterov}, weight_decay={self.weight_decay}")
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # --- Load Pretrained Weights (if provided) ---
        if weights_c1 is not None:
            self._load_pretrained_weights(self.model_c1, weights_c1)
        if weights_c2 is not None:
            self._load_pretrained_weights(self.model_c2, weights_c2)
        
        # --- Checkpoint Loading (Handles loading both models/optimizers) ---
        if continue_tr:
            # Checkpoint loading happens *after* directory setup in run_training
            # The load function will need modification
            pass
            

        # --- Loss Function Initialization (Per Class) ---
        self.criterion_c1 = self._initialize_binary_loss(self.loss_type_c1, target_class=1)
        self.criterion_c2 = self._initialize_binary_loss(self.loss_type_c2, target_class=2)
        print(f"Initialized C1 loss: {self.loss_type_c1.upper()}, C2 loss: {self.loss_type_c2.upper()}")

        # Deep supervision weights (shared for both models if enabled)
        if self.enable_deep_supervision:
            self.deep_supervision_weights = np.array([1 / (2**i) for i in range(5)])
            self.deep_supervision_weights = self.deep_supervision_weights / self.deep_supervision_weights.sum()
        else:
             self.deep_supervision_weights = None


        # --- Scheduler Initialization (Separate) ---
        self.lr = self.initial_lr # Store initial LR, might need adjustment for logging

        self.lr_scheduler_c1 = CosineAnnealingWarmRestarts(
            self.optimizer_c1,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.min_lr
        )
        self.lr_scheduler_c2 = CosineAnnealingWarmRestarts(
            self.optimizer_c2,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.min_lr
        )
        
        # Defer wandb init until fold_dir is known in run_training
        self.wandb_initialized = False

    # Helper function to load weights
    def _load_pretrained_weights(self, model, weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(self.device))
            # Handle potential checkpoint structure vs raw weights
            if 'model_state_dict' in state_dict:
                 state_dict = state_dict['model_state_dict']
                 
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            # Log missing/extra keys if needed
            missing_keys = [k for k in model_dict if k not in pretrained_dict]
            extra_keys = [k for k in state_dict if k not in model_dict]
            if missing_keys: print(f"Warning: Missing keys in pretrained weights for {weights_path}: {missing_keys}")
            if extra_keys: print(f"Warning: Extra keys in pretrained weights for {weights_path}: {extra_keys}")

            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} matching keys from pretrained weights: {weights_path}")
        except Exception as e:
            print(f"Error loading pretrained weights from {weights_path}: {e}. Continuing without them.")

    # Helper function to initialize individual binary losses
    def _initialize_binary_loss(self, loss_type, target_class):
        print(f"Initializing loss for C{target_class}: {loss_type}")
        loss_type_lower = loss_type.lower() # Ensure case-insensitivity

        if loss_type_lower == 'dicece':
            # Note: Using default lambda_dice=1.0, lambda_ce=1.0 for MONAI's DiceCELoss
            # If you want to use self.lambda_dice/ce, pass them explicitly.
            return DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        elif loss_type_lower == 'tverskyce':
            lambda_tversky = self.lambda_dice # Use shared dice/tversky weight
            lambda_ce = self.lambda_focal_c1 if target_class == 1 else self.lambda_focal_c2 # Use class-specific CE/focal weight
            alpha = self.tversky_alpha_c1 if target_class == 1 else self.tversky_alpha_c2
            beta = self.tversky_beta_c1 if target_class == 1 else self.tversky_beta_c2
            print(f"  using TverskyCE (C{target_class}) alpha={alpha:.2f}, beta={beta:.2f}, lambda_tversky={lambda_tversky:.2f}, lambda_ce={lambda_ce:.2f}")
            return TverskyCELoss(
                to_onehot_y=False, 
                sigmoid=True, 
                alpha=alpha, 
                beta=beta,
                lambda_tversky=lambda_tversky,
                lambda_ce=lambda_ce
            )
        elif loss_type_lower == 'asym_unified_focal':
            # AUFL handles sigmoid internally, to_onehot_y=False because we do it manually
            # Using fixed delta/gamma for now, could be made configurable
            gamma = self.focal_gamma_c1 if target_class == 1 else self.focal_gamma_c2 # Use class-specific gamma
            print(f"  using Asymmetric Unified Focal delta={self.aufl_delta}, gamma={gamma}")
            return AsymmetricUnifiedFocalLoss(delta=self.aufl_delta, gamma=gamma, weight=self.aufl_weight, reduction='mean') # Pass gamma/delta/weight
        elif loss_type_lower == 'tverskyfocal': # Custom loss
            gamma = self.focal_gamma_c1 if target_class == 1 else self.focal_gamma_c2 # Use class-specific gamma
            alpha = self.tversky_alpha_c1 if target_class == 1 else self.tversky_alpha_c2
            beta = self.tversky_beta_c1 if target_class == 1 else self.tversky_beta_c2
            print(f"  using TverskyFocal (C{target_class}) alpha={alpha:.2f}, beta={beta:.2f}, focal gamma={gamma:.2f}")
            return TverskyFocalLoss(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                sigmoid=True, # Expect logits
                smooth_nr=1e-5,
                smooth_dr=1e-5
            )
        elif loss_type_lower == 'dicefocal': # Add missing case
            gamma = self.focal_gamma_c1 if target_class == 1 else self.focal_gamma_c2 # Use class-specific gamma
            lambda_focal = self.lambda_focal_c1 if target_class == 1 else self.lambda_focal_c2 # Use class-specific lambda_focal
            print(f"  using DiceFocal lambda_dice={self.lambda_dice}, lambda_focal={lambda_focal}, gamma={gamma}")
            return DiceFocalLoss(
                sigmoid=True,
                to_onehot_y=False,
                squared_pred=True,
                lambda_dice=self.lambda_dice,
                lambda_focal=lambda_focal, # Use class-specific focal weight
                gamma=gamma
            )
        else:
            raise ValueError(f"Unsupported loss type '{loss_type}' for C{target_class}")

        # --- Optionally add Hausdorff Loss ---
        if self.lambda_hausdorff > 0:
            print(f"  Adding Hausdorff Loss Component (lambda={self.lambda_hausdorff})")
            from monai.losses import HausdorffDTLoss
            # Hausdorff expects sigmoid=True if input is logits, to_onehot_y=False for binary
            hausdorff_loss = HausdorffDTLoss(sigmoid=True, to_onehot_y=False)
            
            # Return a tuple: (primary_criterion, hausdorff_loss)
            # The weight (self.lambda_hausdorff) will be applied in _calculate_loss
            return (primary_criterion, hausdorff_loss)
        else:
            # Return only the primary criterion if Hausdorff weight is zero
            return primary_criterion

    def train_model(self):
        """Train the SwinUNETR model"""
        
        # Optimizers and schedulers are already initialized in __init__
        
        # Initialize metrics tracking
        train_loss_history = []
        # Validation dice will be tracked epoch-wise
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Joint Training...") # Updated print
        print(f"Epochs: {self.num_epochs}, Train steps/epoch: {self.num_iterations_per_epoch}, Val steps/epoch: {self.num_val_iterations_per_epoch}")
        print(f"Initial LR: {self.initial_lr}, Weight Decay: {self.weight_decay}")
        print(f"Patch Size: {self.default_patch_size}, Batch Size: {self.batch_size}")
        print(f"Device: cuda:{self.device}")
        print(f"Output Directory: {self.fold_dir}")
        print(f"Loss C1: {self.loss_type_c1}, Loss C2: {self.loss_type_c2}, Penalty Lambda: {self.lambda_penalty}") # Added loss info
        print(f"-------------------------\n")

        # --- Training Loop ---
        for epoch in range(self.current_epoch, self.num_epochs):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Epoch {epoch} Training...") # Added print
            self.model_c1.train()
            self.model_c2.train()
            epoch_loss = 0
            epoch_loss_c1 = 0 # Track components
            epoch_loss_c2 = 0
            epoch_penalty = 0
            step = 0
            
            # Adjust LR for both models - need to update adjust_learning_rate or call schedulers directly
            # adjust_learning_rate(self, epoch) # This needs modification for two schedulers
            # Simple step call for CosineAnnealingWarmRestarts (assumes called per epoch)
            self.lr_scheduler_c1.step(epoch)
            self.lr_scheduler_c2.step(epoch)
            lr_c1 = self.optimizer_c1.param_groups[0]['lr']
            lr_c2 = self.optimizer_c2.param_groups[0]['lr']
            if self.verbose and wandb and self.wandb_initialized: # Log LRs
                 wandb.log({'train/learning_rate_c1': lr_c1, 'train/learning_rate_c2': lr_c2, 'epoch': epoch}, step=epoch * self.num_iterations_per_epoch)
            print(f"Epoch {epoch}: LR C1={lr_c1:.6f}, LR C2={lr_c2:.6f}")

            # Use the already initialized self.train_loader
            train_data_loader = self.train_loader 
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ---> Attempting train_data_loader.restart() for Epoch {epoch}...") # Added print
            sys.stdout.flush() # Force write to log file
            train_data_loader.restart() 
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] <--- train_data_loader.restart() finished for Epoch {epoch}.") # Added print
            sys.stdout.flush() # Force write to log file

            epoch_start_time = time.time()
            
            for i, batch_data in enumerate(train_data_loader):
                # Limit steps per epoch
                if i >= self.num_iterations_per_epoch:
                    break 
                
                # Handle potentially empty batches from dataloader filtering
                if not batch_data or 'data' not in batch_data or batch_data['data'].shape[0] == 0:
                     print(f"Warning: Skipping empty batch at step {i+1}")
                     continue
                
                step += 1
                data, seg_original = batch_data["data"], batch_data["seg"] # Original seg (0, 1, 2)
                roi = batch_data.get("roi", None)
                filename_keys = batch_data.get("keys", []) # Get filenames

                # --- Process Batch Sample by Sample (Patching) ---
                # This part remains similar, using original seg for sampling logic
                data_patches_list = []
                seg_patches_list = [] # Still holds original labels (0,1,2) after patching
                actual_batch_size = data.shape[0]
                
                for b in range(actual_batch_size):
                    data_sample = data[b:b+1]
                    seg_sample = seg_original[b:b+1]
                    roi_sample = roi[b:b+1] if roi is not None else None
                    keys_sample = [filename_keys[b]] if filename_keys and len(filename_keys) > b else [f"unknown_sample_{b}"]

                    if self.use_roi and roi_sample is not None:
                        # Pass patch_size for potential validation padding logic reuse if needed?
                        # For training, calculate_roi doesn't use patch_size.
                        data_sample_roi, seg_sample_roi = calculate_roi(
                            data_sample, seg_sample, roi_sample, 
                            patch_size=self.default_patch_size, # Pass anyway
                            val=False, verbose=self.verbose
                        )
                    else:
                        data_sample_roi, seg_sample_roi = data_sample, seg_sample
                 
                    current_data_dict_sample = {
                        'data': data_sample_roi,
                        'seg': seg_sample_roi,
                        'roi': roi_sample, 
                        'keys': keys_sample 
                    }
                 
                    current_data_shape_sample = data_sample_roi.shape[2:] 
                    # Generate patch size based on sample, but use default for sampling call?
                    # Let's use default_patch_size for consistency in sampling
                    # patch_size_for_sampling = generate_patch_size(
                    #     current_data_shape_sample, 
                    #     default_patch_size=self.default_patch_size
                    # )
 
                    # Sample patch using original segmentation for foreground logic
                    sampled_dict_sample = sample_patch_foreground_based(
                        current_data_dict_sample,
                        patch_size=self.default_patch_size, 
                        num_samples=1, 
                        foreground_classes=[1, 2], # Base sampling on presence of *any* artifact
                        foreground_prob=self.foreground_prob,
                        allow_empty=True 
                    )
                 
                    data_patches_list.append(sampled_dict_sample['data'])
                    seg_patches_list.append(sampled_dict_sample['seg']) # Contains 0,1,2
                
                # --- Concatenate collected patches --- 
                if not data_patches_list:
                    print(f"Warning: No patches were generated for step {step}, epoch {epoch}. Skipping step.")
                    continue 
                    
                data_patch_batch = np.concatenate(data_patches_list, axis=0)
                seg_patch_batch_original = np.concatenate(seg_patches_list, axis=0)
                 
                # Convert final batch numpy arrays to tensors and move to device
                data_tensor = torch.from_numpy(data_patch_batch).to(self.device).float()
                seg_tensor_original = torch.from_numpy(seg_patch_batch_original).to(self.device).long()

                # --- Create Binary Targets --- 
                target_c1 = (seg_tensor_original == 1).float()
                target_c2 = (seg_tensor_original == 2).float()
                
                if not torch.any(target_c1>0) and not torch.any(target_c2>0):
                    print(f"Warning: No foreground classes in target. Skipping step.")
                    continue

                # --- Input Normalization --- 
                mean = torch.mean(data_tensor, dim=[1, 2, 3, 4], keepdim=True)
                std = torch.std(data_tensor, dim=[1, 2, 3, 4], keepdim=True)
                data_tensor = (data_tensor - mean) / (std + 1e-6) 
                
                # --- Zero Gradients --- 
                self.optimizer_c1.zero_grad()
                self.optimizer_c2.zero_grad()
                
                # --- Forward Passes --- 
                outputs_c1 = self.model_c1(data_tensor) 
                outputs_c2 = self.model_c2(data_tensor)
                
                # --- Loss Calculation ---
                # Calculate individual losses (C1 and C2) including deep supervision if enabled
                loss_c1, final_output_c1 = self._calculate_loss(outputs_c1, target_c1, self.criterion_c1)
                loss_c2, final_output_c2 = self._calculate_loss(outputs_c2, target_c2, self.criterion_c2)

                # --- Mutual Exclusivity Penalty (Revised Logic) --- 
                # Use final outputs (highest resolution) for penalty calculation
                prob_c1 = torch.sigmoid(final_output_c1)
                prob_c2 = torch.sigmoid(final_output_c2)
                # Ensure shapes match (should be guaranteed by model structure)
                if prob_c1.shape != prob_c2.shape:
                    print(f"Warning: Output shapes mismatch for penalty calc: C1={prob_c1.shape}, C2={prob_c2.shape}. Skipping penalty.")
                    penalty = torch.tensor(0.0, device=self.device)
                else:
                    # Create individual binary masks for true locations
                    true_mask_c1 = (seg_tensor_original == 1).float()
                    true_mask_c2 = (seg_tensor_original == 2).float()

                    # Define dilation parameters (larger kernel for ~15 voxel padding)
                    # K=31 -> (31-1)/2 = 15 voxel padding
                    dilation_kernel_size = 31 
                    padding_size = dilation_kernel_size // 2

                    # Dilate each mask individually using Max Pooling approximation
                    # Wrap in try-except for potentially empty masks
                    try:
                        dilated_mask_c1 = F.max_pool3d(
                            true_mask_c1,
                            kernel_size=dilation_kernel_size,
                            stride=1,
                            padding=padding_size
                        )
                        # Threshold back to binary mask
                        penalty_zone_c1 = (dilated_mask_c1 > 0).float()
                    except Exception as e:
                         print(f"Warning: Max pooling failed for C1 mask (likely empty?): {e}. Setting zone to zeros.")
                         penalty_zone_c1 = torch.zeros_like(true_mask_c1)

                    try:
                        dilated_mask_c2 = F.max_pool3d(
                            true_mask_c2,
                            kernel_size=dilation_kernel_size,
                            stride=1,
                            padding=padding_size
                        )
                        # Threshold back to binary mask
                        penalty_zone_c2 = (dilated_mask_c2 > 0).float()
                    except Exception as e:
                         print(f"Warning: Max pooling failed for C2 mask (likely empty?): {e}. Setting zone to zeros.")
                         penalty_zone_c2 = torch.zeros_like(true_mask_c2)

                    # --- Refine Penalty Zones --- 
                    # Exclude regions where the *other* class is actually present
                    refined_penalty_zone_c1 = penalty_zone_c1 * (1 - true_mask_c2) # Zone where C2 is penalized
                    refined_penalty_zone_c2 = penalty_zone_c2 * (1 - true_mask_c1) # Zone where C1 is penalized

                    # --- Calculate Penalties using Refined Zones --- 
                    # Penalize C1 if it predicts inside C2's refined zone
                    penalty_on_c1_map = prob_c1 * refined_penalty_zone_c2
                    # Penalize C2 if it predicts inside C1's refined zone
                    penalty_on_c2_map = prob_c2 * refined_penalty_zone_c1

                    # Calculate average penalty within the respective refined zones
                    # Add epsilon to avoid division by zero if a zone is empty
                    sum_refined_zone_c1 = torch.sum(refined_penalty_zone_c1) + 1e-6 # Size of zone where C2 is penalized
                    sum_refined_zone_c2 = torch.sum(refined_penalty_zone_c2) + 1e-6 # Size of zone where C1 is penalized

                    avg_penalty_on_c1 = torch.sum(penalty_on_c1_map) / sum_refined_zone_c2 # Normalize by C2's refined zone size
                    avg_penalty_on_c2 = torch.sum(penalty_on_c2_map) / sum_refined_zone_c1 # Normalize by C1's refined zone size

                    # Combine: Average the two normalized penalties for fairness
                    penalty = (avg_penalty_on_c1 + avg_penalty_on_c2) / 2.0

                # --- Total Loss --- 
                #loss_c1 = loss_c1 if torch.any(target_c1[:]>0) else torch.tensor(0.0, device=self.device)
                #loss_c2 = loss_c2 if torch.any(target_c2[:]>0) else torch.tensor(0.0, device=self.device)
                total_loss = loss_c1 + loss_c2

                # --- Backward Pass & Optimizer Steps --- 
                total_loss.backward()
                
                # Optional: Gradient clipping (apply per optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model_c1.parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(self.model_c2.parameters(), max_norm=1.0)

                # Gradient norms (separate)
                total_norm_c1 = self._get_gradient_norm(self.model_c1)
                total_norm_c2 = self._get_gradient_norm(self.model_c2)
                
                self.optimizer_c1.step()
                self.optimizer_c2.step()
                
                # --- Post-Step Processing --- 
                current_loss_c1 = loss_c1.item()
                current_loss_c2 = loss_c2.item()
                current_penalty = penalty.item()
                current_total_loss = total_loss.item()
                
                epoch_loss += current_total_loss
                epoch_loss_c1 += current_loss_c1
                epoch_loss_c2 += current_loss_c2
                epoch_penalty += current_penalty
                train_loss_history.append(current_total_loss)
                
                # --- Metrics Calculation (Dice per class) ---
                # Use final outputs and corresponding binary targets
                with torch.no_grad(): 
                    batch_avg_dice_c1 = calculate_dice(final_output_c1, target_c1)
                    batch_avg_dice_c2 = calculate_dice(final_output_c2, target_c2)

                # --- Logging ---
                # Log every 10 steps and first step
                if self.verbose and (step % 10 == 0 or step == 1):
                    avg_recent_loss = np.mean(train_loss_history[-min(10, len(train_loss_history)):]) # Adjust avg window too
                    print(f"Epoch {epoch}/{self.num_epochs}, Step {step}/{self.num_iterations_per_epoch}: ")
                    print(f"  Loss C1: {current_loss_c1:.4f}, Loss C2: {current_loss_c2:.4f}, Penalty: {current_penalty:.4f}, Total: {current_total_loss:.4f} (Avg@10: {avg_recent_loss:.4f})") # Updated avg window
                    print(f"  Dice C1: {batch_avg_dice_c1:.4f}, GradNorm C1: {total_norm_c1:.4f}")
                    print(f"  Dice C2: {batch_avg_dice_c2:.4f}, GradNorm C2: {total_norm_c2:.4f}")

                    if self.wandb_initialized and wandb:
                        log_dict = {
                            'train/step_loss_total': current_total_loss,
                            'train/step_loss_c1': current_loss_c1,
                            'train/step_loss_c2': current_loss_c2,
                            'train/step_penalty': current_penalty * self.lambda_penalty, # Log weighted penalty
                            'train/step_loss_avg10': avg_recent_loss,
                            'train/gradient_norm_C1': total_norm_c1,
                            'train/gradient_norm_C2': total_norm_c2,
                            'train/step_dice_C1': batch_avg_dice_c1,
                            'train/step_dice_C2': batch_avg_dice_c2
                        }
                        wandb.log(log_dict, step=epoch * self.num_iterations_per_epoch + step) 

                # --- Debug Image Saving ---
                # Calculate sample dice for title
                sample_dice_overall_c1 = np.nan # Default
                sample_dice_overall_c2 = np.nan # Default
                try:
                    with torch.no_grad():
                        first_output_c1 = final_output_c1[0:1] 
                        first_target_c1 = target_c1[0:1] # Use the binary target
                        sample_dice_overall_c1 = calculate_dice(first_output_c1, first_target_c1)
                        first_output_c2 = final_output_c2[0:1] 
                        first_target_c2 = target_c2[0:1] # Use the binary target
                        sample_dice_overall_c2 = calculate_dice(first_output_c2, first_target_c2)
                except Exception as sample_dice_e:
                    print(f"Warning: Failed to calculate sample Dice for debug image: {sample_dice_e}")

                # Conditional saving
                if (step % 10 == 0) or (sample_dice_overall_c1 > 0.9 or sample_dice_overall_c2 > 0.9):
                     print(f"--- Saving debug image (Step {step}, Sample Dice C1: {sample_dice_overall_c1:.4f}, C2: {sample_dice_overall_c2:.4f}) ---")
                     filename_sample0 = "UnknownFile"
                     try:
                         if filename_keys and len(filename_keys) > 0:
                             filename_sample0 = Path(filename_keys[0]).name
                     except Exception as fname_e:
                         print(f"Warning: Could not get filename for debug image: {fname_e}")

                     # Call the refactored function
                     save_debug_images(
                         self.fold_dir, 
                         epoch, step, data_tensor, 
                         final_output_c1, target_c1, 
                         final_output_c2, target_c2,
                         sample_dice_overall_c1, sample_dice_overall_c2,
                         filename_sample0
                     )

            # --- End of Epoch ---
            avg_epoch_loss = epoch_loss / step if step > 0 else 0
            avg_epoch_loss_c1 = epoch_loss_c1 / step if step > 0 else 0
            avg_epoch_loss_c2 = epoch_loss_c2 / step if step > 0 else 0
            avg_epoch_penalty = epoch_penalty / step if step > 0 else 0
            epoch_duration = time.time() - epoch_start_time
            print(f"--- Epoch {epoch} Summary ---")
            print(f"Avg Train Loss: Total={avg_epoch_loss:.4f} (C1={avg_epoch_loss_c1:.4f}, C2={avg_epoch_loss_c2:.4f}, Penalty={avg_epoch_penalty:.4f})")
            print(f"Duration: {epoch_duration:.2f} seconds")

            # --- Validation (Needs Adaptation) ---
            # validate method needs refactoring to handle both models
            # We'll call it twice for now, once per effective target class
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting validation for Epoch {epoch}...") # Added print
            val_start_time = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ---> Validating Class 1...") # Added print
            avg_val_dice_c1, avg_val_loss_c1 = self.validate(epoch, 1) 
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ---> Validating Class 2...") # Added print
            avg_val_dice_c2, avg_val_loss_c2 = self.validate(epoch, 2)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] <--- Validation finished for Epoch {epoch}.") # Added print
            val_duration = time.time() - val_start_time
            print(f"Avg Val Dice (C1/C2): {avg_val_dice_c1:.4f} / {avg_val_dice_c2:.4f}")
            print(f"Avg Val Loss (C1/C2): {avg_val_loss_c1:.4f} / {avg_val_loss_c2:.4f}")
            print(f"Validation Duration: {val_duration:.2f} seconds")

            # --- Checkpointing (Save best based on average dice?) ---
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting checkpoint saving for Epoch {epoch}...")
            current_avg_dice = (avg_val_dice_c1 + avg_val_dice_c2) / 2.0
            is_best_avg = current_avg_dice > self.best_avg_dice

            # Update best average dice score if current is better
            if is_best_avg:
                print(f"  New best average validation dice: {current_avg_dice:.4f} (previous: {self.best_avg_dice:.4f})")
                self.best_avg_dice = current_avg_dice
            else:
                print(f"  Current average validation dice: {current_avg_dice:.4f} (best: {self.best_avg_dice:.4f})")

            # --- Always save the latest checkpoint ---
            latest_save_path_c1 = self.fold_dir / 'checkpoint_C1.pt'
            latest_save_path_c2 = self.fold_dir / 'checkpoint_C2.pt'
            
            checkpoint_latest_c1 = {
                'epoch': epoch,
                'state_dict': self.model_c1.state_dict(),
                'optimizer_state_dict': self.optimizer_c1.state_dict(),
                'scheduler_state_dict': self.lr_scheduler_c1.state_dict(),
                'best_avg_dice': self.best_avg_dice # Store the *overall* best score
            }
            torch.save(checkpoint_latest_c1, latest_save_path_c1)
            print(f"  Latest Checkpoint C1 saved to {os.path.basename(latest_save_path_c1)}")

            checkpoint_latest_c2 = {
                'epoch': epoch,
                'state_dict': self.model_c2.state_dict(),
                'optimizer_state_dict': self.optimizer_c2.state_dict(),
                'scheduler_state_dict': self.lr_scheduler_c2.state_dict(),
                'best_avg_dice': self.best_avg_dice # Store the *overall* best score
            }
            torch.save(checkpoint_latest_c2, latest_save_path_c2)
            print(f"  Latest Checkpoint C2 saved to {os.path.basename(latest_save_path_c2)}")
            
            # --- Save the best checkpoint if this epoch was the best so far ---
            if is_best_avg:
                best_save_path_c1 = self.fold_dir / 'checkpoint_C1_best.pt'
                best_save_path_c2 = self.fold_dir / 'checkpoint_C2_best.pt'
                
                # Create the same checkpoint dictionaries as above (or copy them)
                checkpoint_best_c1 = checkpoint_latest_c1.copy()
                checkpoint_best_c2 = checkpoint_latest_c2.copy()
                
                torch.save(checkpoint_best_c1, best_save_path_c1)
                print(f"  Best Checkpoint C1 saved to {os.path.basename(best_save_path_c1)}")
                torch.save(checkpoint_best_c2, best_save_path_c2)
                print(f"  Best Checkpoint C2 saved to {os.path.basename(best_save_path_c2)}")

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] <--- Checkpoint saving finished for Epoch {epoch}.")

            # --- WandB Logging (Epoch Level) ---
            if self.wandb_initialized and wandb:
                 global_step_end_epoch = (epoch + 1) * self.num_iterations_per_epoch
                 log_dict = {
                     'train/epoch_loss_total': avg_epoch_loss,
                     'train/epoch_loss_c1': avg_epoch_loss_c1,
                     'train/epoch_loss_c2': avg_epoch_loss_c2,
                     'train/epoch_penalty': avg_epoch_penalty * self.lambda_penalty, # Log weighted penalty
                     'val/epoch_dice_C1': avg_val_dice_c1,
                     'val/epoch_loss_C1': avg_val_loss_c1,
                     'val/epoch_dice_C2': avg_val_dice_c2,
                     'val/epoch_loss_C2': avg_val_loss_c2,
                     'val/epoch_dice_Avg': current_avg_dice,
                     'epoch': epoch,
                     'val/best_avg_dice': self.best_avg_dice,
                     'train/learning_rate_c1': lr_c1, # Log LR again for epoch summary
                     'train/learning_rate_c2': lr_c2
                 }
                 wandb.log(log_dict, step=global_step_end_epoch)

            print(f"---------------------------\n")

        print("--- Joint Training Finished ---")

    # Helper function to calculate loss (handles deep supervision)
    def _calculate_loss(self, outputs, target, criterion):
        # Check if criterion includes Hausdorff component
        has_hausdorff = isinstance(criterion, tuple)
        if has_hausdorff:
            primary_criterion, hausdorff_criterion = criterion
        else:
            primary_criterion = criterion
            hausdorff_criterion = None

        # Check if the primary criterion is AUFL
        is_aufl = isinstance(primary_criterion, AsymmetricUnifiedFocalLoss)

        if self.enable_deep_supervision and isinstance(outputs, (list, tuple)):
            total_loss = 0.0
            total_primary_loss_unweighted = 0.0 # For potential logging
            total_hausdorff_loss_unweighted = 0.0 # For potential logging
            final_output_logits = outputs[0] # Store the original logits for return

            for ds_idx, ds_output_logits in enumerate(outputs):
                # Resize target to match current deep supervision level output
                if ds_output_logits.shape[-3:] != target.shape[-3:]:
                    resized_target = F.interpolate(target.float(), size=ds_output_logits.shape[-3:], mode='nearest')
                else:
                    resized_target = target

                # --- Preprocessing for different loss types ---
                if is_aufl:
                    target_long = resized_target.long()
                    target_one_hot = one_hot(target_long, num_classes=2, dim=1)
                    probs = torch.sigmoid(ds_output_logits)
                    input_for_primary = torch.cat([1 - probs, probs], dim=1)
                else:
                    input_for_primary = ds_output_logits # Most losses take logits
                    target_one_hot = resized_target.float() # and float target

                # Hausdorff loss always takes logits and float target (sigmoid=True internal)
                input_for_hausdorff = ds_output_logits 
                target_for_hausdorff = resized_target.float()
                # --- End Preprocessing ---

                # Calculate primary loss
                loss_level_primary = primary_criterion(input_for_primary, target_one_hot)
                
                # Calculate Hausdorff loss if enabled
                loss_level_hausdorff = 0.0
                if has_hausdorff and hausdorff_criterion is not None:
                    loss_level_hausdorff = hausdorff_criterion(input_for_hausdorff, target_for_hausdorff)
                
                # Combine losses for this deep supervision level
                combined_loss_level = loss_level_primary
                if has_hausdorff:
                    combined_loss_level += self.lambda_hausdorff * loss_level_hausdorff
                
                # Apply deep supervision weight
                weight = self.deep_supervision_weights[ds_idx]
                total_loss += weight * combined_loss_level
                total_primary_loss_unweighted += weight * loss_level_primary.item() # Track component
                total_hausdorff_loss_unweighted += weight * loss_level_hausdorff.item() if isinstance(loss_level_hausdorff, torch.Tensor) else loss_level_hausdorff # Track component

            # Return the total weighted loss and the *original* highest-resolution logits
            return total_loss, final_output_logits
        else:
            # Handle case without deep supervision
            final_output_logits = outputs # Assume single tensor output if not list/tuple

            # Resize target if necessary
            if final_output_logits.shape[-3:] != target.shape[-3:]:
                 resized_target = F.interpolate(target.float(), size=final_output_logits.shape[-3:], mode='nearest')
            else:
                 resized_target = target

            # --- Preprocessing (No deep supervision) ---
            if is_aufl:
                target_long = resized_target.long()
                target_one_hot = one_hot(target_long, num_classes=2, dim=1)
                probs = torch.sigmoid(final_output_logits)
                input_for_primary = torch.cat([1 - probs, probs], dim=1)
            else:
                input_for_primary = final_output_logits
                target_one_hot = resized_target.float()
            
            input_for_hausdorff = final_output_logits 
            target_for_hausdorff = resized_target.float()
            # --- End Preprocessing ---

            # Calculate primary loss
            loss_primary = primary_criterion(input_for_primary, target_one_hot)
            
            # Calculate Hausdorff loss if enabled
            loss_hausdorff = 0.0
            if has_hausdorff and hausdorff_criterion is not None:
                loss_hausdorff = hausdorff_criterion(input_for_hausdorff, target_for_hausdorff)
                
            # Combine losses
            total_loss = loss_primary
            if has_hausdorff:
                total_loss += self.lambda_hausdorff * loss_hausdorff
            
            # Return the loss and the *original* logits
            return total_loss, final_output_logits

    # Helper function to get gradient norm
    def _get_gradient_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def validate(self, epoch, target_class):
        """Validate the specific model corresponding to target_class."""

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ---> Entering validate() for C{target_class}, Epoch {epoch}") # Added print
        model_to_eval = self.model_c1 if target_class == 1 else self.model_c2
        criterion_to_eval = self.criterion_c1 if target_class == 1 else self.criterion_c2
        model_to_eval.eval() # Set the correct model to evaluation mode
        
        total_val_dice = 0.0
        total_val_loss = 0.0
        steps = 0
        
        # Metrics tracking for the evaluated class
        all_dice_target_class = []
        all_prec_target_class = [] 
        all_rec_target_class = []  
        all_loss_dice_comp = [] 
        all_loss_ce_comp = []   

        # Use the test loader for validation
        val_loader = self.test_loader 
        val_loader.restart()
        
        with torch.no_grad():
            # Define a predictor wrapper for the *specific* model being evaluated
            def inference_predictor(x):
                model_output = model_to_eval(x) # Use model_to_eval here
                # Return only the main output (index 0) if deep supervision is enabled
                if self.enable_deep_supervision and isinstance(model_output, (list, tuple)):
                    return model_output[0]
                else:
                    return model_output

            for step, data in enumerate(val_loader):
                if step >= self.num_val_iterations_per_epoch:
                    break

                # Handle potentially empty batches from dataloader
                if not data or 'data' not in data or data['data'].shape[0] == 0:
                     print(f"Warning: Skipping empty validation batch at step {step+1}")
                     continue

                # Get data (original labels)
                inputs, labels_original = data['data'], data['seg'] 
                roi = data.get('roi', None)

                # --- Modify Target for Binary Validation --- 
                if target_class == 1:
                    labels = (labels_original == 1).float()
                elif target_class == 2:
                    labels = (labels_original == 2).float()
                else:
                    raise ValueError(f"Invalid target_class during validation: {target_class}")
                
                # --- ROI Cropping/Padding --- 
                # Apply ROI cropping and ensure data is tensor on correct device
                if self.use_roi and roi is not None:
                     roi_np = roi 
                     inputs_np = inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs
                     labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

                     inputs_np, labels_np = calculate_roi(
                         inputs_np, labels_np, roi_np, 
                         patch_size=self.default_patch_size, 
                         val=True
                     )

                     inputs = torch.from_numpy(inputs_np).to(self.device).float()
                     labels = torch.from_numpy(labels_np).to(self.device).float() 
                else:
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.from_numpy(inputs).to(self.device).float()
                    else:
                        inputs = inputs.to(self.device) 

                    if not isinstance(labels, torch.Tensor):
                        labels = torch.from_numpy(labels).to(self.device).float() 
                    else:
                        labels = labels.to(self.device) 

                # --- Sliding Window Inference --- 
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ----> Starting sliding_window_inference for C{target_class}, Step {step}, Input shape: {inputs.shape}") # Added print
                outputs = sliding_window_inference(
                     inputs=inputs,
                     roi_size=self.default_patch_size, 
                     sw_batch_size=self.batch_size, 
                     predictor=inference_predictor, # Uses the wrapper for model_to_eval
                     overlap=0.5, 
                     mode="gaussian", 
                     padding_mode="constant",
                     device=self.device,
                     progress=False
                 ) 
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] <---- Finished sliding_window_inference for C{target_class}, Step {step}, Output shape: {outputs.shape}") # Added print

                # --- Ensure label shape matches output shape ---
                if outputs.shape[-3:] != labels.shape[-3:]:
                     resized_labels = F.interpolate(labels.float(), size=outputs.shape[-3:], mode='nearest')
                else:
                     resized_labels = labels

                # --- Calculate Loss ---
                # Check if the criterion is AUFL for preprocessing
                is_aufl_val = isinstance(criterion_to_eval, AsymmetricUnifiedFocalLoss)

                # --- AUFL Preprocessing (Validation) ---
                if is_aufl_val:
                    target_long_val = resized_labels.long()
                    target_one_hot_val = one_hot(target_long_val, num_classes=2, dim=1)
                    probs_val = torch.sigmoid(outputs) # 'outputs' are logits from sliding window
                    input_for_loss_val = torch.cat([1 - probs_val, probs_val], dim=1)
                    loss = criterion_to_eval(input_for_loss_val, target_one_hot_val)
                else:
                    # No preprocessing needed for other losses
                    input_for_loss_val = outputs
                    target_for_loss_val = resized_labels.float() # Keep target as float for other losses
                    loss = criterion_to_eval(input_for_loss_val, target_for_loss_val) # Use correct criterion
                # --- End AUFL Preprocessing (Validation) ---

                total_val_loss += loss.item()

                # --- Log Loss Components ---
                if isinstance(criterion_to_eval, (DiceCELoss, TverskyCELoss)):
                    try:
                        if isinstance(criterion_to_eval, TverskyCELoss):
                            # Access internal components if available (might need adjustment based on loss impl.)
                            dice_comp_unweighted = getattr(criterion_to_eval, 'component_tversky_loss', np.nan)
                            ce_comp_unweighted = getattr(criterion_to_eval, 'component_ce_loss', np.nan)
                            lambda_d = getattr(criterion_to_eval, 'lambda_tversky', 1.0)
                            lambda_c = getattr(criterion_to_eval, 'lambda_ce', 1.0)
                        elif isinstance(criterion_to_eval, DiceCELoss): # Correct Indentation
                            output_sig = torch.sigmoid(outputs)
                            target_bin = resized_labels # Already binary float
                            lambda_d = getattr(criterion_to_eval, 'lambda_dice', 1.0)
                            lambda_c = getattr(criterion_to_eval, 'lambda_ce', 1.0)
                            # Recalculate components as they aren't stored directly
                            # Use sigmoid=False for DiceLoss as input is already sigmoid probability
                            dice_comp_unweighted = DiceLoss(sigmoid=False, squared_pred=True)(output_sig, target_bin).item()
                            # Use BCEWithLogitsLoss for CE as input is logits
                            ce_comp_unweighted = BCEWithLogitsLoss()(outputs, target_bin).item()
                        else: # Add missing else
                            dice_comp_unweighted, ce_comp_unweighted, lambda_d, lambda_c = np.nan, np.nan, 1.0, 1.0

                        all_loss_dice_comp.append(dice_comp_unweighted * lambda_d)
                        all_loss_ce_comp.append(ce_comp_unweighted * lambda_c)
                    except Exception as loss_comp_e:
                        print(f"Warning: Failed to log loss components for C{target_class}: {loss_comp_e}")
                        all_loss_dice_comp.append(np.nan)
                        all_loss_ce_comp.append(np.nan)
                
                # --- Calculate Metrics (Dice, Precision, Recall) --- 
                with torch.no_grad():
                    avg_batch_dice = calculate_dice(outputs, resized_labels)
                    total_val_dice += avg_batch_dice
                    all_dice_target_class.append(avg_batch_dice)
                    
                    output_prob = torch.sigmoid(outputs)
                    pred_binary = (output_prob > 0.5).float()
                    target_binary = resized_labels

                    eps = 1e-6
                    tp = torch.sum(pred_binary * target_binary, dim=[1, 2, 3, 4]).float()
                    fp = torch.sum(pred_binary * (1 - target_binary), dim=[1, 2, 3, 4]).float()
                    fn = torch.sum((1 - pred_binary) * target_binary, dim=[1, 2, 3, 4]).float()
                    
                    prec = (tp + eps) / (tp + fp + eps)
                    rec = (tp + eps) / (tp + fn + eps)
                    
                    all_prec_target_class.extend(prec.tolist())
                    all_rec_target_class.extend(rec.tolist())

                steps += 1
                
                if self.verbose and (step % 10 == 0 or step == 1):
                    print(f"Validation step {step}/{self.num_val_iterations_per_epoch} (Class {target_class}): Batch Dice: {avg_batch_dice:.4f}, Batch Loss: {loss.item():.4f}") 

        # --- Calculate final average metrics --- 
        avg_val_dice = np.mean(all_dice_target_class) if all_dice_target_class else 0.0
        avg_val_loss = total_val_loss / steps if steps > 0 else 0.0
        avg_prec = np.mean(all_prec_target_class) if all_prec_target_class else 0.0 
        avg_rec = np.mean(all_rec_target_class) if all_rec_target_class else 0.0   
        avg_loss_dice_comp = np.nanmean(all_loss_dice_comp) if all_loss_dice_comp else 0.0 
        avg_loss_ce_comp = np.nanmean(all_loss_ce_comp) if all_loss_ce_comp else 0.0     
        
        # --- Logging Summary --- 
        print(f"\n--- Validation Summary (Class {target_class}) --- Benchmark") 
        print(f"Average Dice: {avg_val_dice:.4f}")
        print(f"Average Loss: {avg_val_loss:.4f}")
        print(f"Average Prec/Rec: {avg_prec:.4f} / {avg_rec:.4f}") 
        if not np.isnan(avg_loss_dice_comp) and not np.isnan(avg_loss_ce_comp) and (avg_loss_dice_comp > 0 or avg_loss_ce_comp > 0):
            print(f"Average Loss Components (Dice/Tversky / CE): {avg_loss_dice_comp:.4f} / {avg_loss_ce_comp:.4f}")
        print("--------------------------")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] <--- Exiting validate() for C{target_class}, Epoch {epoch}") # Added print

        # Note: WandB logging for validation happens in train_model after both validate calls

        model_to_eval.train() # Set model back to training mode
        return avg_val_dice, avg_val_loss 

    # --- Removed Methods Start ---
    # def analyze_weighting_effectiveness(...)
    # def sample_patch_foreground_based(...)
    # def sample_foreground_coordinate(...)
    # def sample_random_coordinate(...)
    # --- Removed Methods End ---

    def run_training(self):
        """Sets up directories, loads checkpoints, initializes wandb, and runs the main training loop."""

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Entering run_training...")
        # --- Setup Output Directory & Logging ---
        parent_dir = Path(self.dataset_dir).parent
        dataset_name = parent_dir.name if parent_dir and parent_dir.name else Path(self.dataset_dir).name
        output_base_name = f"{dataset_name}_Joint"
        self.fold_dir, log_file, checkpoint_exists_legacy = setup_output_directory(
            self.output_dir, self.fold, output_base_name, self.continue_tr
        )
        run_identifier = f"{output_base_name}_Fold{self.fold}_{self.loss_type_c1}_{self.loss_type_c2}_Pen{self.lambda_penalty}"

        # Redirect stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        if log_file != sys.stdout:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Redirecting stdout/stderr to log file: {self.fold_dir / log_file.name}")
            sys.stdout = log_file
            sys.stderr = log_file
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Logging to stdout, file could not be opened.")

        # Initialize epoch and best dice score (will be overwritten if resuming)
        self.current_epoch = 0 
        self.best_avg_dice = 0.0

        # --- Load Checkpoint if Continuing --- 
        load_successful_c1 = False
        load_successful_c2 = False
        if self.continue_tr:
            print(f"Attempting to resume training from: {self.fold_dir}")

            # Determine checkpoint paths (only latest)
            latest_path_c1 = self.fold_dir / 'checkpoint_C1.pt'
            latest_path_c2 = self.fold_dir / 'checkpoint_C2.pt'

            path_to_load_c1 = None
            path_to_load_c2 = None

            if latest_path_c1.exists() and latest_path_c2.exists():
                print("  Attempting to load latest checkpoints (.pt).")
                path_to_load_c1 = latest_path_c1
                path_to_load_c2 = latest_path_c2
            else:
                print("  Could not find latest checkpoint files (.pt). Cannot resume.")

            # Attempt loading if paths were found
            if path_to_load_c1 and path_to_load_c2:
                load_successful_c1 = self._load_checkpoint(path_to_load_c1, self.model_c1, self.optimizer_c1, self.lr_scheduler_c1)
                load_successful_c2 = self._load_checkpoint(path_to_load_c2, self.model_c2, self.optimizer_c2, self.lr_scheduler_c2)
            
            # If loading failed for *either* checkpoint, reset state to start from scratch
            if not load_successful_c1 or not load_successful_c2:
                print("Checkpoint loading failed for one or both models. Starting training from scratch (Epoch 0).")
                self.current_epoch = 0
                self.best_avg_dice = 0.0
                # Reset optimizers/schedulers? Might be needed if partially loaded.
                # Re-initialize optimizers and schedulers to clear any partial state
                print("  Re-initializing optimizers and schedulers.")
                # Re-initialize based on the configured optimizer type
                if self.optimizer_type == 'adamw':
                    self.optimizer_c1 = AdamW(self.model_c1.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, betas=(0.9, 0.999), eps=1e-8)
                    self.optimizer_c2 = AdamW(self.model_c2.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, betas=(0.9, 0.999), eps=1e-8)
                elif self.optimizer_type == 'sgd':
                    self.optimizer_c1 = SGD(self.model_c1.parameters(), lr=self.initial_lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
                    self.optimizer_c2 = SGD(self.model_c2.parameters(), lr=self.initial_lr, momentum=self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
                self.lr_scheduler_c1 = CosineAnnealingWarmRestarts(self.optimizer_c1, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)
                self.lr_scheduler_c2 = CosineAnnealingWarmRestarts(self.optimizer_c2, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.min_lr)
        else:
            print("Starting training from scratch.")
            # current_epoch and best_avg_dice already initialized to 0

        # --- Initialize WandB (now that fold_dir is known) --- 
        initialize_wandb(self, run_identifier, self.continue_tr) # Pass continue_tr flag

        # --- Setup DataLoaders --- 
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
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Training Loop (from train_model call)...")
        try:
            self.train_model()
        finally:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Exiting training loop (finished or error).") # Added print
            # Ensure log file is closed and stdout restored
            if log_file != sys.stdout:
                 sys.stdout = original_stdout # Restore original stdout
                 sys.stderr = original_stderr # Restore original stderr
                 log_file.close()
            if wandb and wandb.run:
                 wandb.finish()

    # Helper function to load a single model/optimizer checkpoint
    def _load_checkpoint(self, path, model, optimizer, scheduler):
        if not Path(path).exists():
            print(f"Checkpoint file not found at {path}. Skipping load.")
            return False # Indicate loading failed
        try:
            # Explicitly set weights_only=False to load optimizer/scheduler states and epoch number
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda(self.device), weights_only=False) 

            # Try loading model state dict - handle both current and legacy keys
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                print(f"  INFO: Loading model using legacy key 'model_state_dict' from {os.path.basename(path)}")
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 raise KeyError("Checkpoint does not contain 'state_dict' or 'model_state_dict' for the model.")

            # Load optimizer and scheduler (assuming keys are consistent across versions)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Restore epoch and best metrics directly here
            # Resume from the epoch AFTER the one saved
            loaded_epoch = checkpoint.get('epoch', -1)
            self.current_epoch = loaded_epoch + 1 
            # Restore best dice score (use the value saved in the checkpoint)
            # Use existing self.best_avg_dice as default if key missing in older checkpoints
            self.best_avg_dice = checkpoint.get('best_avg_dice', self.best_avg_dice) 

            print(f"Successfully loaded checkpoint from {os.path.basename(path)}. Resuming from epoch {self.current_epoch}.")
            print(f"  Restored best average validation dice: {self.best_avg_dice:.4f}")
            return True # Indicate loading succeeded

        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}. Checkpoint might be incompatible or corrupted. Cannot resume.")
            return False # Indicate loading failed

    
def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser(description="Train Joint SwinUNETR for 4DCT Artifact Correction")
    
    # --- Essential Arguments ---
    parser.add_argument('dataset_dir', type=str, 
                        help="Root directory containing the 'train', 'validate', 'test' subfolders with data files.")
    parser.add_argument('fold', type=str, 
                        help="Fold number (0-4) for 5-fold cross-validation.")
    parser.add_argument('--output_dir', type=str, default='./results_joint', required=False, # Updated default
                        help="Directory to save checkpoints, logs, and validation files.")
    parser.add_argument('--c', action='store_true', required=False, 
                        help="Continue training from the latest checkpoint in the fold's output directory.")
    parser.add_argument('--pretrained_weights_c1', type=str, default=None, required=False, 
                        help="Path to pretrained weights for Class 1 model (.pt file).")
    parser.add_argument('--pretrained_weights_c2', type=str, default=None, required=False, 
                        help="Path to pretrained weights for Class 2 model (.pt file).")
    parser.add_argument('--device', type=int, required=False, default=0, 
                        help="GPU device ID to train on.")    
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size.') 
    parser.add_argument('--initial_lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for scheduler.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of linear warmup epochs.')
    parser.add_argument('--scheduler_T0', type=int, default=30, help='T_0 for CosineAnnealingWarmRestarts scheduler.')
    parser.add_argument('--scheduler_T_mult', type=int, default=2, help='T_mult for CosineAnnealingWarmRestarts scheduler.')
    
    # --- Optimizer Selection ---
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='Optimizer to use (adamw or sgd).')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum for SGD optimizer (if used).')
    parser.add_argument('--nesterov', action=argparse.BooleanOptionalAction, default=True, help='Use Nesterov momentum for SGD optimizer (if used).')

    # --- Model & Data Parameters ---
    parser.add_argument('--img_size', type=int, nargs=3, default=[64, 160, 256], help='Input image size (depth, height, width) for the model.')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 160, 256], help='Patch size for training.')
    parser.add_argument('--feature_size', type=int, default=24, help='Feature size for SwinUNETR.')
    parser.add_argument('--no_deep_supervision', action='store_false', dest='enable_deep_supervision', 
                        help='Disable deep supervision.')
    parser.add_argument('--use_roi', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable ROI cropping/padding during loading.')

    # --- Loss Parameters (Updated for Joint Binary) ---
    parser.add_argument('--loss_type_c1', type=str, default='tverskyce', 
                        choices=['dice', 'tversky', 'focal', 'dicefocal', 'dicece', 'tverskyce', 'asym_unified_focal', 'tverskyfocal'], 
                        help='Loss function for Class 1 model.')
    parser.add_argument('--loss_type_c2', type=str, default='dicece', 
                        choices=['dice', 'tversky', 'focal', 'dicefocal', 'dicece', 'tverskyce', 'asym_unified_focal', 'tverskyfocal'], 
                        help='Loss function for Class 2 model.')
    parser.add_argument('--focal_gamma_c1', type=float, default=2, help='Gamma for Focal-based losses (C1, used if selected).')
    parser.add_argument('--focal_gamma_c2', type=float, default=2, help='Gamma for Focal-based losses (C2, used if selected).')
    parser.add_argument('--tversky_alpha_c1', type=float, default=0.60, help='Alpha for Tversky/TverskyCE/TverskyFocal loss (weights FP, used if selected).')
    parser.add_argument('--tversky_beta_c1', type=float, default=0.40, help='Beta for Tversky/TverskyCE/TverskyFocal loss (weights FN, used if selected).')
    parser.add_argument('--tversky_alpha_c2', type=float, default=0.70, help='Alpha for Tversky/TverskyCE/TverskyFocal loss (weights FP, used if selected).')
    parser.add_argument('--tversky_beta_c2', type=float, default=0.30, help='Beta for Tversky/TverskyCE/TverskyFocal loss (weights FN, used if selected).')
    parser.add_argument('--lambda_dice', type=float, default=1.0, help='Shared weight for Dice/Tversky component in combined losses.')
    parser.add_argument('--lambda_focal_c1', type=float, default=1.7, help='Weight for CE/Focal component in combined losses (C1).')
    parser.add_argument('--lambda_focal_c2', type=float, default=1.7, help='Weight for CE/Focal component in combined losses (C2).')
    parser.add_argument('--lambda_penalty', type=float, default=0.1, help='Weight for the mutual exclusivity penalty term.')
    parser.add_argument('--lambda_hausdorff', type=float, default=0.0, help='Weight for the Hausdorff Distance Transform Loss component (0 to disable).')
    parser.add_argument('--aufl_delta', type=float, default=0.6, help='Delta (background weight) for AsymmetricUnifiedFocalLoss.')
    parser.add_argument('--aufl_weight', type=float, default=0.5, help='Weight between AsymmetricFocalLoss and AsymmetricFocalTverskyLoss components in AsymmetricUnifiedFocalLoss.')

    # --- Performance & Logging ---\
    parser.add_argument('--train_steps_per_epoch', type=int, default=500, help='Number of training steps per epoch.')
    parser.add_argument('--val_steps_per_epoch', type=int, default=250, help='Number of validation steps per epoch.')
    parser.add_argument('--num_workers_train', type=int, default=12, help='Number of workers for training data augmentation.')
    parser.add_argument('--num_workers_val', type=int, default=4, help='Number of workers for validation data augmentation.')
    parser.add_argument('--quiet', action='store_true', required=False,
                        help="Reduce verbosity and disable wandb logging.")
    parser.add_argument('--foreground_prob', type=float, default=0.9, help='Probability to sample foreground patches')

    args = parser.parse_args()

    # --- Validate arguments ---
    if not (args.fold.isdigit() and 0 <= int(args.fold) < 5):
         raise ValueError(f"Fold must be an integer between 0 and 4, got {args.fold}")
    # Target class is validated by choices=[1, 2]
    args.img_size = tuple(args.img_size)
    args.patch_size = tuple(args.patch_size)
    if not Path(args.dataset_dir).is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {args.dataset_dir}")
    # Add more validation as needed

    # --- Resume Logic ---
    resume_training = args.c is not None
    if resume_training:
        print(f"Resuming training using weights from directory: {args.c}")
        # The actual loading happens inside SWINUNETRTrainer based on self.resume and self.output_dir
        # This warning is likely outdated now
        # print("Warning: Checkpoint loading (--c) for joint training not fully implemented yet.", file=sys.stderr)
        # Set the output directory to the resume directory -- THIS WAS THE BUG
        # args.output_dir = args.c # Don't overwrite output_dir with True!
    else:
        print("Starting training from scratch.")

    # --- Device Setup ---
    # The trainer expects a single integer device ID
    # device_list = [int(d) for d in args.device.split(',')] # This was incorrect as args.device is int

    # --- Create Trainer ---
    trainer = SWINUNETRTrainer(
        weights_c1=args.pretrained_weights_c1, # Pass C1 weights
        weights_c2=args.pretrained_weights_c2, # Pass C2 weights
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
        loss_type_c1=args.loss_type_c1, # Pass C1 loss type
        loss_type_c2=args.loss_type_c2, # Pass C2 loss type
        focal_gamma_c1=args.focal_gamma_c1, # Pass C1 gamma
        focal_gamma_c2=args.focal_gamma_c2, # Pass C2 gamma
        tversky_alpha_c1=args.tversky_alpha_c1,
        tversky_beta_c1=args.tversky_beta_c1,
        tversky_alpha_c2=args.tversky_alpha_c2,
        tversky_beta_c2=args.tversky_beta_c2,
        foreground_prob=args.foreground_prob,
        lambda_dice=args.lambda_dice, # Pass shared dice lambda
        lambda_focal_c1=args.lambda_focal_c1, # Pass C1 focal lambda
        lambda_focal_c2=args.lambda_focal_c2, # Pass C2 focal lambda
        lambda_penalty=args.lambda_penalty,
        lambda_hausdorff=args.lambda_hausdorff, # Pass Hausdorff lambda
        aufl_delta=args.aufl_delta,
        aufl_weight=args.aufl_weight,
        verbose=not args.quiet,
        optimizer_type=args.optimizer, # Pass optimizer type
        momentum=args.momentum,       # Pass SGD momentum
        nesterov=args.nesterov       # Pass SGD Nesterov flag
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

