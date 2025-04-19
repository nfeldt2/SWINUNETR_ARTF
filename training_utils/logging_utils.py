import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable # For colorbar
import torch
import time
import warnings # Added for warnings

try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. Logging functions may fail.")
    wandb = None

def initialize_wandb(trainer_self, run_identifier: str, continue_tr: bool):
    """Initializes Weights & Biases logging.
    
    Args:
        trainer_self: The trainer instance (to access config attributes like fold_dir, verbose etc.).
        run_identifier: A unique identifier for the run (e.g., Dataset_C#_Fold#_LossType).
        continue_tr: Boolean flag indicating if training is being resumed.
    """
    if not trainer_self.verbose: # Check verbose flag from trainer
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Verbose is False, WandB initialization skipped.")
        trainer_self.wandb_initialized = False
        return
        
    if wandb is None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WandB not installed, logging disabled.")
        trainer_self.wandb_initialized = False
        return

    if trainer_self.wandb_initialized: # Avoid re-initializing
         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WandB already initialized.")
         return

    try:
        # Extract config from trainer instance
        config = {
            "batch_size": getattr(trainer_self, 'batch_size', None),
            "patch_size": getattr(trainer_self, 'default_patch_size', None),
            "img_size": getattr(trainer_self, 'img_size', None),
            "feature_size": getattr(trainer_self, 'feature_size', None),
            "max_epochs": getattr(trainer_self, 'num_epochs', None),
            "initial_lr": getattr(trainer_self, 'initial_lr', None),
            "min_lr": getattr(trainer_self, 'min_lr', None),
            "weight_decay": getattr(trainer_self, 'weight_decay', None),
            "warmup_epochs": getattr(trainer_self, 'warmup_epochs', None),
            "scheduler_T0": getattr(trainer_self, 'T_0', None),
            "scheduler_T_mult": getattr(trainer_self, 'T_mult', None),
            "fold": getattr(trainer_self, 'fold', None),
            "model": "SwinUNETR-artf-binary", 
            "dataset": run_identifier.split('_Fold')[0], # Extract base name before Fold
            "target_class": getattr(trainer_self, 'target_class', None),
            "deep_supervision": getattr(trainer_self, 'enable_deep_supervision', None),
            "use_roi": getattr(trainer_self, 'use_roi', None),
            "loss_type": getattr(trainer_self, 'loss_type', None),
            "foreground_prob": getattr(trainer_self, 'foreground_prob', None),
            "lambda_dice": getattr(trainer_self, 'lambda_dice', None),
            "lambda_ce": getattr(trainer_self, 'lambda_ce', None),
            # Add other relevant hyperparameters if needed
        }
        # Use the provided identifier for name and id
        run_name = run_identifier
        # Determine resume behavior based on continue_tr
        resume_status = "allow" if continue_tr else None
        run_id = run_identifier if continue_tr else None # Only provide ID if resuming
        
        if wandb.run is None: # Check if a run is already active
             # Ensure fold_dir exists on the trainer object
             if not hasattr(trainer_self, 'fold_dir') or trainer_self.fold_dir is None:
                 print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: fold_dir not set on trainer. Cannot initialize WandB.")
                 trainer_self.wandb_initialized = False
                 return
                 
             wandb.init(project="swinunetr-artifact-correction", 
                        config=config, 
                        dir=str(trainer_self.fold_dir), # Log wandb runs within the fold directory
                        name=run_name, # Name the run based on the fold
                        resume=resume_status, # Use determined resume status
                        id=run_id) # Provide ID only if resuming
             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WandB initialized successfully for run '{run_name}' (Resuming: {continue_tr})") # Added resume status to log
             trainer_self.wandb_initialized = True # Set flag on trainer
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WandB run {wandb.run.id} already active.")
            trainer_self.wandb_initialized = True # Assume it's initialized if active
            
    except ImportError:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WandB not installed, disabling logging.")
        trainer_self.wandb_initialized = False
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error initializing wandb: {e}. Disabling logging.")
        trainer_self.wandb_initialized = False

def save_debug_images(fold_dir: Path, epoch: int, step: int, image_tensor: torch.Tensor, 
                        output_c1: torch.Tensor, target_c1: torch.Tensor, 
                        output_c2: torch.Tensor, target_c2: torch.Tensor, 
                        sample_dice_c1: float, sample_dice_c2: float,
                        filename_sample0: str):
    """Saves slices for debugging joint binary segmentation models.
    
    Args:
        fold_dir: Path to the current fold's output directory.
        epoch: Current epoch number.
        step: Current training step number.
        image_tensor: Batch of input image tensors (B, C, D, H, W).
        output_c1: Batch of model C1 output tensors (logits) (B, 1, D, H, W).
        target_c1: Batch of binary target label tensors for C1 (B, 1, D, H, W).
        output_c2: Batch of model C2 output tensors (logits) (B, 1, D, H, W).
        target_c2: Batch of binary target label tensors for C2 (B, 1, D, H, W).
        sample_dice_c1: Dice score C1 for the first sample in the batch.
        sample_dice_c2: Dice score C2 for the first sample in the batch.
        filename_sample0: Filename/identifier for the first sample.
        # Removed target_class
    """
    image_dir = None
    fig_axial, fig_coronal = None, None 
    try:
        image_dir = fold_dir / "debug_images" / f"epoch_{epoch}"
        image_dir.mkdir(parents=True, exist_ok=True)

        # --- Data Preparation (First Sample) ---
        try:
            input_img_np = image_tensor[0].squeeze().detach().cpu().numpy()
            target_c1_np = target_c1[0].squeeze().detach().cpu().numpy().astype(int) 
            target_c2_np = target_c2[0].squeeze().detach().cpu().numpy().astype(int)
            prob_c1_np = torch.sigmoid(output_c1[0]).squeeze().detach().cpu().numpy() # Shape (D, H, W)
            prob_c2_np = torch.sigmoid(output_c2[0]).squeeze().detach().cpu().numpy() # Shape (D, H, W)
            
            if np.isnan(prob_c1_np).any() or np.isinf(prob_c1_np).any():
                print(f"--- WARNING: NaN or Inf detected in C1 probabilities! ---")
                prob_c1_np = np.nan_to_num(prob_c1_np)
            if np.isnan(prob_c2_np).any() or np.isinf(prob_c2_np).any():
                 print(f"--- WARNING: NaN or Inf detected in C2 probabilities! ---")
                 prob_c2_np = np.nan_to_num(prob_c2_np)
        except Exception as data_prep_e:
            print(f"--- ERROR during data preparation: {data_prep_e} ---")
            return

        # --- Slice Selection (Based on any foreground target) --- 
        try:
            # Combine targets to find slices with any artifact
            any_target_np = (target_c1_np > 0) | (target_c2_np > 0)
            has_foreground = np.any(any_target_np)

            if not has_foreground:
                mid_slice_d = input_img_np.shape[0] // 2
                mid_slice_h = input_img_np.shape[1] // 2
                target_label_str = "BG Only"
            else:
                target_label_str = ""
                if np.any(target_c1_np > 0): target_label_str += "C1 "
                if np.any(target_c2_np > 0): target_label_str += "C2"
                
                # Find slice with max pixels for *any* target class 
                sum_pixels_d = np.sum(any_target_np, axis=(1, 2)) 
                sum_pixels_h = np.sum(any_target_np, axis=(0, 2)) 

                if np.max(sum_pixels_d) > 0:
                    mid_slice_d = np.argmax(sum_pixels_d)
                else: 
                     artifact_indices_d = np.where(np.any(any_target_np, axis=(1, 2)))[0]
                     mid_slice_d = artifact_indices_d[len(artifact_indices_d) // 2] if len(artifact_indices_d) > 0 else input_img_np.shape[0] // 2

                if np.max(sum_pixels_h) > 0:
                    mid_slice_h = np.argmax(sum_pixels_h)
                else: 
                     artifact_indices_h = np.where(np.any(any_target_np, axis=(0, 2)))[0]
                     mid_slice_h = artifact_indices_h[len(artifact_indices_h) // 2] if len(artifact_indices_h) > 0 else input_img_np.shape[1] // 2

        except Exception as slice_sel_e:
             print(f"--- ERROR during slice selection: {slice_sel_e} ---")
             mid_slice_d = input_img_np.shape[0] // 2
             mid_slice_h = input_img_np.shape[1] // 2
             target_label_str = "ErrorInSliceSelection"
             
        # --- Format Metrics for Titles ---
        metrics_str = f"Dice C1: {sample_dice_c1:.3f} | Dice C2: {sample_dice_c2:.3f}"

        # --- Prepare Slice Data --- 
        try:
            slice_axial = input_img_np[mid_slice_d, :, :]
            target1_axial = target_c1_np[mid_slice_d, :, :]
            target2_axial = target_c2_np[mid_slice_d, :, :]
            prob1_axial = prob_c1_np[mid_slice_d, :, :]
            prob2_axial = prob_c2_np[mid_slice_d, :, :]

            slice_coronal = input_img_np[:, mid_slice_h, :]
            target1_coronal = target_c1_np[:, mid_slice_h, :]
            target2_coronal = target_c2_np[:, mid_slice_h, :]
            prob1_coronal = prob_c1_np[:, mid_slice_h, :]
            prob2_coronal = prob_c2_np[:, mid_slice_h, :]
        except Exception as slice_prep_e:
            print(f"--- ERROR preparing slice data: {slice_prep_e} ---")
            return

        # --- Calculate Thresholded Predictions --- 
        try:
            pred_c1_np = (prob_c1_np > 0.5).astype(int)
            pred_c2_np = (prob_c2_np > 0.5).astype(int)

            pred1_axial = pred_c1_np[mid_slice_d, :, :]
            pred2_axial = pred_c2_np[mid_slice_d, :, :]
            pred1_coronal = pred_c1_np[:, mid_slice_h, :]
            pred2_coronal = pred_c2_np[:, mid_slice_h, :]
        except Exception as calc_e:
             print(f"--- ERROR during prediction thresholding: {calc_e} ---")
             return

        # --- Plotting Axial View --- 
        try:
            fig_axial, axes_axial = plt.subplots(1, 7, figsize=(30, 5)) # 1 row, 7 columns
            fig_axial.suptitle(f"Axial Joint - Step: {step}, Epoch: {epoch}, Slice: {mid_slice_d}, Target(s): {target_label_str}\nFile: {filename_sample0} | {metrics_str}", fontsize=11)

            # Panels: Input, Target C1, Target C2, Prob C1, Prob C2, Pred C1, Pred C2
            axes_axial[0].imshow(slice_axial, cmap='gray', origin='lower'); axes_axial[0].set_title('Input'); axes_axial[0].axis('off')
            im_t1a = axes_axial[1].imshow(target1_axial, cmap='Reds', vmin=0, vmax=1, origin='lower'); axes_axial[1].set_title('Target C1'); axes_axial[1].axis('off')
            im_t2a = axes_axial[2].imshow(target2_axial, cmap='Blues', vmin=0, vmax=1, origin='lower'); axes_axial[2].set_title('Target C2'); axes_axial[2].axis('off')
            im_p1a = axes_axial[3].imshow(prob1_axial, cmap='viridis', vmin=0, vmax=1, origin='lower'); axes_axial[3].set_title('Prob C1'); axes_axial[3].axis('off')
            im_p2a = axes_axial[4].imshow(prob2_axial, cmap='viridis', vmin=0, vmax=1, origin='lower'); axes_axial[4].set_title('Prob C2'); axes_axial[4].axis('off')
            im_pred1a = axes_axial[5].imshow(pred1_axial, cmap='Reds', vmin=0, vmax=1, origin='lower'); axes_axial[5].set_title('Pred C1'); axes_axial[5].axis('off')
            im_pred2a = axes_axial[6].imshow(pred2_axial, cmap='Blues', vmin=0, vmax=1, origin='lower'); axes_axial[6].set_title('Pred C2'); axes_axial[6].axis('off')

            # Add colorbars for probabilities
            fig_axial.colorbar(im_p1a, ax=axes_axial[3], shrink=0.8)
            fig_axial.colorbar(im_p2a, ax=axes_axial[4], shrink=0.8)
            # Optional: colorbars for predictions if needed (simple 0/1)
            # fig_axial.colorbar(im_pred1a, ax=axes_axial[5], shrink=0.8)
            # fig_axial.colorbar(im_pred2a, ax=axes_axial[6], shrink=0.8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 

            # Save Axial Figure
            filename_tag = "_highDice" if (sample_dice_c1 > 0.9 or sample_dice_c2 > 0.9) else ""
            save_path_axial = image_dir / f"train_joint_epoch_{epoch}_step_{step}{filename_tag}_axial.png"
            plt.savefig(save_path_axial)
            plt.close(fig_axial)
        except Exception as plot_axial_e:
            print(f"--- ERROR during axial plotting: {plot_axial_e} ---")
            if fig_axial: plt.close(fig_axial)

        # --- Plotting Coronal View --- 
        try:
            fig_coronal, axes_coronal = plt.subplots(1, 7, figsize=(30, 5)) 
            fig_coronal.suptitle(f"Coronal Joint - Step: {step}, Epoch: {epoch}, Slice: {mid_slice_h}, Target(s): {target_label_str}\nFile: {filename_sample0} | {metrics_str}", fontsize=11)

            # Panels: Input, Target C1, Target C2, Prob C1, Prob C2, Pred C1, Pred C2
            axes_coronal[0].imshow(slice_coronal, cmap='gray', aspect='auto', origin='lower'); axes_coronal[0].set_title('Input'); axes_coronal[0].axis('off')
            im_t1c = axes_coronal[1].imshow(target1_coronal, cmap='Reds', vmin=0, vmax=1, aspect='auto', origin='lower'); axes_coronal[1].set_title('Target C1'); axes_coronal[1].axis('off')
            im_t2c = axes_coronal[2].imshow(target2_coronal, cmap='Blues', vmin=0, vmax=1, aspect='auto', origin='lower'); axes_coronal[2].set_title('Target C2'); axes_coronal[2].axis('off')
            im_p1c = axes_coronal[3].imshow(prob1_coronal, cmap='viridis', vmin=0, vmax=1, aspect='auto', origin='lower'); axes_coronal[3].set_title('Prob C1'); axes_coronal[3].axis('off')
            im_p2c = axes_coronal[4].imshow(prob2_coronal, cmap='viridis', vmin=0, vmax=1, aspect='auto', origin='lower'); axes_coronal[4].set_title('Prob C2'); axes_coronal[4].axis('off')
            im_pred1c = axes_coronal[5].imshow(pred1_coronal, cmap='Reds', vmin=0, vmax=1, aspect='auto', origin='lower'); axes_coronal[5].set_title('Pred C1'); axes_coronal[5].axis('off')
            im_pred2c = axes_coronal[6].imshow(pred2_coronal, cmap='Blues', vmin=0, vmax=1, aspect='auto', origin='lower'); axes_coronal[6].set_title('Pred C2'); axes_coronal[6].axis('off')

            # Add colorbars for probabilities
            fig_coronal.colorbar(im_p1c, ax=axes_coronal[3], shrink=0.8)
            fig_coronal.colorbar(im_p2c, ax=axes_coronal[4], shrink=0.8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.92])

            # Save Coronal Figure
            save_path_coronal = image_dir / f"train_joint_epoch_{epoch}_step_{step}{filename_tag}_coronal.png"
            plt.savefig(save_path_coronal)
            plt.close(fig_coronal)
        except Exception as plot_coronal_e:
            print(f"--- ERROR during coronal plotting: {plot_coronal_e} ---")
            if fig_coronal: plt.close(fig_coronal)

    except Exception as outer_e:
        print(f"--- UNEXPECTED ERROR in save_debug_images: {outer_e} ---")
        # Ensure figures are closed if created before error
        if fig_axial: plt.close(fig_axial)
        if fig_coronal: plt.close(fig_coronal)

def training_metrics_report(trainer_self, epoch: int, step: int, train_loss_history: list, 
                            num_iterations_per_epoch: int):
    """Generate a simplified report on recent training metrics.
    
    Args:
        trainer_self: The trainer instance (for wandb_initialized, model).
        epoch: Current epoch.
        step: Current step within epoch.
        train_loss_history: List containing recent training losses.
        num_iterations_per_epoch: Total steps per epoch (for wandb global step).
    """
    
    # Calculate stability metrics
    recent_losses = train_loss_history[-min(50, len(train_loss_history)):]
    current_loss = recent_losses[-1] if recent_losses else np.nan
    loss_std = np.std(recent_losses) if len(recent_losses) > 1 else 0
    
    # Get gradient norm (if available on trainer model)
    total_norm = np.nan
    if hasattr(trainer_self, 'model'):
        try:
            total_norm = 0
            for p in trainer_self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        except Exception as e:
             print(f"Warning: Could not calculate gradient norm: {e}")
             total_norm = np.nan

    print("\n--- Training Metrics Report (Simplified) ---")
    print(f"Epoch {epoch}, Step {step}")
    print(f"- Loss (current): {current_loss:.4f}")
    print(f"- Loss std (last 50 steps): {loss_std:.4f}")
    print(f"- Gradient norm: {total_norm:.4f}")
    
    # Simplified Dice tracking is now done per-step in the main loop and logged there
    # No complex history needed here anymore

    # Log simplified metrics to wandb
    if hasattr(trainer_self, 'wandb_initialized') and trainer_self.wandb_initialized and wandb:
        log_dict = {
            f'train/report/loss_std': loss_std,
            f'train/report/gradient_norm': total_norm,
        }
        # Log report metrics against the current global training step
        global_step = epoch * num_iterations_per_epoch + step # Calculate global step
        wandb.log(log_dict, step=global_step)
    
    print("---------------------------------------") 