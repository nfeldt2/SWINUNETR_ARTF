import torch
from pathlib import Path
import time # Added for print timestamp

def load_most_recent_checkpoint(trainer_self, fold_dir: Path):
    """Loads the most recent checkpoint if it exists.

    Args:
        trainer_self: The trainer instance (to access/update model, optimizer, etc.).
        fold_dir: Path to the fold-specific output directory.
    """
    # Check for pt file 
    checkpoint_path = fold_dir / 'checkpoint.pt'
    if checkpoint_path.exists():
        # Load checkpoint onto the correct device
        try:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(trainer_self.device))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint found at {checkpoint_path}.")
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR loading checkpoint file {checkpoint_path}: {e}")
            # Initialize variables for starting fresh if checkpoint load fails
            trainer_self.current_epoch = 0
            trainer_self.loss = 1000
            trainer_self.best_dice = 0.0
            return

        # Load model state - handle potential architecture changes carefully
        if hasattr(trainer_self, 'model') and 'model_state_dict' in checkpoint:
            try:
                # Direct load if architecture matches
                trainer_self.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                 print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Error loading model state dict, likely architecture mismatch: {e}")
                 # Attempt partial load (load matching keys)
                 try:
                     model_dict = trainer_self.model.state_dict()
                     pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                        if k in model_dict and v.shape == model_dict[k].shape}
                     model_dict.update(pretrained_dict) 
                     trainer_self.model.load_state_dict(model_dict)
                     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Partially loaded {len(pretrained_dict)} matching keys from checkpoint model state.")
                 except Exception as partial_load_e:
                      print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error during partial model state load: {partial_load_e}")
            except Exception as model_load_e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error loading model state: {model_load_e}")
        else:
             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Model or model state dict not found in checkpoint or trainer.")

        # Load optimizer state
        if hasattr(trainer_self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            try:
                trainer_self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not load optimizer state: {e}. Optimizer will start from scratch.")
        else:
             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Optimizer or optimizer state dict not found in checkpoint or trainer.")

        # Load epoch, loss, best_dice
        trainer_self.current_epoch = checkpoint.get('epoch', 0) + 1 # Start from next epoch
        trainer_self.loss = checkpoint.get('loss', 1000) # Use training loss?
        trainer_self.best_dice = checkpoint.get('best_dice', 0.0) # Use best validation dice
        
        # Load scheduler state if available
        if hasattr(trainer_self, 'lr_scheduler') and 'scheduler_state_dict' in checkpoint:
             try:
                 trainer_self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
             except Exception as e:
                 print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Could not load scheduler state: {e}. Scheduler will start from scratch.")
        else:
             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Info: Scheduler state not found in checkpoint or trainer. Scheduler will start from scratch.")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Continuing training from epoch {trainer_self.current_epoch} with best validation Dice {trainer_self.best_dice:.4f}")
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: No checkpoint found at {checkpoint_path}. Training from scratch.")
        # Ensure variables are initialized correctly for starting fresh
        trainer_self.current_epoch = 0
        trainer_self.loss = 1000
        trainer_self.best_dice = 0.0 