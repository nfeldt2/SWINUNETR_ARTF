import time

try:
    import wandb
except ImportError:
    wandb = None

def adjust_learning_rate(trainer_self, epoch: int):
    """Adjusts learning rate based on warmup and scheduler.

    Args:
        trainer_self: The trainer instance (to access optimizer, scheduler, config attrs).
        epoch: Current epoch number.
    """
    # Ensure optimizer and scheduler exist
    if not hasattr(trainer_self, 'optimizer') or not hasattr(trainer_self, 'lr_scheduler'):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Optimizer or LR scheduler not found on trainer instance. Cannot adjust LR.")
        return

    current_lr = trainer_self.optimizer.param_groups[0]['lr']
    
    if epoch < trainer_self.warmup_epochs:
        # Linear warmup
        lr = trainer_self.initial_lr * ((epoch + 1) / trainer_self.warmup_epochs)
        for param_group in trainer_self.optimizer.param_groups:
            param_group['lr'] = lr
        new_lr = lr
    else:
        # Step the scheduler after warmup phase
        try:
            trainer_self.lr_scheduler.step(epoch - trainer_self.warmup_epochs) # Pass relative epoch for scheduler
            new_lr = trainer_self.optimizer.param_groups[0]['lr'] # Get LR after scheduler step
        except Exception as e:
             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error stepping LR scheduler: {e}")
             new_lr = current_lr # Keep current LR if scheduler step fails

    # Print message only if LR actually changed and verbose is enabled
    if trainer_self.verbose and new_lr != current_lr:
         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}: LR adjusted from {current_lr:.8f} to {new_lr:.8f}")
    
    # Log LR to wandb if used
    if hasattr(trainer_self, 'wandb_initialized') and trainer_self.wandb_initialized and wandb:
        try:
            wandb.log({"train/learning_rate": new_lr}, step=epoch)
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Failed to log learning rate to WandB: {e}") 