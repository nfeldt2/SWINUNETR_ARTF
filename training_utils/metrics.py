import torch
import numpy as np
# Potentially add imports for other MONAI metrics if needed later
# from monai.metrics import DiceMetric, compute_meandice

def calculate_dice(output, target):
    """Calculate Dice coefficient between prediction and target for binary segmentation.
    Assumes output is (B, C=1, D, H, W) logits.
    Assumes target is (B, 1, D, H, W) binary labels (0 or 1).
    
    Args:
        output (torch.Tensor): Model output tensor (logits).
        target (torch.Tensor): Ground truth tensor (binary 0/1).
        # per_class argument removed
    
    Returns:
        float: Average overall dice across the batch.
    """
    batch_size = target.shape[0]
    
    # Apply sigmoid to get probabilities 
    output_prob = torch.sigmoid(output) # Shape (B, 1, D, H, W)
    
    # Threshold probabilities to get binary predictions
    pred_binary = (output_prob > 0.5).float() # Predictions for the target class

    # Ensure target is float
    target_binary = target.float()

    # Calculate overall foreground dice
    intersection_fg = torch.sum(pred_binary * target_binary, dim=[1, 2, 3, 4]).float()
    union_fg = torch.sum(pred_binary, dim=[1, 2, 3, 4]) + torch.sum(target_binary, dim=[1, 2, 3, 4])
    
    # Calculate dice per batch item
    dice_per_sample = (2.0 * intersection_fg + 1e-6) / (union_fg + 1e-6) # Shape (B,)
    
    # Return average overall dice across the batch
    return torch.mean(dice_per_sample).item()

# Removed old multi-class logic 