import torch
from torch.nn.modules.loss import _Loss
from monai.losses import DiceLoss, TverskyLoss, FocalLoss, DiceFocalLoss, DiceCELoss
from torch.nn import BCEWithLogitsLoss # Import BCEWithLogitsLoss
import warnings

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
                continue # Skip to next sample

            if len(gt_classes) > 1:
                # Multiple artifact classes found, this shouldn't happen based on assumption
                # Handle this case (e.g., log a warning, average losses, take max?)
                # For now, let's calculate loss for the first artifact found and warn
                # print(f"Warning: Multiple artifact classes {gt_classes.tolist()} found in target sample {b}. Using {gt_classes[0]}.")
                gt_class = gt_classes[0].item() # Use the first one found
            else:
                gt_class = gt_classes[0].item() # Should be 1 or 2


            # Determine the target channel and weight based on the ground truth class
            if gt_class == 1:
                target_channel_idx = 0 # Output channel for class 1
                weight = self.class_weights[0] 
                target_binary = (target[b] == 1).float() # Binary target for class 1
            elif gt_class == 2:
                target_channel_idx = 1 # Output channel for class 2
                weight = self.class_weights[1]
                target_binary = (target[b] == 2).float() # Binary target for class 2
            else:
                # Should not happen if gt_classes logic is correct
                continue 

            # Select the corresponding output channel
            output_channel = output[b, target_channel_idx].unsqueeze(0) # Keep batch dim (1, D, H, W)
            
            # Apply sigmoid if required by the loss function
            if self.sigmoid:
                 output_channel = torch.sigmoid(output_channel)
            
            # Calculate the loss for this sample
            # Target should be (1, 1, D, H, W), Output should be (1, 1, D, H, W) for most MONAI losses
            # Reshape output if necessary (FocalLoss might expect different shape)
            # MONAI losses generally handle (B, C, ...) where C=1 for binary cases.
            sample_loss = self.binary_loss(output_channel, target_binary)
            
            # Apply class weight
            weighted_loss = sample_loss * weight
            
            total_loss += weighted_loss
            samples_with_artifact += 1

        # Average the loss over samples that actually had an artifact
        if samples_with_artifact > 0:
            return total_loss / samples_with_artifact
        else:
            # Return 0 loss if no artifacts were present in the batch
            # Ensure it requires gradients if necessary, though unlikely if loss is 0
            return torch.tensor(0.0, device=output.device, requires_grad=output.requires_grad) 

class TverskyCELoss(_Loss):
    """
    Computes the Tversky loss and Binary Cross Entropy Loss, sums them after weighting.
    Applies sigmoid activation to the input tensor before calculating losses.
    Uses BCEWithLogitsLoss for numerical stability.
    """

    def __init__(
        self,
        to_onehot_y: bool = False, # Keep False for binary
        sigmoid: bool = True,      # Apply sigmoid internally
        squared_pred: bool = True, # Often used with Dice variants
        alpha: float = 0.5,        # Tversky alpha (controls FP penalty)
        beta: float = 0.5,         # Tversky beta (controls FN penalty)
        lambda_tversky: float = 1.0, # Weight for Tversky component
        lambda_ce: float = 1.0,    # Weight for CE component
    ) -> None:
        super().__init__()
        if alpha + beta != 1.0:
            warnings.warn("Tversky alpha + beta should sum to 1.0 for F-beta score interpretation.")
            
        if not sigmoid:
            warnings.warn("TverskyCELoss expects raw logits (sigmoid=True is enforced internally). Passed sigmoid=False is ignored.")

        self.lambda_tversky = lambda_tversky
        self.lambda_ce = lambda_ce
        
        # Instantiate TverskyLoss - Note: it applies sigmoid internally if sigmoid=True
        self.tversky = TverskyLoss(
            to_onehot_y=to_onehot_y, 
            sigmoid=True, # TverskyLoss handles sigmoid
            alpha=alpha, 
            beta=beta
        )
        
        # Instantiate BCEWithLogitsLoss (handles logits directly)
        self.cross_entropy = BCEWithLogitsLoss()

        # Placeholders to store component losses for potential logging
        self.component_tversky_loss = 0.0
        self.component_ce_loss = 0.0

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. Raw logits required.
            target: the shape should be BNH[WD]. Binary target (0 or 1).

        Raises:
            ValueError: When input and target (after one hot) have different shapes.
            ValueError: When target has values other than 0 or 1.

        """
        if target.shape != input.shape:
             raise ValueError(f"Target shape ({target.shape}) must match input shape ({input.shape})")
        # Target should be binary (0 or 1) and float for BCE
        target_float = target.float()
        if not torch.all((target_float == 0) | (target_float == 1)):
             warnings.warn("Target tensor contains values other than 0 or 1.")

        # Calculate Tversky Loss (expects logits if sigmoid=True internally)
        tversky_loss = self.tversky(input, target_float)

        # Calculate CE Loss (expects raw logits)
        # Ensure target is float for BCEWithLogitsLoss
        cross_entropy_loss = self.cross_entropy(input, target_float)

        # Store component losses (before weighting) for logging access
        self.component_tversky_loss = tversky_loss.item()
        self.component_ce_loss = cross_entropy_loss.item()

        # Combine losses with weights
        total_loss: torch.Tensor = (self.lambda_tversky * tversky_loss) + (self.lambda_ce * cross_entropy_loss)

        return total_loss 