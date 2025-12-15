"""
Loss functions for forgery detection training.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_matched_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """
    Compute BCE loss with Hungarian matching between predicted and ground truth channels.
    
    This handles the assignment problem when the order of object masks doesn't matter.
    Uses the Hungarian algorithm to find the optimal matching between predicted and
    ground truth channels that minimizes the total loss.
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        pos_weight: Positive class weight for BCE loss (to handle class imbalance)
        
    Returns:
        Mean matched loss across batch
    """
    B, C, H, W = outputs.shape
    device = outputs.device
    
    # Convert pos_weight to tensor if provided
    pos_weight_tensor = torch.tensor([pos_weight], device=device) if pos_weight else None
    
    matched_losses = []
    
    for b in range(B):
        pred = outputs[b]  # (C, H, W)
        tgt = targets[b]   # (C, H, W)
        
        # Build cost matrix: cost[i,j] = BCE(pred_channel_i, gt_channel_j)
        with torch.no_grad():
            cost_matrix = torch.zeros(C, C, device=device)
            for i in range(C):
                for j in range(C):
                    cost_matrix[i, j] = F.binary_cross_entropy_with_logits(
                        pred[i], tgt[j],
                        pos_weight=pos_weight_tensor,
                        reduction='mean'
                    )
        
        # Hungarian matching (find optimal assignment)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        # Compute loss for matched pairs (WITH gradients)
        for r, c in zip(row_ind, col_ind):
            loss = F.binary_cross_entropy_with_logits(
                pred[r], tgt[c],
                pos_weight=pos_weight_tensor,
                reduction='mean'
            )
            matched_losses.append(loss)
    
    return torch.stack(matched_losses).mean()


def bce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """
    Standard BCE loss without Hungarian matching.
    
    Use this when channel ordering is consistent (e.g., for classification).
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        pos_weight: Positive class weight
        
    Returns:
        Mean BCE loss
    """
    pos_weight_tensor = torch.tensor([pos_weight], device=outputs.device) if pos_weight else None
    return F.binary_cross_entropy_with_logits(
        outputs, targets,
        pos_weight=pos_weight_tensor,
        reduction='mean'
    )


def dice_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss for segmentation.
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Mean Dice loss (1 - Dice coefficient)
    """
    probs = torch.sigmoid(outputs)
    
    # Flatten spatial dimensions
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (probs_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    
    return 1 - dice


def combined_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float | None = None,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    use_hungarian: bool = True,
) -> torch.Tensor:
    """
    Combined BCE + Dice loss.
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        pos_weight: Positive class weight for BCE
        bce_weight: Weight for BCE loss component
        dice_weight: Weight for Dice loss component
        use_hungarian: Whether to use Hungarian matching for BCE
        
    Returns:
        Combined loss
    """
    if use_hungarian:
        bce = hungarian_matched_loss(outputs, targets, pos_weight)
    else:
        bce = bce_loss(outputs, targets, pos_weight)
    
    dice = dice_loss(outputs, targets)
    
    return bce_weight * bce + dice_weight * dice
