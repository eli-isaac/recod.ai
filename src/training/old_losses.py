"""
Loss functions for forgery detection training.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_matching(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Hungarian matching to find optimal channel assignment.
    
    Returns reordered predictions and targets where channels are optimally matched.
    This should be used BEFORE computing any metrics (loss, F1, etc).
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        
    Returns:
        (matched_outputs, matched_targets): Both shape (B, C, H, W) with channels 
        reordered so that matched_outputs[b, i] corresponds to matched_targets[b, i]
    """
    B, C, H, W = outputs.shape
    device = outputs.device
    
    matched_outputs = torch.zeros_like(outputs)
    matched_targets = torch.zeros_like(targets)
    
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
                        reduction='mean'
                    )
        
        # Hungarian matching (find optimal assignment)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        # Reorder predictions to match targets
        for new_idx, (pred_idx, tgt_idx) in enumerate(zip(row_ind, col_ind)):
            matched_outputs[b, new_idx] = outputs[b, pred_idx]
            matched_targets[b, new_idx] = targets[b, tgt_idx]
    
    return matched_outputs, matched_targets


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
) -> torch.Tensor:
    """
    Combined BCE + Dice loss.
    
    Note: Apply hungarian_matching() BEFORE calling this if channel order doesn't matter.
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        pos_weight: Positive class weight for BCE
        bce_weight: Weight for BCE loss component
        dice_weight: Weight for Dice loss component
        
    Returns:
        Combined loss
    """
    bce = bce_loss(outputs, targets, pos_weight)
    dice = dice_loss(outputs, targets)
    
    return bce_weight * bce + dice_weight * dice