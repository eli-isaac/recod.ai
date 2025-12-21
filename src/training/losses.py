"""
Loss functions for forgery detection training.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Callable


def hungarian_matching(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Hungarian matching to find optimal channel assignment.

    Uses scipy's linear_sum_assignment for efficient matching.

    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        cost_fn: Function to compute cost between prediction and target channels.
                Takes two tensors of shape (B, C, C, H*W) and returns a cost tensor
                of the same shape. Defaults to binary_cross_entropy_with_logits.

    Returns:
        (matched_outputs, matched_targets): Both shape (B, C, H, W) with channels
        reordered so that matched_outputs[b, i] corresponds to matched_targets[b, i]
    """
    B, C, H, W = outputs.shape
    device = outputs.device

    # Default cost function: BCE with logits
    if cost_fn is None:

        def default_cost_fn(pred, tgt):
            return F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")

        cost_fn = default_cost_fn

    matched_targets_list = []

    with torch.no_grad():
        # Compute cost matrix for all batch items at once: (B, C_pred, C_tgt)
        pred_flat = outputs.view(B, C, -1)  # (B, C, H*W)
        tgt_flat = targets.view(B, C, -1)  # (B, C, H*W)

        # Compute cost for all pairs using broadcasting
        pred_for_cost = pred_flat.unsqueeze(2)  # (B, C, 1, H*W)
        tgt_for_cost = tgt_flat.unsqueeze(1)  # (B, 1, C, H*W)

        # Expand for all pairs: (B, C, C, H*W)
        pred_expanded = pred_for_cost.expand(-1, -1, C, -1)
        tgt_expanded = tgt_for_cost.expand(-1, C, -1, -1)

        # Compute cost matrix using provided function
        cost_matrix = cost_fn(pred_expanded, tgt_expanded).mean(
            dim=-1
        )  # (B, C_pred, C_tgt)

        # Apply Hungarian matching for each batch item
        for b in range(B):
            # Find optimal assignment using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix[b].cpu().numpy())

            # Reorder targets: col_ind[i] is the target channel matched to output channel i
            matched_targets_b = targets[b, col_ind, :, :]  # (C, H, W)
            matched_targets_list.append(matched_targets_b)

    # Stack matched targets: (B, C, H, W)
    matched_targets = torch.stack(matched_targets_list).to(device)

    # Outputs stay the same (we're matching targets to outputs)
    matched_outputs = outputs.contiguous()

    return matched_outputs, matched_targets


def bce_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """
    Standard BCE loss without Hungarian matching.

    Use this when channel ordering is consistent (e.g., for classification).

    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        pos_weight: Positive class weight (tensor preferred to avoid CPU->GPU transfer)

    Returns:
        Mean BCE loss
    """
    # Handle pos_weight: prefer pre-created tensor to avoid CPU->GPU transfer
    if pos_weight is None:
        pos_weight_tensor = None
    elif isinstance(pos_weight, torch.Tensor):
        pos_weight_tensor = pos_weight
    else:
        pos_weight_tensor = torch.tensor([pos_weight], device=outputs.device)

    return F.binary_cross_entropy_with_logits(
        outputs, targets, pos_weight=pos_weight_tensor, reduction="mean"
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

    # Flatten spatial dimensions (reshape handles non-contiguous tensors)
    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)

    intersection = (probs_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        probs_flat.sum() + targets_flat.sum() + smooth
    )

    return 1 - dice


def combined_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
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
