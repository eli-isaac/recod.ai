"""
Loss functions for forgery detection training.
"""

import torch
import torch.nn.functional as F
from functools import lru_cache
from itertools import permutations


@lru_cache(maxsize=8)
def _get_perm_tensor(num_channels: int, device: torch.device) -> torch.Tensor:
    """Cache permutation tensor per (C, device) to avoid recreation each batch."""
    all_perms = list(permutations(range(num_channels)))
    return torch.tensor(all_perms, device=device, dtype=torch.long)


def hungarian_matching(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Hungarian matching to find optimal channel assignment.
    
    Fully vectorized GPU implementation - no CPU transfers.
    Optimized for small C (<=6): uses brute-force over all permutations.
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        
    Returns:
        (matched_outputs, matched_targets): Both shape (B, C, H, W) with channels 
        reordered so that matched_outputs[b, i] corresponds to matched_targets[b, i]
    """
    B, C, H, W = outputs.shape
    device = outputs.device
    
    # Get cached permutation tensor (avoids recreation each batch)
    perm_tensor = _get_perm_tensor(C, device)
    num_perms = perm_tensor.shape[0]
    
    with torch.no_grad():
        # Compute cost matrix for all batch items at once: (B, C_pred, C_tgt)
        pred_flat = outputs.view(B, C, -1)  # (B, C, H*W)
        tgt_flat = targets.view(B, C, -1)   # (B, C, H*W)
        
        # Compute BCE for all pairs using broadcasting
        pred_for_cost = pred_flat.unsqueeze(2)  # (B, C, 1, H*W)
        tgt_for_cost = tgt_flat.unsqueeze(1)    # (B, 1, C, H*W)
        
        # BCE with logits for all pairs
        cost_matrix = F.binary_cross_entropy_with_logits(
            pred_for_cost.expand(-1, -1, C, -1),
            tgt_for_cost.expand(-1, C, -1, -1),
            reduction='none'
        ).mean(dim=-1)  # (B, C_pred, C_tgt)
        
        # Vectorized: compute cost for all permutations at once
        # perm_tensor: (num_perms, C) - each row is a permutation
        # For each perm, we need sum of cost_matrix[b, i, perm[i]] for i in range(C)
        
        # Gather costs for all permutations: (B, num_perms, C)
        # For each permutation p, gather cost_matrix[:, i, perm_tensor[p, i]] for all i
        perm_expanded = perm_tensor.unsqueeze(0).expand(B, -1, -1)  # (B, num_perms, C)
        
        # For channel i, we want cost_matrix[:, i, perm[i]]
        # Reshape cost_matrix for gathering: (B, C, C) -> gather along dim 2
        perm_costs = torch.zeros(B, num_perms, device=device)
        for i in range(C):
            # Get the target channel index for channel i in each permutation
            tgt_indices = perm_tensor[:, i]  # (num_perms,)
            # Gather: cost_matrix[:, i, tgt_indices] -> (B, num_perms)
            costs_i = cost_matrix[:, i, :].index_select(1, tgt_indices)  # (B, num_perms)
            perm_costs += costs_i
        
        # Get best permutation index for each batch item (stays on GPU)
        best_perm_idx = perm_costs.argmin(dim=1)  # (B,)
        
        # Get the best permutation for each batch item: (B, C)
        best_perms = perm_tensor[best_perm_idx]  # (B, C)
    
    # Apply best permutation using vectorized gather (no Python loops!)
    # outputs stays the same (we're matching targets to outputs)
    matched_outputs = outputs.contiguous()
    
    # Reorder targets according to best_perms
    # best_perms[b, i] tells us which target channel to put at position i
    # We need to gather targets along dim 1 using best_perms as indices
    best_perms_expanded = best_perms.view(B, C, 1, 1).expand(-1, -1, H, W)
    matched_targets = torch.gather(targets, dim=1, index=best_perms_expanded).contiguous()
    
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
    
    # Flatten spatial dimensions (reshape handles non-contiguous tensors)
    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)
    
    intersection = (probs_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
    
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
