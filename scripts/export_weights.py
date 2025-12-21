#!/usr/bin/env python3
"""
Export clean weights from a training checkpoint for Kaggle submission.

Handles:
- Stripping '_orig_mod.' prefix from torch.compile()
- Extracting just the model weights (no config, optimizer, etc.)

Usage:
    python scripts/export_weights.py checkpoints/best_model.pt
    python scripts/export_weights.py checkpoints/best_model.pt -o weights.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path (needed to unpickle TrainConfig)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def export_weights(input_path: str, output_path: str | None = None) -> str:
    """Export clean weights from a checkpoint."""
    
    # Default output path
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_clean.pt")
    
    # Load checkpoint
    print(f"Loading {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Extracted model_state_dict from checkpoint")
    else:
        state_dict = checkpoint
    
    # Strip '_orig_mod.' prefix (from torch.compile)
    new_state_dict = {}
    stripped = 0
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_key = k.replace('_orig_mod.', '')
            stripped += 1
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    if stripped > 0:
        print(f"  Stripped '_orig_mod.' prefix from {stripped} keys")
    
    # Save
    torch.save(new_state_dict, output_path)
    print(f"✅ Saved to {output_path}")
    print(f"   {len(new_state_dict)} parameters")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export clean weights for Kaggle")
    parser.add_argument("checkpoint", type=str, help="Input checkpoint file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path")
    args = parser.parse_args()
    
    export_weights(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

