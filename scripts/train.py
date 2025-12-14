#!/usr/bin/env python3
"""
Main training script for forgery detection model.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset_from_config


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train forgery detection model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Load dataset (downloads from HF if needed)
    dataset = load_dataset_from_config(config["data"])
    print(f"Dataset loaded: {dataset}")

    # TODO: Implement training loop
    # 1. Create PyTorch DataLoaders from HF dataset
    # 2. Initialize model
    # 3. Setup optimizer and scheduler
    # 4. Training loop with validation
    # 5. Save checkpoints

    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
