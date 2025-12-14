#!/usr/bin/env python3
"""
Main training script for forgery detection model.
"""

import argparse
from pathlib import Path

import yaml


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

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Config: {config}")

    # TODO: Implement training loop
    # 1. Setup data loaders
    # 2. Initialize model
    # 3. Setup optimizer and scheduler
    # 4. Training loop with validation
    # 5. Save checkpoints

    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
