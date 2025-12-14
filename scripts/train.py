#!/usr/bin/env python3
"""
Main training script for forgery detection model.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config


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

    data_dir = Path(config["data"].get("data_dir", "data/raw"))
    print(f"Data directory: {data_dir}")

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        print("Run 'python scripts/download_data.py' first.")
        sys.exit(1)

    # TODO: Implement training loop
    # 1. Create PyTorch Dataset from local image paths
    # 2. Create DataLoaders
    # 3. Initialize model
    # 4. Setup optimizer and scheduler
    # 5. Training loop with validation
    # 6. Save checkpoints

    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
