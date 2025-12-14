#!/usr/bin/env python3
"""
Evaluation script for forgery detection model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
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
    parser = argparse.ArgumentParser(description="Evaluate forgery detection model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Load dataset
    dataset = load_dataset_from_config(config["data"])
    eval_data = dataset[args.split]
    print(f"Evaluating on {args.split} split: {len(eval_data)} samples")

    # TODO: Implement evaluation
    # 1. Load model from checkpoint
    # 2. Run inference on eval_data
    # 3. Compute metrics
    # 4. Save predictions to output_dir

    raise NotImplementedError("Evaluation not yet implemented")


if __name__ == "__main__":
    main()
