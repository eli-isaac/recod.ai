#!/usr/bin/env python3
"""
Download dataset from Hugging Face.

This script pre-downloads the dataset so training can run offline.
Note: The train.py script will also auto-download if data isn't cached.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dataset "username/dataset-name"
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Download forgery detection dataset from HuggingFace")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to config file (to read dataset ID)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset ID (overrides config)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (overrides config)",
    )
    args = parser.parse_args()

    # Load config for defaults
    config = load_config(args.config)
    data_config = config["data"]

    # Use CLI args or fall back to config
    dataset_id = args.dataset or data_config["dataset"]
    cache_dir = args.cache_dir or Path(data_config.get("cache_dir", "data/"))

    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_id=dataset_id, cache_dir=cache_dir)

    print(f"\nDataset downloaded successfully!")
    print(f"Dataset info: {dataset}")
    print(f"\nCached at: {cache_dir}")


if __name__ == "__main__":
    main()
