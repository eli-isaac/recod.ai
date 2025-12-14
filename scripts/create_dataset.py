#!/usr/bin/env python3
"""
Dataset creation and preprocessing script.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create/preprocess dataset")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Input directory with raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # TODO: Implement dataset creation
    # 1. Load raw data
    # 2. Process/clean images
    # 3. Create train/val split (group-aware)
    # 4. Generate metadata.csv
    # 5. Save processed data

    raise NotImplementedError("Dataset creation not yet implemented")


if __name__ == "__main__":
    main()
