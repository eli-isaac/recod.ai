#!/usr/bin/env python3
"""
Download dataset from Hugging Face or Kaggle.
"""

import argparse
from pathlib import Path


def download_from_huggingface(output_dir: Path):
    """Download dataset from Hugging Face."""
    # TODO: Implement HuggingFace download
    # from datasets import load_dataset
    # dataset = load_dataset("YOUR_USERNAME/forgery-detection")
    raise NotImplementedError("HuggingFace download not yet implemented")


def download_from_kaggle(output_dir: Path):
    """Download dataset from Kaggle."""
    # TODO: Implement Kaggle download
    # import kaggle
    # kaggle.api.competition_download_files('competition-name', path=output_dir)
    raise NotImplementedError("Kaggle download not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Download forgery detection dataset")
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "kaggle"],
        default="huggingface",
        help="Data source to download from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded data",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        download_from_huggingface(args.output_dir)
    else:
        download_from_kaggle(args.output_dir)

    print(f"Data downloaded to {args.output_dir}")


if __name__ == "__main__":
    main()
