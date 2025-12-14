#!/usr/bin/env python3
"""
Dataset creation script.

Downloads images from HuggingFace, creates forgeries using SAM3 segmentation,
and uploads the processed dataset back to HuggingFace.

Usage:
    python scripts/create_dataset.py --config configs/dataset_config.yaml
    python scripts/create_dataset.py --config configs/dataset_config.yaml --step download
    python scripts/create_dataset.py --config configs/dataset_config.yaml --step create
    python scripts/create_dataset.py --config configs/dataset_config.yaml --step upload
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DatasetConfig
from src.data.download import download_dataset
from src.data.pipeline import run_pipeline
from src.data.segmentation import load_sam3_model
from src.data.upload import upload_dataset


def step_download(config: DatasetConfig) -> None:
    """Step 1: Download dataset from HuggingFace."""
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOADING DATASET")
    print("=" * 60 + "\n")

    download_dataset(
        dataset_id=config.source.dataset_id,
        output_dir=config.storage.download_path,
        split=config.source.split,
        image_column=config.source.image_column,
        max_size_bytes=config.segmentation.max_image_size_bytes,
    )

    print(f"\nDownload complete. Images saved to: {config.storage.download_path}")


def step_create(config: DatasetConfig) -> None:
    """Step 2: Create forgeries using SAM3 segmentation (batch processing)."""
    print("\n" + "=" * 60)
    print("STEP 2: CREATING VARIATIONS AND MASKS")
    print("=" * 60 + "\n")

    # Load SAM3 model once
    print("Loading SAM3 model...")
    model, processor = load_sam3_model(
        model_id=config.segmentation.model_id,
    )

    # Run the batch processing pipeline
    results = run_pipeline(
        config=config,
        model=model,
        processor=processor,
    )

    print("\nCreation complete.")
    print(f"Total forgeries: {results['total_forgeries']}")
    print(f"Errors: {results['total_errors']}")


def step_upload(config: DatasetConfig) -> None:
    """Step 3: Upload processed dataset to HuggingFace."""
    print("\n" + "=" * 60)
    print("STEP 3: UPLOADING TO HUGGINGFACE")
    print("=" * 60 + "\n")

    if not config.storage.output_path.exists():
        print(f"Output directory does not exist: {config.storage.output_path}")
        print("Run the create step first.")
        return

    url = upload_dataset(
        output_dir=config.storage.output_path,
        repo_id=config.output.dataset_id,
        use_chunking=True,
        chunk_size=config.upload.chunk_size,
        private=config.output.private,
        ignore_patterns=config.upload.ignore_patterns,
    )

    print(f"\nUpload complete: {url}")


def main():
    parser = argparse.ArgumentParser(
        description="Create forgery dataset from HuggingFace source"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["download", "create", "upload", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    # Load typed configuration
    config = DatasetConfig.from_yaml(args.config)

    print("=" * 60)
    print("FORGERY DATASET CREATION")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Step: {args.step}")
    print(f"Source dataset: {config.source.dataset_id}")
    print(f"Output dataset: {config.output.dataset_id}")
    print(f"Batch size: {config.processing.process_batch_size}")

    # Run requested step(s)
    if args.step in ["download", "all"]:
        step_download(config)

    if args.step in ["create", "all"]:
        step_create(config)

    if args.step in ["upload", "all"]:
        step_upload(config)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
