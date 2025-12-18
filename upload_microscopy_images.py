#!/usr/bin/env python3
"""
Temporary script to upload microscopy images to HuggingFace.

Uploads images from the GFS-ExtremeNet dataset (Babesia, Toxoplasma, Trypanosoma)
to a HuggingFace dataset repository.

Usage:
    huggingface-cli login  # login first
    python upload_microscopy_images.py --repo-id your-username/dataset-name
"""

import argparse
from pathlib import Path

from datasets import Dataset, Features, Image, ClassLabel, Value
from huggingface_hub import HfApi


def collect_images(base_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Collect all image paths from the dataset folders."""
    image_paths = []
    labels = []
    filenames = []

    parasites = ["Babesia", "Toxoplasma", "Trypanosoma"]

    for parasite in parasites:
        # Get train images (main split we care about)
        train_dir = base_path / parasite / "images" / "train2017"

        if not train_dir.exists():
            print(f"Warning: {train_dir} does not exist, skipping")
            continue

        for img_path in sorted(train_dir.glob("*.png")):
            image_paths.append(str(img_path))
            labels.append(parasite)
            filenames.append(img_path.name)

        print(f"Found {sum(1 for l in labels if l == parasite)} images for {parasite}")

    return image_paths, labels, filenames


def main():
    parser = argparse.ArgumentParser(description="Upload microscopy images to HuggingFace")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/microscopy-parasites)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/Users/eliplutchok/Downloads/Microscopy image datasets/GFS-ExtremeNet/dataset",
        help="Path to the GFS-ExtremeNet dataset folder",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )
    args = parser.parse_args()

    base_path = Path(args.data_path)

    print("Collecting images...")
    image_paths, labels, filenames = collect_images(base_path)

    print(f"\nTotal images: {len(image_paths)}")

    # Create dataset with image column
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=["Babesia", "Toxoplasma", "Trypanosoma"]),
        "filename": Value("string"),
    })

    dataset = Dataset.from_dict(
        {
            "image": image_paths,
            "label": labels,
            "filename": filenames,
        },
        features=features,
    )

    print(f"\nDataset created: {dataset}")
    print(f"\nUploading to {args.repo_id}...")

    dataset.push_to_hub(
        args.repo_id,
        private=args.private,
    )

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
