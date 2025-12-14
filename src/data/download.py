"""
Download dataset from HuggingFace.
"""

from pathlib import Path

from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm


def download_dataset(
    dataset_id: str,
    output_dir: Path,
    split: str = "train",
    image_column: str = "image",
    filename_column: str | None = None,
) -> None:
    """
    Download images from a HuggingFace dataset to local directory.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "username/dataset-name")
        output_dir: Directory to save downloaded images
        split: Dataset split to download
        image_column: Name of the column containing images
        filename_column: Column with filenames (if None, uses index)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_id} (split: {split})")
    dataset = hf_load_dataset(dataset_id, split=split)

    for idx, item in enumerate(tqdm(dataset, desc="Downloading")):
        image = item[image_column]

        # Get filename from column, image attribute, or use index
        if filename_column and filename_column in item:
            filename = Path(item[filename_column]).stem + ".png"
        elif hasattr(image, "filename") and image.filename:
            filename = Path(image.filename).stem + ".png"
        else:
            filename = f"{idx:06d}.png"

        image.save(output_dir / filename)

    print(f"Done. Saved {len(dataset)} images to {output_dir}")
