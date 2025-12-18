"""
Download dataset from HuggingFace.
"""

import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset as hf_load_dataset
from PIL import Image


def download_dataset(
    dataset_id: str,
    output_dir: Path,
    split: str = "train",
    image_column: str = "image",
    filename_column: str | None = None,
) -> None:
    """
    Download images from a HuggingFace dataset to local directory.
    Uses parallel processing for speed.

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
    print(f"Total: {len(dataset)} samples")

    # Save function for parallel processing
    def save_image(example, idx):
        image = example[image_column]
        
        # Get filename
        if filename_column and filename_column in example:
            filename = Path(example[filename_column]).stem + ".png"
        elif hasattr(image, "filename") and image.filename:
            filename = Path(image.filename).stem + ".png"
        else:
            filename = f"{idx:06d}.png"
        
        if isinstance(image, Image.Image):
            image.save(str(output_dir / filename))
        else:
            Image.fromarray(np.array(image)).save(str(output_dir / filename))
        
        return example

    # Use parallel processing
    num_workers = min(os.cpu_count() or 8, 16)
    print(f"Saving with {num_workers} workers...")
    start_time = time.time()
    
    dataset.map(
        save_image,
        with_indices=True,
        num_proc=num_workers,
        desc="Downloading",
    )
    
    elapsed = time.time() - start_time
    print(f"Done. Saved {len(dataset)} images to {output_dir} in {elapsed:.1f}s")


def download_training_data(
    dataset_id: str,
    config_name: str | None,
    output_dir: Path,
    split: str = "train",
) -> None:
    """
    Download training dataset (images + masks) from HuggingFace to local directory.
    
    Creates the following structure:
        output_dir/
        ├── images/
        │   ├── 000000.png
        │   └── ...
        └── masks/
            ├── 000000.npy
            └── ...
    
    Args:
        dataset_id: HuggingFace dataset ID
        config_name: Dataset config name (e.g., "pretrain")
        output_dir: Directory to save data
        split: Dataset split to download
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {dataset_id}" + (f" ({config_name})" if config_name else ""))
    
    if config_name:
        dataset = hf_load_dataset(dataset_id, config_name, split=split)
    else:
        dataset = hf_load_dataset(dataset_id, split=split)
    
    print(f"Saving {len(dataset)} samples...")
    
    for idx, item in enumerate(tqdm(dataset, desc="Downloading")):
        filename = f"{idx:06d}"
        
        # Save image
        image = item["image"]
        image.save(images_dir / f"{filename}.png")
        
        # Save mask if present
        mask = item.get("mask")
        if mask is not None:
            np.save(masks_dir / f"{filename}.npy", np.array(mask))
    
    print(f"Done. Saved to {output_dir}")
    print(f"  - Images: {len(list(images_dir.glob('*.png')))}")
    print(f"  - Masks: {len(list(masks_dir.glob('*.npy')))}")
