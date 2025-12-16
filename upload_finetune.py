#!/usr/bin/env python3
"""
Temporary script to upload train_images and train_masks to HuggingFace as finetune config.

Structure:
- train_images/forged/ -> images with masks
- train_images/authentic/ -> images without masks  
- train_masks/ -> npy files matching forged images (uint8 0/1, needs bool conversion)

Usage:
    python upload_finetune.py
"""

from pathlib import Path

import numpy as np
from datasets import Dataset, Image
from huggingface_hub import HfApi


def create_dataset_from_files(
    forged_dir: Path,
    authentic_dir: Path,
    masks_dir: Path,
) -> Dataset | None:
    """
    Create a HuggingFace Dataset from local image and mask files.
    
    Args:
        forged_dir: Directory containing forged PNG images
        authentic_dir: Directory containing authentic PNG images
        masks_dir: Directory containing NPY mask files (matching forged images)
        
    Returns:
        Dataset or None if no data found
    """
    forged_paths = sorted(forged_dir.glob("*.png"))
    authentic_paths = sorted(authentic_dir.glob("*.png"))
    
    if not forged_paths and not authentic_paths:
        print("No images found")
        return None
    
    print(f"Preparing {len(forged_paths)} forged + {len(authentic_paths)} authentic = {len(forged_paths) + len(authentic_paths)} total samples...")
    
    def data_generator():
        # Forged images (have masks)
        for img_path in forged_paths:
            sample_id = img_path.stem
            mask_path = masks_dir / f"{sample_id}.npy"
            
            mask = None
            if mask_path.exists():
                # Load and convert uint8 (0/1) to bool
                mask = np.load(mask_path).astype(bool)
            
            yield {
                "image": str(img_path),
                "mask": mask,
            }
        
        # Authentic images (no masks)
        for img_path in authentic_paths:
            yield {
                "image": str(img_path),
                "mask": None,
            }
    
    # Create dataset using generator to avoid memory overflow
    # Use smaller writer_batch_size to avoid Arrow 2GB batch limit with large mask arrays
    dataset = Dataset.from_generator(data_generator, writer_batch_size=500)
    
    # Cast image column to Image feature for proper handling
    dataset = dataset.cast_column("image", Image())
    
    return dataset


def main():
    base_dir = Path(__file__).parent
    forged_dir = base_dir / "train_images" / "forged"
    authentic_dir = base_dir / "train_images" / "authentic"
    masks_dir = base_dir / "train_masks"
    
    repo_id = "eliplutchok/recod-finetune"
    
    print("=" * 60)
    print("FINETUNE DATASET UPLOAD")
    print("=" * 60)
    print(f"Forged images: {forged_dir}")
    print(f"Authentic images: {authentic_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Output: {repo_id}")
    print()
    
    # Verify directories exist
    if not forged_dir.exists():
        raise FileNotFoundError(f"Forged directory does not exist: {forged_dir}")
    if not authentic_dir.exists():
        raise FileNotFoundError(f"Authentic directory does not exist: {authentic_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory does not exist: {masks_dir}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    
    # Create dataset from local files
    dataset = create_dataset_from_files(forged_dir, authentic_dir, masks_dir)
    
    if dataset is None:
        raise ValueError("No data to upload")
    
    print(f"Uploading {len(dataset)} samples to {repo_id}...")
    
    # Push to hub with efficient settings
    dataset.push_to_hub(
        repo_id,
        private=False,
        max_shard_size="500MB",
    )
    
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Upload complete: {url}")
    print(f"Load with: load_dataset(\"{repo_id}\")")


if __name__ == "__main__":
    main()
