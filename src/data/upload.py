"""
Upload dataset to HuggingFace Hub.
"""

from pathlib import Path

import numpy as np
from datasets import Dataset, Image
from huggingface_hub import HfApi


def create_dataset_from_files(
    images_dir: Path,
    masks_dir: Path,
) -> Dataset | None:
    """
    Create a HuggingFace Dataset from local image and mask files.
    
    Args:
        images_dir: Directory containing PNG images
        masks_dir: Directory containing NPY mask files
        
    Returns:
        Dataset or None if no data found
    """
    image_paths = sorted(images_dir.glob("*.png"))
    
    if not image_paths:
        print("No images found")
        return None
    
    data = {
        "image": [],
        "mask": [],
    }
    
    for img_path in image_paths:
        sample_id = img_path.stem
        mask_path = masks_dir / f"{sample_id}.npy"
        
        if not mask_path.exists():
            print(f"Warning: No mask for {sample_id}, skipping")
            continue
        
        # Load mask
        mask = np.load(mask_path)
        
        data["image"].append(str(img_path))  # datasets.Image loads from path
        data["mask"].append(mask)
    
    if not data["image"]:
        print("No valid samples found")
        return None
    
    print(f"Preparing {len(data['image'])} samples...")
    
    # Create dataset with proper features
    dataset = Dataset.from_dict(data)
    
    # Cast image column to Image feature for proper handling
    dataset = dataset.cast_column("image", Image())
    
    return dataset


def upload_dataset(
    output_dir: Path,
    repo_id: str,
    private: bool = False,
    max_shard_size: str = "500MB",
) -> str:
    """
    Upload a local dataset directory to HuggingFace Hub.
    
    Uses the `datasets` library for efficient upload with:
    - Proper Arrow/Parquet format
    - Automatic sharding for large datasets
    
    Args:
        output_dir: Local directory containing images/ and masks/ subdirs
        repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
        private: Whether the dataset should be private
        max_shard_size: Max size per Parquet shard (default 500MB)
        
    Returns:
        URL of the uploaded dataset
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory does not exist: {masks_dir}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    
    # Create dataset from local files
    dataset = create_dataset_from_files(images_dir, masks_dir)
    
    if dataset is None:
        raise ValueError("No data to upload")
    
    print(f"Uploading {len(dataset)} samples to {repo_id}...")
    
    # Push to hub with efficient settings
    dataset.push_to_hub(
        repo_id,
        private=private,
        max_shard_size=max_shard_size,
    )
    
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Upload complete: {url}")
    return url
