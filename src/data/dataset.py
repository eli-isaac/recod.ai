"""
PyTorch Dataset for forgery detection training.

Data flow:
1. Download from HuggingFace to local disk (if datasets specified)
2. Dataset class loads from local paths on-demand
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512) -> A.Compose:
    """Get training augmentations."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.05, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.ImageCompression(quality_range=(70, 100), p=0.3),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 512) -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])


def download_from_huggingface(
    dataset_ids: list[str],
    local_dir: str | Path,
) -> int:
    """
    Download datasets from HuggingFace and save to local disk.
    
    If local data exists, prompts user to: (r)eplace, (a)dd, or (c)ancel.
    
    Args:
        dataset_ids: List of HuggingFace dataset IDs
        local_dir: Local directory to save data (e.g., "data/pretrain")
        
    Returns:
        Number of samples downloaded
    """
    from datasets import load_dataset, concatenate_datasets
    
    local_dir = Path(local_dir)
    images_dir = local_dir / "images"
    masks_dir = local_dir / "masks"
    
    # Check if local data exists
    existing_count = 0
    if images_dir.exists():
        existing_count = len(list(images_dir.glob("*.jpg")))
    
    start_idx = 0
    if existing_count > 0:
        print(f"\nFound {existing_count} existing samples in {local_dir}")
        print("  (r)eplace - Delete existing data and download fresh")
        print("  (a)dd     - Add new data to existing (continue from where we left off)")
        print("  (c)ancel  - Abort download")
        
        while True:
            choice = input("\nChoice [r/a/c]: ").strip().lower()
            if choice in ('r', 'replace'):
                print("Deleting existing data...")
                shutil.rmtree(local_dir)
                break
            elif choice in ('a', 'add'):
                start_idx = existing_count
                print(f"Will add to existing data, starting from index {start_idx}")
                break
            elif choice in ('c', 'cancel'):
                print("Download cancelled.")
                return 0
            else:
                print("Invalid choice. Please enter 'r', 'a', or 'c'.")
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and combine all HuggingFace datasets
    print("\nLoading datasets from HuggingFace...")
    all_datasets = []
    for dataset_id in dataset_ids:
        print(f"  Loading {dataset_id}...")
        hf_ds = load_dataset(dataset_id, split="train")
        all_datasets.append(hf_ds)
        print(f"    → {len(hf_ds)} samples")
    
    if len(all_datasets) == 1:
        combined_ds = all_datasets[0]
    else:
        combined_ds = concatenate_datasets(all_datasets)
    
    print(f"Total from HuggingFace: {len(combined_ds)} samples")
    
    # Save to disk using HuggingFace's native parallel processing
    print(f"\nSaving to {local_dir}...")
    start_time = time.time()
    
    # Create a function that saves a single example
    def save_example(example, idx):
        img_path = images_dir / f"{start_idx + idx:06d}.jpg"
        mask_path = masks_dir / f"{start_idx + idx:06d}.npy"
        
        # Save image
        img = example["image"]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            img = Image.fromarray(np.array(img)).convert("RGB")
        img.save(str(img_path), quality=95)
        
        # Save mask
        mask = example["mask"]
        if mask is not None:
            if isinstance(mask, list):
                mask = np.array(mask)
            np.save(str(mask_path), mask.astype(np.uint8))
        
        return example
    
    # Use HuggingFace's map with multiprocessing - this parallelizes the Arrow decoding
    # Cap at 12 workers - disk I/O becomes the bottleneck, not CPU
    num_workers = min(os.cpu_count() or 8, 12)
    print(f"  Processing with {num_workers} workers...")
    combined_ds.map(
        save_example,
        with_indices=True,
        num_proc=num_workers,
        desc="Saving",
    )
    
    elapsed = time.time() - start_time
    saved_count = len(combined_ds)
    total_count = start_idx + saved_count
    print(f"\nDone! Saved {saved_count} samples in {elapsed:.1f}s ({saved_count/elapsed:.1f} samples/sec)")
    print(f"Total samples in {local_dir}: {total_count}")
    
    return saved_count


class ForgeryDataset(Dataset):
    """
    Dataset for forgery segmentation.
    
    Loads images and masks from local disk on-demand.
    Images stored as JPG, masks as NPY files.
    """
    
    def __init__(
        self,
        local_dir: str | Path,
        image_paths: list[Path] | None = None,
        img_size: int = 512,
        num_channels: int = 4,
        transform: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            local_dir: Directory with images/ and masks/ subdirs
            image_paths: Optional list of specific image paths to use (for train/val split)
            img_size: Target image size
            num_channels: Number of mask channels
            transform: Whether to apply augmentations
        """
        self.local_dir = Path(local_dir)
        self.images_dir = self.local_dir / "images"
        self.masks_dir = self.local_dir / "masks"
        self.img_size = img_size
        self.num_channels = num_channels
        
        # Use provided paths or discover all images
        if image_paths is not None:
            self.image_paths = image_paths
        else:
            self.image_paths = sorted(self.images_dir.glob("*.jpg"))
        
        # Setup transforms
        if transform:
            self.aug = get_train_transforms(img_size)
        else:
            self.aug = get_val_transforms(img_size)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            img: Image tensor, shape (3, H, W), values in [0, 1]
            mask: Mask tensor, shape (num_channels, H, W), binary values
        """
        img_path = self.image_paths[idx]
        
        # Load image
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        
        # Load mask if exists
        mask_path = self.masks_dir / f"{img_path.stem}.npy"
        mask = np.load(mask_path) if mask_path.exists() else None
        
        # Prepare mask array (H, W, C) for albumentations
        masks = np.zeros((h, w, self.num_channels), dtype=np.uint8)
        if mask is not None:
            num_masks = min(len(mask), self.num_channels)
            for i in range(num_masks):
                masks[:, :, i] = mask[i].astype(np.uint8)
        
        # Apply transforms
        transformed = self.aug(image=img, mask=masks)
        img_t = transformed['image']
        masks_t = transformed['mask']
        
        # Normalize image to [0, 1]
        img_t = img_t.float() / 255.0
        
        # Convert mask to (C, H, W) tensor
        if isinstance(masks_t, np.ndarray):
            masks_t = torch.from_numpy(masks_t.astype(np.float32)).permute(2, 0, 1)
        else:
            masks_t = masks_t.permute(2, 0, 1).float()
        
        return img_t, masks_t


def create_dataloaders(
    local_dir: str | Path,
    datasets: list[str] | None = None,
    img_size: int = 512,
    num_channels: int = 4,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    If HuggingFace dataset IDs are provided and local data doesn't exist,
    downloads the data first. Otherwise uses existing local data.
    
    Args:
        local_dir: Local directory for data (e.g., "data/pretrain" or "data/finetune")
        datasets: Optional list of HuggingFace dataset IDs to download
        img_size: Target image size
        num_channels: Number of mask channels
        batch_size: Batch size
        num_workers: Number of dataloader workers
        val_split: Validation split ratio
        seed: Random seed for split
        
    Returns:
        (train_loader, val_loader)
    """
    local_dir = Path(local_dir)
    images_dir = local_dir / "images"
    
    # Download from HuggingFace if datasets specified
    if datasets:
        download_from_huggingface(datasets, local_dir)

    # Get all image paths
    if not images_dir.exists():
        raise ValueError(f"No data found in {local_dir}. Provide HuggingFace datasets to download.")
    
    all_paths = sorted(images_dir.glob("*.jpg"))
    if not all_paths:
        raise ValueError(f"No images found in {images_dir}")
    
    print(f"Total samples: {len(all_paths)}")
    
    # Split into train/val
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_paths)
    val_size = int(len(all_paths) * val_split)
    val_paths = shuffled[:val_size]
    train_paths = shuffled[val_size:]
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets
    train_dataset = ForgeryDataset(
        local_dir=local_dir,
        image_paths=train_paths,
        img_size=img_size,
        num_channels=num_channels,
        transform=True,
    )
    val_dataset = ForgeryDataset(
        local_dir=local_dir,
        image_paths=val_paths,
        img_size=img_size,
        num_channels=num_channels,
        transform=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
