"""
PyTorch Dataset for forgery detection training.

Supports loading from HuggingFace datasets or local files.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512) -> A.Compose:
    """Get training augmentations."""
    return A.Compose([
        A.Resize(img_size, img_size),
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Color augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.05, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        # Noise/blur for robustness
        # A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # A.GaussNoise(p=0.2),
        # JPEG compression artifacts (relevant for forgery detection)
        A.ImageCompression(quality_range=(70, 100), p=0.3),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 512) -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])


class ForgeryDataset(Dataset):
    """
    Dataset for forgery segmentation.
    
    Supports two modes:
    1. HuggingFace dataset (loaded via datasets library)
    2. Local files (images/ and masks/ directories)
    
    For authentic images (no forgery), mask is None/zeros.
    """
    
    def __init__(
        self,
        hf_dataset: Any | None = None,
        local_dir: Path | None = None,
        img_size: int = 512,
        num_channels: int = 4,
        transform: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            hf_dataset: HuggingFace dataset split (from datasets.load_dataset)
            local_dir: Local directory with images/ and masks/ subdirs
            img_size: Target image size
            num_channels: Number of mask channels
            transform: Whether to apply augmentations
        """
        self.img_size = img_size
        self.num_channels = num_channels
        self.transform = transform
        
        if hf_dataset is not None:
            self.mode = "huggingface"
            self.dataset = hf_dataset
            self.length = len(hf_dataset)
        elif local_dir is not None:
            self.mode = "local"
            self.local_dir = Path(local_dir)
            self.images_dir = self.local_dir / "images"
            self.masks_dir = self.local_dir / "masks"
            self.image_paths = sorted(self.images_dir.glob("*.png"))
            self.length = len(self.image_paths)
        else:
            raise ValueError("Either hf_dataset or local_dir must be provided")
        
        # Setup transforms
        if transform:
            self.aug = get_train_transforms(img_size)
        else:
            self.aug = get_val_transforms(img_size)
    
    def __len__(self) -> int:
        return self.length
    
    def _load_huggingface_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Load sample from HuggingFace dataset."""
        sample = self.dataset[idx]
        
        # Image comes as PIL Image from datasets.Image feature
        img = sample["image"]
        if not isinstance(img, np.ndarray):
            img = np.array(img.convert("RGB"))
        
        # Mask may be None for authentic images
        mask = sample["mask"]
        
        return img, mask
    
    def _load_local_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Load sample from local files."""
        img_path = self.image_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # Check for corresponding mask
        mask_path = self.masks_dir / f"{img_path.stem}.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
        else:
            mask = None
        
        return img, mask
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            img: Image tensor, shape (3, H, W), values in [0, 1]
            mask: Mask tensor, shape (num_channels, H, W), binary values
        """
        if self.mode == "huggingface":
            img, mask = self._load_huggingface_sample(idx)
        else:
            img, mask = self._load_local_sample(idx)
        
        h, w = img.shape[:2]
        
        # Prepare mask array (H, W, C) for albumentations
        masks = np.zeros((h, w, self.num_channels), dtype=np.uint8)
        if mask is not None:
            num_masks = min(len(mask), self.num_channels)
            for i in range(num_masks):
                masks[:, :, i] = mask[i].astype(np.uint8)
        
        # Apply transforms
        transformed = self.aug(image=img, mask=masks)
        img_t = transformed['image']  # (C, H, W) tensor from ToTensorV2
        masks_t = transformed['mask']  # (H, W, C) numpy array
        
        # Normalize image to [0, 1] (ToTensorV2 keeps [0, 255])
        img_t = img_t.float() / 255.0
        
        # Convert mask to (C, H, W) tensor
        if isinstance(masks_t, np.ndarray):
            masks_t = torch.from_numpy(masks_t.astype(np.float32)).permute(2, 0, 1)
        else:
            masks_t = masks_t.permute(2, 0, 1).float()
        
        return img_t, masks_t


def create_dataloaders(
    datasets: list[str],
    img_size: int = 512,
    num_channels: int = 4,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Create train and validation dataloaders from multiple HuggingFace datasets.
    
    Args:
        datasets: List of HuggingFace dataset IDs to load and combine
        img_size: Target image size
        num_channels: Number of mask channels
        batch_size: Batch size
        num_workers: Number of dataloader workers
        val_split: Validation split ratio
        seed: Random seed for split
        
    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    from datasets import load_dataset, concatenate_datasets
    
    if not datasets:
        raise ValueError("No datasets provided")
    
    # Load and combine all datasets
    all_datasets = []
    for dataset_id in datasets:
        print(f"Loading {dataset_id}...")
        hf_ds = load_dataset(dataset_id, split="train")
        all_datasets.append(hf_ds)
        print(f"  → {len(hf_ds)} samples")
    
    # Concatenate all datasets
    if len(all_datasets) == 1:
        combined_ds = all_datasets[0]
    else:
        combined_ds = concatenate_datasets(all_datasets)
    
    print(f"Total: {len(combined_ds)} samples")
    
    # Split into train/val
    split_ds = combined_ds.train_test_split(test_size=val_split, seed=seed)
    
    train_dataset = ForgeryDataset(
        hf_dataset=split_ds["train"],
        img_size=img_size,
        num_channels=num_channels,
        transform=True,
    )
    val_dataset = ForgeryDataset(
        hf_dataset=split_ds["test"],
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
