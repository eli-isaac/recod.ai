"""
PyTorch Dataset for forgery detection training.

Loads directly from HuggingFace datasets using Arrow format.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512) -> A.Compose:
    """Get training augmentations for forgery detection."""
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-15, 15), p=0.5),

        # Color transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),

        # Noise and blur (mild - don't destroy forgery artifacts)
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.1),

        # Compression artifacts (common in real images)
        A.ImageCompression(quality_range=(60, 100), p=0.4),

        # Dropout (helps with occlusion robustness)
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2),

        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 512) -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])


class ForgeryDataset(Dataset):
    """Dataset for forgery segmentation from HuggingFace datasets."""

    def __init__(
        self,
        hf_dataset,
        img_size: int = 512,
        num_channels: int = 4,
        transform: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            hf_dataset: HuggingFace dataset object
            img_size: Target image size
            num_channels: Number of mask channels
            transform: Whether to apply augmentations
        """
        self.dataset = hf_dataset
        self.img_size = img_size
        self.num_channels = num_channels
        self.aug = get_train_transforms(img_size) if transform else get_val_transforms(img_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            img: Image tensor, shape (3, H, W), values in [0, 1]
            mask: Mask tensor, shape (num_channels, H, W), binary values
        """
        example = self.dataset[idx]

        # Load image
        img = example["image"]
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        img = np.array(img)
        h, w = img.shape[:2]

        # Load mask
        mask = example.get("mask")
        if mask is not None and isinstance(mask, list):
            mask = np.array(mask)

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
            masks_t = torch.from_numpy(masks_t.astype(np.float32))

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
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from HuggingFace datasets.

    Args:
        datasets: List of HuggingFace dataset IDs
        img_size: Target image size
        num_channels: Number of mask channels
        batch_size: Batch size
        num_workers: Number of dataloader workers
        val_split: Validation split ratio
        seed: Random seed for split

    Returns:
        (train_loader, val_loader)
    """
    from datasets import load_dataset, concatenate_datasets

    print("Loading datasets from HuggingFace...")
    all_datasets = []
    for dataset_id in datasets:
        print(f"  Loading {dataset_id}...")
        hf_ds = load_dataset(dataset_id, split="train")
        all_datasets.append(hf_ds)
        print(f"    → {len(hf_ds)} samples")

    if len(all_datasets) == 1:
        combined_ds = all_datasets[0]
    else:
        combined_ds = concatenate_datasets(all_datasets)

    print(f"Total samples: {len(combined_ds)}")

    # Split into train/val
    split = combined_ds.train_test_split(test_size=val_split, seed=seed)
    train_hf = split["train"]
    val_hf = split["test"]

    print(f"Train: {len(train_hf)}, Val: {len(val_hf)}")

    train_dataset = ForgeryDataset(train_hf, img_size, num_channels, transform=True)
    val_dataset = ForgeryDataset(val_hf, img_size, num_channels, transform=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
