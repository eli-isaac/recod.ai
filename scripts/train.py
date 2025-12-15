#!/usr/bin/env python3
"""
Main training script for forgery detection model.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --resume checkpoints/checkpoint_epoch_10.pt
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.data.download import download_training_data
from src.models.dino_segmenter import DinoSegmenter
from src.training.config import TrainConfig
from src.training.trainer import train_model
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train forgery detection model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = TrainConfig.from_yaml(str(args.config))
    print(f"Loaded config from {args.config}")

    # Set seed
    set_seed(config.seed)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Download data if needed
    train_dir = Path(config.data.train_dir)
    if config.data.download:
        if not config.data.dataset_id:
            print("Error: download=True but no dataset_id specified")
            sys.exit(1)
        
        # Check if data already exists
        images_dir = train_dir / "images"
        if images_dir.exists() and len(list(images_dir.glob("*.png"))) > 0:
            print(f"\nData already exists at {train_dir}, skipping download")
        else:
            print("\n" + "=" * 60)
            print("STEP 1: DOWNLOADING DATA")
            print("=" * 60)
            download_training_data(
                dataset_id=config.data.dataset_id,
                config_name=config.data.config_name,
                output_dir=train_dir,
            )
    else:
        if not train_dir.exists():
            print(f"Error: download=False but train_dir does not exist: {train_dir}")
            sys.exit(1)

    # Create dataloaders from local data
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        local_dir=train_dir,
        img_size=config.data.img_size,
        num_channels=config.model.out_channels,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        val_split=config.data.val_split,
        seed=config.seed,
    )

    # Create model
    print("\nCreating model...")
    model = DinoSegmenter(
        backbone=config.model.backbone,
        out_channels=config.model.out_channels,
        unfreeze_blocks=config.model.unfreeze_blocks,
        decoder_dropout=config.model.decoder_dropout,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    results = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        resume_from=str(args.resume) if args.resume else None,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Epochs trained: {results['epochs_trained']}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Best validation F1: {results['best_val_f1']:.4f}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
