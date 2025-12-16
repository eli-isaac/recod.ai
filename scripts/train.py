#!/usr/bin/env python3
"""
Main training script for forgery detection model.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
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
    args = parser.parse_args()

    # Load config
    config = TrainConfig.from_yaml(str(args.config))
    print(f"Loaded config from {args.config}")

    # Set seed
    set_seed(config.seed)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Validate datasets
    if not config.data.datasets:
        print("Error: No datasets specified in config")
        sys.exit(1)

    print(f"\nDatasets: {config.data.datasets}")

    # Create dataloaders (loads directly from HuggingFace)
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        datasets=config.data.datasets,
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
        resume_from=config.training.resume_from,
        weights_from=config.training.weights_from,
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
