"""
Training loop for forgery detection model.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from src.training.config import TrainConfig
from src.training.losses import hungarian_matching, bce_loss, combined_loss


def compute_pos_weight(dataloader: DataLoader) -> float:
    """
    Compute positive class weight from dataset statistics.
    
    Scans the dataset once to count positive vs negative pixels,
    then returns neg_count / pos_count for use in BCE loss.
    
    Args:
        dataloader: Training data loader
        
    Returns:
        Positive class weight (ratio of negative to positive pixels)
    """
    total_pos = 0
    total_neg = 0
    
    print("Computing pos_weight from dataset...")
    for _, masks in tqdm(dataloader, desc="Scanning masks"):
        total_pos += masks.sum().item()
        total_neg += (1 - masks).sum().item()
    
    pos_weight = total_neg / (total_pos + 1e-8)
    print(f"  → Dataset stats: {total_pos:.0f} positive pixels, {total_neg:.0f} negative pixels")
    print(f"  → Computed pos_weight: {pos_weight:.2f}")
    
    return pos_weight


def compute_f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for binary segmentation with Hungarian matching.
    
    Uses Hungarian matching to find optimal channel assignment before computing metrics.
    This handles the case where channel ordering doesn't matter.
    
    Args:
        outputs: Predicted logits, shape (B, C, H, W)
        targets: Ground truth masks, shape (B, C, H, W)
        threshold: Threshold for converting logits to binary predictions
        
    Returns:
        (precision, recall, f1)
    """
    # Apply Hungarian matching to align channels optimally
    matched_outputs, matched_targets = hungarian_matching(outputs, targets)
    
    preds = (torch.sigmoid(matched_outputs) > threshold).float()
    
    # Flatten everything
    preds_flat = preds.view(-1)
    targets_flat = matched_targets.view(-1)
    
    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1


class Trainer:
    """Trainer for forgery segmentation model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup optimizer with different LRs for backbone/decoder
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss function - compute pos_weight from data if not specified
        if config.training.pos_weight is None:
            self.pos_weight = compute_pos_weight(train_loader)
        else:
            self.pos_weight = config.training.pos_weight
            print(f"Using configured pos_weight: {self.pos_weight}")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with parameter groups."""
        cfg = self.config.training
        
        # Get parameter groups from model
        if hasattr(self.model, 'get_param_groups'):
            param_groups = self.model.get_param_groups(cfg.backbone_lr_scale)
            for group in param_groups:
                group['lr'] = cfg.learning_rate * group.pop('lr_scale')
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': cfg.learning_rate}]
        
        return optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler | None:
        """Setup learning rate scheduler."""
        cfg = self.config.training.scheduler
        
        if cfg.type == "cosine":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cfg.warmup_epochs,
                eta_min=cfg.min_lr,
            )
        elif cfg.type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )
        else:
            return None
    
    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs}",
        )
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Apply Hungarian matching, then compute loss on matched pairs
            matched_outputs, matched_masks = hungarian_matching(outputs, masks)
            loss = combined_loss(matched_outputs, matched_masks, self.pos_weight)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """
        Run validation.
        
        Returns:
            (avg_loss, f1_score)
        """
        self.model.eval()
        total_loss = 0.0
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        
        for images, masks in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            
            # Apply Hungarian matching, then compute loss and F1 on matched pairs
            matched_outputs, matched_masks = hungarian_matching(outputs, masks)
            loss = bce_loss(matched_outputs, matched_masks, self.pos_weight)
            total_loss += loss.item()
            
            # Accumulate F1 stats using matched predictions/targets
            preds = (torch.sigmoid(matched_outputs) > 0.5).float()
            total_tp += (preds * matched_masks).sum().item()
            total_fp += (preds * (1 - matched_masks)).sum().item()
            total_fn += ((1 - preds) * matched_masks).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute F1 from accumulated stats
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return avg_loss, f1
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load full checkpoint (model, optimizer, scheduler, training state)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_f1_scores = checkpoint.get('val_f1_scores', [])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def load_weights(self, checkpoint_path: str) -> None:
        """Load model weights only (for finetuning with new config/LR)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Don't load optimizer/scheduler state - use new config's settings
        # Don't restore epoch counter - start fresh
    
    @torch.no_grad()
    def generate_samples(self, n_samples: int = 10) -> None:
        """Generate sample predictions from validation set."""
        self.model.eval()
        epoch = self.current_epoch + 1
        
        # Setup samples directory
        samples_dir = self.checkpoint_dir / "samples"
        images_dir = samples_dir / "images"
        pred_dir = samples_dir / "pred"
        gt_dir = samples_dir / "gt"
        images_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Get random indices from validation set
        val_dataset = self.val_loader.dataset
        indices = random.sample(range(len(val_dataset)), min(n_samples, len(val_dataset)))
        
        for i, idx in enumerate(indices):
            img_tensor, gt_mask = val_dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Get prediction
            pred_logits = self.model(img_tensor)
            pred_mask = torch.sigmoid(pred_logits).squeeze(0).cpu().numpy()
            
            # Save image (convert from tensor to PIL)
            img_np = (img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(images_dir / f"epoch{epoch:03d}_sample{i:02d}.jpg", quality=95)
            
            # Save predicted mask
            np.save(pred_dir / f"epoch{epoch:03d}_sample{i:02d}.npy", pred_mask)
            
            # Save ground truth mask
            np.save(gt_dir / f"epoch{epoch:03d}_sample{i:02d}.npy", gt_mask.numpy())
        
        print(f"  → Saved {len(indices)} samples to {samples_dir}")
    
    def train(self) -> dict:
        """Run full training loop."""
        metric = self.config.training.best_model_metric
        print(f"Starting training for {self.config.training.epochs} epochs...")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Best model metric: {metric}")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"  → Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            print(f"  → Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Check for improvement based on configured metric
            if metric == "f1":
                is_best = val_f1 > self.best_val_f1
                if is_best:
                    self.best_val_f1 = val_f1
            else:  # val_loss
                is_best = val_loss < self.best_val_loss
            
            # Always track best loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            if is_best:
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print(f"  → New best model! ({metric})")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Generate samples
            if (epoch + 1) % self.config.training.sample_every == 0:
                self.generate_samples(n_samples=10)
                
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'epochs_trained': self.current_epoch + 1,
        }


def train_model(
    model: nn.Module,
    config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    resume_from: str | None = None,
    weights_from: str | None = None,
) -> dict:
    """
    Train a forgery detection model.
    
    Args:
        model: Model to train
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        resume_from: Optional checkpoint path to resume from (loads full state)
        weights_from: Optional checkpoint path to load weights only (for finetuning)
        
    Returns:
        Training results dict
    """
    trainer = Trainer(model, config, train_loader, val_loader, device)
    
    if resume_from:
        print(f"Resuming from {resume_from}")
        trainer.load_checkpoint(resume_from)
    elif weights_from:
        print(f"Loading weights from {weights_from} (finetuning mode)")
        trainer.load_weights(weights_from)
    
    return trainer.train()