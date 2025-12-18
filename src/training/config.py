"""
Training configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import load_config


@dataclass
class ModelConfig:
    """Model configuration."""
    backbone: str = "facebook/dinov2-base"
    out_channels: int = 4
    unfreeze_blocks: int = 3
    decoder_dropout: float = 0.1


@dataclass
class DataConfig:
    """Data configuration."""
    # Local directory for data storage
    local_dir: str = "data/pretrain"
    
    # HuggingFace dataset IDs (downloaded to local_dir if it's empty)
    datasets: list[str] = field(default_factory=list)
    
    img_size: int = 512
    num_workers: int = 4
    val_split: float = 0.2


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 8
    learning_rate: float = 1e-4
    backbone_lr_scale: float = 0.1  # Backbone gets LR * this scale
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stopping_patience: int = 10
    pos_weight: float = 99.0  # Positive class weight for BCE loss
    
    # Best model selection: "f1" or "val_loss"
    best_model_metric: str = "f1"
    
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Checkpointing
    save_every: int = 5
    sample_every: int = 5  # Generate sample predictions every N epochs
    checkpoint_dir: str = "checkpoints"
    
    # Resume/finetune from checkpoint
    resume_from: str | None = None   # Load full state (continue training)
    weights_from: str | None = None  # Load weights only (finetune with new config)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "outputs/logs"
    wandb_project: str | None = None
    log_every_n_steps: int = 10


@dataclass
class TrainConfig:
    """Complete training configuration."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainConfig":
        """Load configuration from YAML file."""
        raw = load_config(config_path)
        
        # Parse scheduler config
        scheduler_raw = raw.get("training", {}).get("scheduler", {})
        scheduler = SchedulerConfig(
            type=scheduler_raw.get("type", "cosine"),
            warmup_epochs=scheduler_raw.get("warmup_epochs", 5),
            min_lr=scheduler_raw.get("min_lr", 1e-6),
        )
        
        # Parse model config
        model_raw = raw.get("model", {})
        model = ModelConfig(
            backbone=model_raw.get("backbone", "facebook/dinov2-base"),
            out_channels=model_raw.get("channels", 4),
            unfreeze_blocks=model_raw.get("unfreeze_blocks", 3),
            decoder_dropout=model_raw.get("decoder_dropout", 0.1),
        )
        
        # Parse data config
        data_raw = raw.get("data", {})
        datasets_raw = data_raw.get("datasets", [])
        # Support both list and single string
        if isinstance(datasets_raw, str):
            datasets_raw = [datasets_raw]
        data = DataConfig(
            local_dir=data_raw.get("local_dir", "data/pretrain"),
            datasets=datasets_raw,
            img_size=model_raw.get("img_size", 512),
            num_workers=data_raw.get("num_workers", 4),
            val_split=data_raw.get("val_split", 0.2),
        )
        
        # Parse training config
        training_raw = raw.get("training", {})
        training = TrainingConfig(
            batch_size=training_raw.get("batch_size", 8),
            learning_rate=training_raw.get("learning_rate", 1e-4),
            backbone_lr_scale=training_raw.get("backbone_lr_scale", 0.1),
            weight_decay=training_raw.get("weight_decay", 1e-5),
            epochs=training_raw.get("epochs", 50),
            early_stopping_patience=training_raw.get("early_stopping_patience", 10),
            pos_weight=training_raw.get("pos_weight", 99.0),
            best_model_metric=training_raw.get("best_model_metric", "f1"),
            scheduler=scheduler,
            save_every=training_raw.get("save_every", 5),
            sample_every=training_raw.get("sample_every", 5),
            checkpoint_dir=training_raw.get("checkpoint_dir", "checkpoints"),
            resume_from=training_raw.get("resume_from"),
            weights_from=training_raw.get("weights_from"),
        )
        
        # Parse logging config
        logging_raw = raw.get("logging", {})
        logging = LoggingConfig(
            log_dir=logging_raw.get("log_dir", "outputs/logs"),
            wandb_project=logging_raw.get("wandb_project"),
            log_every_n_steps=logging_raw.get("log_every_n_steps", 10),
        )
        
        return cls(
            model=model,
            data=data,
            training=training,
            logging=logging,
            seed=raw.get("seed", 42),
        )
