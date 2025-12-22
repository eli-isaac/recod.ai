"""
Configuration for dataset creation.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import load_config


@dataclass
class SourceConfig:
    """Source dataset configuration."""
    dataset_id: str
    split: str = "train"
    image_column: str = "image"


@dataclass
class OutputConfig:
    """Output dataset configuration."""
    dataset_id: str
    private: bool = False


@dataclass
class StorageConfig:
    """Local storage paths."""
    download_dir: str = "data/raw"
    output_dir: str = "data/processed"

    @property
    def download_path(self) -> Path:
        return Path(self.download_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def images_path(self) -> Path:
        return self.output_path / "images"

    @property
    def masks_path(self) -> Path:
        return self.output_path / "masks"


@dataclass
class SegmentationConfig:
    """SAM3 segmentation settings."""
    model_id: str = "facebook/sam3"
    threshold: float = 0.5
    mask_threshold: float = 0.4
    prompt: str = "distinct object"
    min_mask_area_ratio: float = 0.005
    max_mask_area_ratio: float = 0.2


@dataclass
class VariationConfig:
    """A forgery variation."""
    name: str
    num_copies: list[int]


@dataclass
class ForgeryConfig:
    """Forgery creation settings."""
    variations_per_image: int = 5  # How many variations to sample per image
    prevent_overlap: bool = True
    max_placement_attempts: int = 1000
    include_authentic: bool = True  # Also save authentic (unmodified) version
    variations: list[VariationConfig] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Complete dataset creation configuration."""
    source: SourceConfig
    output: OutputConfig
    storage: StorageConfig
    segmentation: SegmentationConfig
    forgery: ForgeryConfig
    batch_size: int = 32
    seed: int = 42

    @classmethod
    def from_yaml(cls, config_path: str) -> "DatasetConfig":
        """Load configuration from YAML file."""
        raw = load_config(config_path)

        # Parse variations
        forgery = raw.get("forgery", {})
        variations = [
            VariationConfig(name=v["name"], num_copies=v["num_copies"])
            for v in forgery.get("variations", [])
        ]

        return cls(
            source=SourceConfig(**raw.get("source", {})),
            output=OutputConfig(**raw.get("output", {})),
            storage=StorageConfig(**raw.get("storage", {})),
            segmentation=SegmentationConfig(**raw.get("segmentation", {})),
            forgery=ForgeryConfig(
                variations_per_image=forgery.get("variations_per_image", 5),
                prevent_overlap=forgery.get("prevent_overlap", True),
                max_placement_attempts=forgery.get("max_placement_attempts", 1000),
                include_authentic=forgery.get("include_authentic", True),
                variations=variations,
            ),
            batch_size=raw.get("processing", {}).get("batch_size", 32),
            seed=raw.get("processing", {}).get("seed", 42),
        )
