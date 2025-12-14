"""
Typed configuration dataclasses for dataset creation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    """Local storage configuration."""

    data_dir: str = "data/"
    download_dir: str = "data/raw"
    output_dir: str = "data/processed"
    images_subdir: str = "images"
    masks_subdir: str = "masks"

    @property
    def download_path(self) -> Path:
        return Path(self.download_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def images_path(self) -> Path:
        return self.output_path / self.images_subdir

    @property
    def masks_path(self) -> Path:
        return self.output_path / self.masks_subdir


@dataclass
class SegmentationConfig:
    """SAM3 segmentation configuration."""

    model_id: str = "facebook/sam3"
    batch_size: int = 16
    threshold: float = 0.5
    mask_threshold: float = 0.4
    prompt: str = "distinct object"
    min_mask_area_ratio: float = 0.005
    max_mask_area_ratio: float = 0.2
    max_image_size_bytes: Optional[int] = 2000000


@dataclass
class VariationConfig:
    """A single forgery variation."""

    name: str
    num_copies: list[int]


@dataclass
class ForgeryConfig:
    """Forgery creation configuration."""

    variations_per_image: int = 5
    prevent_overlap: bool = True
    max_placement_attempts: int = 1000
    variations: list[VariationConfig] = field(default_factory=list)


@dataclass
class UploadConfig:
    """Upload configuration."""

    chunk_size: int = 500
    ignore_patterns: list[str] = field(
        default_factory=lambda: ["*.ipynb_checkpoints", "__pycache__", "*.pyc"]
    )


@dataclass
class ProcessingConfig:
    """Processing configuration."""

    seed: int = 42
    num_workers: int = 4
    process_batch_size: int = 32  # Batch size for end-to-end processing


@dataclass
class DatasetConfig:
    """Complete dataset creation configuration."""

    source: SourceConfig
    output: OutputConfig
    storage: StorageConfig
    segmentation: SegmentationConfig
    forgery: ForgeryConfig
    upload: UploadConfig
    processing: ProcessingConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "DatasetConfig":
        """Load configuration from YAML file."""
        raw_config = load_config(config_path)
        return cls.from_dict(raw_config)

    @classmethod
    def from_dict(cls, config: dict) -> "DatasetConfig":
        """Create config from dictionary. Missing values use dataclass defaults."""

        # Helper to extract section with defaults from dataclass
        def get_section(name: str) -> dict:
            return config.get(name, {})

        source = get_section("source")
        output = get_section("output")
        storage = get_section("storage")
        segmentation = get_section("segmentation")
        forgery = get_section("forgery")
        upload = get_section("upload")
        processing = get_section("processing")

        # Parse variations
        variations = [
            VariationConfig(name=v["name"], num_copies=v["num_copies"])
            for v in forgery.get("variations", [])
        ]

        return cls(
            source=SourceConfig(
                dataset_id=source["dataset_id"],
                **{k: v for k, v in source.items() if k != "dataset_id"},
            ),
            output=OutputConfig(
                dataset_id=output["dataset_id"],
                **{k: v for k, v in output.items() if k != "dataset_id"},
            ),
            storage=StorageConfig(**storage),
            segmentation=SegmentationConfig(**segmentation),
            forgery=ForgeryConfig(
                **{k: v for k, v in forgery.items() if k != "variations"},
                variations=variations,
            ),
            upload=UploadConfig(**upload),
            processing=ProcessingConfig(**processing),
        )
