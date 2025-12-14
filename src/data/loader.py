"""
Dataset loading utilities for HuggingFace datasets.
"""

from pathlib import Path
from typing import Optional, Union

from datasets import DatasetDict, load_dataset as hf_load_dataset


def load_dataset(
    dataset_id: str,
    cache_dir: Union[str, Path] = "data/",
    local_override: Optional[Union[str, Path]] = None,
) -> DatasetDict:
    """
    Load dataset from HuggingFace or local path.
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "username/dataset-name")
        cache_dir: Directory to cache downloaded data
        local_override: If provided, load from this local path instead of HF
    
    Returns:
        HuggingFace DatasetDict with train/val splits
    
    Example:
        >>> dataset = load_dataset("eliplutchok/recod-forgery")
        >>> train_data = dataset["train"]
        >>> val_data = dataset["validation"]
    """
    if local_override:
        local_path = Path(local_override)
        print(f"Loading local data from: {local_path}")
        # Load from local directory using HF datasets
        return hf_load_dataset("imagefolder", data_dir=str(local_path))
    
    print(f"Loading dataset from HuggingFace: {dataset_id}")
    print(f"Cache directory: {cache_dir}")
    
    return hf_load_dataset(dataset_id, cache_dir=str(cache_dir))


def load_dataset_from_config(data_config: dict) -> DatasetDict:
    """
    Load dataset using config dict.
    
    Args:
        data_config: Dict with keys 'dataset', 'cache_dir', 'local_override'
    
    Returns:
        HuggingFace DatasetDict
    """
    return load_dataset(
        dataset_id=data_config["dataset"],
        cache_dir=data_config.get("cache_dir", "data/"),
        local_override=data_config.get("local_override"),
    )
