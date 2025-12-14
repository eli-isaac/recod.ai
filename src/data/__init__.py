"""
Data utilities for dataset creation and loading.
"""

from .download import download_dataset
from .forgery import (
    ForgedImage,
    Variation,
    copy_paste_objects,
    create_forgeries,
    parse_variations,
    process_images,
    save_forgeries,
    save_forgery,
)
from .loader import load_dataset, load_dataset_from_config
from .pipeline import get_image_paths, process_batch, run_pipeline
from .segmentation import (
    Sam3Results,
    SegmentedImage,
    get_candidate_masks,
    load_sam3_model,
    run_sam3_on_batch,
)
from .upload import (
    create_zip_chunks,
    upload_chunked,
    upload_dataset,
    upload_to_huggingface,
)

__all__ = [
    # Download
    "download_dataset",
    # Loader
    "load_dataset",
    "load_dataset_from_config",
    # Segmentation
    "Sam3Results",
    "SegmentedImage",
    "load_sam3_model",
    "run_sam3_on_batch",
    "get_candidate_masks",
    # Forgery
    "Variation",
    "ForgedImage",
    "copy_paste_objects",
    "create_forgeries",
    "save_forgeries",
    "save_forgery",
    "process_images",
    "parse_variations",
    # Pipeline
    "get_image_paths",
    "run_pipeline",
    "process_batch",
    # Upload
    "create_zip_chunks",
    "upload_to_huggingface",
    "upload_chunked",
    "upload_dataset",
]
