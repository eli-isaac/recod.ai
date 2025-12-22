"""
Forgery creation pipeline.

Processes images in batches:
1. Load images
2. Run SAM segmentation
3. Create forgery variations
4. Save results
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.data.config import DatasetConfig


def get_image_paths(input_dir: Path) -> list[Path]:
    """Get all image paths from directory."""
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    paths = [p for p in input_dir.iterdir() if p.suffix.lower() in extensions]
    return sorted(paths)


def load_images(paths: list[Path]) -> list[Image.Image]:
    """Load PIL images from paths."""
    return [Image.open(p).convert("RGB") for p in paths]


def run_sam_batch(
    images: list[Image.Image],
    model: Any,
    processor: Any,
    prompt: str = "distinct object",
    threshold: float = 0.5,
    mask_threshold: float = 0.4,
) -> list[dict]:
    """
    Run SAM on a batch of images.

    Returns list of dicts with 'masks', 'scores' for each image.
    """
    device = next(model.parameters()).device

    inputs = processor(
        images=images,
        text=[prompt] * len(images),
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[(img.height, img.width) for img in images],
        )

    # Convert to numpy
    batch_results = []
    for r in results:
        batch_results.append({
            "masks": r["masks"].cpu().numpy() if torch.is_tensor(r["masks"]) else r["masks"],
            "scores": r["scores"].cpu().numpy() if torch.is_tensor(r["scores"]) else r["scores"],
        })

    # Cleanup
    del inputs, outputs, results
    torch.cuda.empty_cache()

    return batch_results


def get_candidate_masks(
    masks: np.ndarray,
    image_size: int,
    min_ratio: float = 0.005,
    max_ratio: float = 0.2,
) -> list[np.ndarray]:
    """Filter masks by area ratio."""
    candidates = []
    for mask in masks:
        mask_area = np.sum(mask)
        ratio = mask_area / image_size
        if min_ratio < ratio < max_ratio:
            candidates.append(mask)
    return candidates


def copy_paste_objects(
    image: Image.Image,
    masks: list[np.ndarray],
    num_copies_list: list[int],
    prevent_overlap: bool = True,
    max_attempts: int = 1000,
) -> tuple[Image.Image, np.ndarray]:
    """
    Copy-paste objects to create forgery.

    Args:
        image: Original PIL image
        masks: List of binary masks for objects to copy
        num_copies_list: Number of copies for each object
        prevent_overlap: If True, pasted objects won't overlap
        max_attempts: Max placement attempts per copy

    Returns:
        (forged_image, mask) where mask has shape (N, H, W) with one channel per object.
        Each channel includes BOTH the source location AND all pasted copies.
        If no copies were placed, mask will have shape (0, H, W).
    """
    img = np.array(image)
    h, w = img.shape[:2]
    occupied = np.zeros((h, w), dtype=bool)
    result = img.copy()
    object_masks = []  # One mask per object

    # Mark original mask locations as occupied
    if prevent_overlap:
        for mask in masks:
            occupied[mask.astype(bool)] = True

    for mask, num_copies in zip(masks, num_copies_list):
        mask_bool = mask.astype(bool)
        # Start with source location included in mask
        obj_mask = mask_bool.copy()
        copies_placed = 0

        # Get bounding box
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        if not np.any(rows) or not np.any(cols):
            continue

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        obj_h, obj_w = y_max - y_min + 1, x_max - x_min + 1

        # Extract object
        object_crop = img[y_min:y_max + 1, x_min:x_max + 1]
        mask_crop = mask_bool[y_min:y_max + 1, x_min:x_max + 1]

        for _ in range(num_copies):
            for _ in range(max_attempts):
                # Random position
                offset_x = random.randint(0, max(0, w - obj_w))
                offset_y = random.randint(0, max(0, h - obj_h))

                # Check overlap
                paste_region = occupied[offset_y:offset_y + obj_h, offset_x:offset_x + obj_w]
                if prevent_overlap and np.any(paste_region[mask_crop]):
                    continue

                # Paste object
                result[offset_y:offset_y + obj_h, offset_x:offset_x + obj_w][mask_crop] = object_crop[mask_crop]

                # Update this object's mask (add pasted location)
                obj_mask[offset_y:offset_y + obj_h, offset_x:offset_x + obj_w][mask_crop] = True
                if prevent_overlap:
                    occupied[offset_y:offset_y + obj_h, offset_x:offset_x + obj_w][mask_crop] = True
                copies_placed += 1
                break

        # Only include mask if at least one copy was placed
        if copies_placed > 0:
            object_masks.append(obj_mask)

    # Stack into (N, H, W) array
    mask_3d = np.stack(object_masks) if object_masks else np.zeros((0, h, w), dtype=bool)

    return Image.fromarray(result), mask_3d


def process_batch(
    paths: list[Path],
    model: Any,
    processor: Any,
    config: DatasetConfig,
) -> dict:
    """
    Process a batch of images end-to-end.

    Returns dict with counts of forgeries created and errors.
    """
    seg_config = config.segmentation
    forgery_config = config.forgery
    storage = config.storage

    # Ensure output dirs exist
    storage.images_path.mkdir(parents=True, exist_ok=True)
    storage.masks_path.mkdir(parents=True, exist_ok=True)

    # Load images
    images = load_images(paths)

    # Run SAM
    sam_results = run_sam_batch(
        images=images,
        model=model,
        processor=processor,
        prompt=seg_config.prompt,
        threshold=seg_config.threshold,
        mask_threshold=seg_config.mask_threshold,
    )

    forgeries_created = 0
    errors = 0

    authentics_created = 0

    # Process each image
    for path, image, sam_result in zip(paths, images, sam_results):
        try:
            stem = path.stem

            # Save authentic version if enabled (no mask file)
            if forgery_config.include_authentic:
                image.save(storage.images_path / f"{stem}.png")
                authentics_created += 1

            image_size = image.width * image.height

            # Get candidate masks
            candidates = get_candidate_masks(
                masks=sam_result["masks"],
                image_size=image_size,
                min_ratio=seg_config.min_mask_area_ratio,
                max_ratio=seg_config.max_mask_area_ratio,
            )

            if not candidates:
                continue

            # Sample N variations randomly (without replacement)
            n = min(forgery_config.variations_per_image, len(forgery_config.variations))
            sampled_variations = random.sample(forgery_config.variations, n)

            for variation in sampled_variations:
                # Select random masks (up to number needed)
                k = min(len(variation.num_copies), len(candidates))
                selected_masks = random.sample(candidates, k)

                # Create forgery
                forged_image, mask = copy_paste_objects(
                    image=image,
                    masks=selected_masks,
                    num_copies_list=variation.num_copies[:k],
                    prevent_overlap=forgery_config.prevent_overlap,
                    max_attempts=forgery_config.max_placement_attempts,
                )

                # Skip if no copies were placed (mask is empty)
                if mask.shape[0] == 0:
                    continue

                # Save forged image with mask
                forged_image.save(storage.images_path / f"{stem}_{variation.name}.png")
                np.save(storage.masks_path / f"{stem}_{variation.name}.npy", mask)
                forgeries_created += 1

        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            errors += 1

    return {"forgeries": forgeries_created, "authentics": authentics_created, "errors": errors}


def run_pipeline(
    config: DatasetConfig,
    model: Any,
    processor: Any,
) -> dict:
    """
    Run the full forgery creation pipeline.

    Processes images in batches from download_dir, saves to output_dir.
    """
    random.seed(config.seed)

    image_paths = get_image_paths(config.storage.download_path)
    print(f"Found {len(image_paths)} images")

    total_forgeries = 0
    total_authentics = 0
    total_errors = 0

    for i in range(0, len(image_paths), config.batch_size):
        batch_paths = image_paths[i:i + config.batch_size]
        batch_num = i // config.batch_size + 1
        total_batches = (len(image_paths) + config.batch_size - 1) // config.batch_size

        print(f"Batch {batch_num}/{total_batches} ({len(batch_paths)} images)")

        result = process_batch(batch_paths, model, processor, config)
        total_forgeries += result["forgeries"]
        total_authentics += result["authentics"]
        total_errors += result["errors"]

    return {
        "total_forgeries": total_forgeries,
        "total_authentics": total_authentics,
        "total_errors": total_errors,
    }
