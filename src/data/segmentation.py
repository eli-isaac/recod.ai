"""
SAM3 (Segment Anything Model 3) segmentation utilities.
"""

from typing import Tuple

from transformers import Sam3Model, Sam3Processor

from src.utils import get_device


def load_sam3_model(
    model_id: str = "facebook/sam3",
) -> Tuple[Sam3Model, Sam3Processor]:
    """
    Load SAM3 model and processor from HuggingFace.

    Args:
        model_id: HuggingFace model ID (default: "facebook/sam3")

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading SAM3 model: {model_id}")

    # Determine device
    device = get_device()

    print(f"Using device: {device}")

    # Load processor and model
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    print("SAM3 model loaded successfully")
    return model, processor
