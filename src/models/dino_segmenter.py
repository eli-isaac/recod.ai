"""
DINOv2-based segmentation model for forgery detection.

Uses DINOv2 as encoder with a progressive upsampling decoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

# ImageNet normalization (used by DINOv2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoDecoder(nn.Module):
    """Progressive upsampling decoder with regularization."""

    def __init__(
        self, in_channels: int = 768, out_channels: int = 4, dropout: float = 0.1
    ):
        super().__init__()

        # Conv blocks for each upsampling stage
        l1_kernels = 384
        l2_kernels = 192
        l3_kernels = 96
        l4_kernels = 48
        self.up1 = self._block(in_channels, l1_kernels, dropout)  # 768 → 384
        self.up2 = self._block(l1_kernels, l2_kernels, dropout)  # 384 → 192
        self.up3 = self._block(l2_kernels, l3_kernels, dropout)  # 192 → 96
        self.up4 = self._block(l3_kernels, l4_kernels, dropout)  # 96 → 48

        self.final = nn.Conv2d(l4_kernels, out_channels, kernel_size=1)

    def _block(self, in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, features: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Progressive upsampling from feature map to target size.

        Args:
            features: Feature map from encoder, shape (B, C, H, W)
            target_size: Target output size (height, width)

        Returns:
            Segmentation logits, shape (B, out_channels, target_H, target_W)
        """
        # Progressive upsampling: e.g., 37×37 → 74 → 148 → 296 → 512
        x = F.interpolate(
            features, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = self.up1(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)

        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.up4(x)

        return self.final(x)


class DinoSegmenter(nn.Module):
    """
    DINOv2-based segmentation model for forgery detection.

    Uses DINOv2 as the encoder backbone with optional fine-tuning of the
    last N transformer blocks, plus a progressive upsampling decoder.
    """

    def __init__(
        self,
        backbone: str = "facebook/dinov2-base",
        out_channels: int = 4,
        unfreeze_blocks: int = 3,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.out_channels = out_channels

        # For segmentation, we skip resize/crop and process at full resolution.
        # DINOv2 handles arbitrary sizes via position embedding interpolation.

        # Load DINOv2 encoder
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden_size = self.encoder.config.hidden_size

        self.processor = AutoImageProcessor.from_pretrained(backbone, use_fast=True)
        # Freeze all parameters first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last N transformer blocks
        num_blocks = len(self.encoder.encoder.layer)
        for i in range(num_blocks - unfreeze_blocks, num_blocks):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

        # Unfreeze the final layernorm
        for param in self.encoder.layernorm.parameters():
            param.requires_grad = True

        # Decoder head
        self.decoder = DinoDecoder(hidden_size, out_channels, decoder_dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from DINOv2 encoder.

        Args:
            x: Input images, shape (B, 3, H, W), values in [0, 1]

        Returns:
            Feature map, shape (B, C, h, w) where h, w are spatial dims
        """
        # Convert from [0, 1] to [0, 255]
        imgs = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1)
        inputs = self.processor(images=imgs, return_tensors="pt")

        # Forward through encoder
        feats = self.encoder(**inputs).last_hidden_state

        # Reshape to spatial feature map
        B, N, C = feats.shape
        # Remove CLS token and reshape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N - 1))
        fmap = fmap.reshape(B, C, s, s)

        return fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation.

        Args:
            x: Input images, shape (B, 3, H, W), values in [0, 1]

        Returns:
            Segmentation logits, shape (B, out_channels, H, W)
        """
        target_size = (x.shape[2], x.shape[3])
        fmap = self.forward_features(x)
        return self.decoder(fmap, target_size)

    def get_param_groups(self, backbone_lr_scale: float = 0.1) -> list[dict]:
        """
        Get parameter groups with different learning rates.

        Args:
            backbone_lr_scale: Scale factor for backbone LR (default 0.1 = 10x smaller)

        Returns:
            List of param group dicts for optimizer
        """
        backbone_params = []
        decoder_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if "decoder" in name:
                    decoder_params.append(param)
                else:
                    backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": decoder_params, "lr_scale": 1.0},
        ]


def load_model(
    checkpoint_path: str,
    device: str | None = None,
    backbone: str = "facebook/dinov2-base",
    out_channels: int = 4,
    unfreeze_blocks: int = 3,
) -> DinoSegmenter:
    """
    Load DinoSegmenter model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (auto-detected if None)
        backbone: HuggingFace model ID for DINOv2 backbone (used if not in checkpoint)
        out_channels: Number of output channels (used if not in checkpoint)
        unfreeze_blocks: Number of transformer blocks to unfreeze

    Returns:
        Loaded model
    """
    if device is None:
        from src.utils import get_device

        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and full checkpoint format
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # Try to get config from checkpoint
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            backbone = cfg.model.backbone
            out_channels = cfg.model.out_channels
            unfreeze_blocks = cfg.model.unfreeze_blocks
    else:
        state_dict = checkpoint

    model = DinoSegmenter(
        backbone=backbone,
        out_channels=out_channels,
        unfreeze_blocks=unfreeze_blocks,
    )
    model.load_state_dict(state_dict)

    return model.to(device)
