# Scientific Image Forgery Detection

A DINOv2-based segmentation model for detecting manipulated regions in scientific images.

## Overview

This project implements pixel-level forgery localization using DINOv2 vision transformer features combined with a progressive upsampling decoder. The model is trained to identify copy-move and other manipulation artifacts in scientific figures.

## Project Structure

```
├── configs/              # YAML configuration files
├── data/                 # Local data directory
├── scripts/              # Training and utility scripts
├── src/
│   ├── models/           # DINOv2 segmenter architecture
│   ├── data/             # Dataset loading and transforms
│   ├── training/         # Training loop and losses
│   └── utils/            # Utilities
├── notebooks/            # Jupyter notebooks
├── checkpoints/          # Saved model weights
└── outputs/              # Logs and predictions
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/recod.ai.git
cd recod.ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --config configs/train_config.yaml
```

Training data is automatically downloaded from Hugging Face on first run.

### Configuration

Key parameters in `configs/train_config.yaml`:

```yaml
model:
  backbone: "facebook/dinov2-base"
  img_size: 512
  unfreeze_blocks: 3

training:
  batch_size: 32
  learning_rate: 3.0e-4
  epochs: 40
```

## Dataset Creation

Due to limited availability of labeled forgery data, we implemented a synthetic data generation pipeline. The process:

1. Source authentic scientific images from Hugging Face
2. Segment distinct objects using SAM3 (Segment Anything Model)
3. Generate copy-move forgeries by pasting segmented objects at random locations
4. Create corresponding ground truth masks for each manipulation

This allows generating diverse training samples with precise pixel-level annotations. Configuration supports multiple copy variations (1-4 copies per object, multiple objects per image) to cover different forgery scenarios.

```bash
python scripts/create_dataset.py --config configs/dataset_config.yaml
```

## Model Architecture

- **Encoder**: DINOv2-base with last 3 transformer blocks fine-tuned
- **Decoder**: 4-stage progressive upsampling with batch normalization
- **Output**: Multi-channel segmentation masks with Hungarian matching loss

## License

MIT

## Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- RECOD.AI LUC Scientific Image Forgery Detection Competition
