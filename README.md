# Scientific Image Forgery Detection with DINOv2

A deep learning approach for detecting manipulated regions in scientific images using DINOv2 vision transformers.

## 🎯 Overview

This project implements a forgery detection model that identifies manipulated regions in scientific figures. The approach leverages DINOv2 pre-trained features combined with a custom decoder for pixel-level forgery localization.

## 📊 Results

| Model | Validation Score | Leaderboard |
|-------|-----------------|-------------|
| DINOv2 + Decoder | TBD | TBD |

## 🏗️ Project Structure

```
├── configs/              # Configuration files (YAML)
├── data/                 # Local data directory (gitignored)
├── scripts/              # Executable scripts
│   ├── download_data.py  # Download dataset
│   ├── create_dataset.py # Dataset preprocessing
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── src/                  # Core source code
│   ├── models/           # Model architectures
│   ├── data/             # Dataset classes & transforms
│   ├── training/         # Training loop & callbacks
│   └── utils/            # Utilities
├── notebooks/            # Jupyter notebooks
├── checkpoints/          # Saved model weights (gitignored)
└── outputs/              # Logs, predictions (gitignored)
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/forgery-detection.git
cd forgery-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Download Data

```bash
python scripts/download_data.py
```

### Training

```bash
python scripts/train.py --config configs/train_config.yaml
```

### Inference

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## 📁 Dataset

The dataset is hosted on [Hugging Face Datasets](https://huggingface.co/datasets/YOUR_USERNAME/forgery-detection).

## 🔧 Configuration

Training parameters can be modified in `configs/train_config.yaml`:

```yaml
model:
  backbone: "facebook/dinov2-base"
  img_size: 512

training:
  batch_size: 8
  learning_rate: 1e-4
  epochs: 50
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- RECOD.AI LUC Scientific Image Forgery Detection Competition
