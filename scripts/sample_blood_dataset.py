#!/usr/bin/env python3
"""
Temporary script to sample 1000 random images from ehottl/blood_dataset
and upload as a new dataset called recod_pretrain on HuggingFace.
"""

import random
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# Configuration
SOURCE_DATASET = "ehottl/blood_dataset"
TARGET_DATASET = "recod_pretrain"  # Will be uploaded to your HF account
NUM_SAMPLES = 1000
SEED = 42


def main():
    print(f"Loading dataset from {SOURCE_DATASET}...")
    dataset = load_dataset(SOURCE_DATASET, split="train")
    
    print(f"Total samples in source dataset: {len(dataset)}")
    
    # Set seed for reproducibility
    random.seed(SEED)
    
    # Sample random indices
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    print(f"Sampled {len(indices)} random indices")
    
    # Select the samples
    sampled_dataset = dataset.select(indices)
    print(f"Created sampled dataset with {len(sampled_dataset)} samples")
    
    # Show some stats about the sampled data
    if "label" in sampled_dataset.features:
        labels = sampled_dataset["label"]
        unique_labels = set(labels)
        print(f"Label distribution in sample:")
        for label in sorted(unique_labels):
            count = labels.count(label)
            print(f"  {label}: {count}")
    
    # Push to HuggingFace Hub
    print(f"\nPushing to HuggingFace Hub as {TARGET_DATASET}...")
    print("(Make sure you're logged in with `huggingface-cli login`)")
    
    sampled_dataset.push_to_hub(TARGET_DATASET, private=False)
    
    print(f"\nDone! Dataset uploaded to: https://huggingface.co/datasets/<your-username>/{TARGET_DATASET}")


if __name__ == "__main__":
    main()
