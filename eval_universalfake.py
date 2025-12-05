"""
Simple UniversalFakeDetect evaluation script for authentic/forged classification.
This model is specifically designed for detecting AI-generated images.
"""
import sys
import os

# Add UniversalFakeDetect to path
UFD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UniversalFakeDetect')
sys.path.insert(0, UFD_DIR)

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from models import get_model

# CLIP normalization
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]

def load_model(weights_path):
    """Load UniversalFakeDetect model with CLIP ViT-L/14 backbone"""
    model = get_model('CLIP:ViT-L/14')
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.fc.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model

def get_transform():
    return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

def predict_score(model, image_path, transform, device):
    """Get fake probability score (0=real, 1=fake)"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(img).sigmoid().item()
        return score
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'UniversalFakeDetect', 'pretrained_weights', 'fc_weights.pth')
    authentic_dir = os.path.join(script_dir, 'data', 'train_images', 'authentic')
    forged_dir = os.path.join(script_dir, 'data', 'train_images', 'forged')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading UniversalFakeDetect model (CLIP ViT-L/14)...")
    model = load_model(weights_path)
    transform = get_transform()
    print("Model loaded!")
    
    # Get image lists
    authentic_images = glob(os.path.join(authentic_dir, '*'))
    forged_images = glob(os.path.join(forged_dir, '*'))
    
    print(f"\nAuthentic images: {len(authentic_images)}")
    print(f"Forged images: {len(forged_images)}")
    
    # Evaluate authentic images (should predict low scores)
    print("\n--- Evaluating AUTHENTIC images ---")
    authentic_correct = 0
    authentic_total = 0
    authentic_scores = []
    
    for img_path in tqdm(authentic_images, desc="Authentic"):
        score = predict_score(model, img_path, transform, device)
        if score is not None:
            authentic_scores.append(score)
            authentic_total += 1
            if score < 0.5:  # Correctly classified as authentic/real
                authentic_correct += 1
    
    # Evaluate forged images (should predict high scores)
    print("\n--- Evaluating FORGED images ---")
    forged_correct = 0
    forged_total = 0
    forged_scores = []
    
    for img_path in tqdm(forged_images, desc="Forged"):
        score = predict_score(model, img_path, transform, device)
        if score is not None:
            forged_scores.append(score)
            forged_total += 1
            if score >= 0.5:  # Correctly classified as fake
                forged_correct += 1
    
    # Results
    print("\n" + "="*50)
    print("RESULTS - UniversalFakeDetect")
    print("="*50)
    
    auth_acc = authentic_correct / authentic_total * 100 if authentic_total > 0 else 0
    forge_acc = forged_correct / forged_total * 100 if forged_total > 0 else 0
    total_correct = authentic_correct + forged_correct
    total_images = authentic_total + forged_total
    overall_acc = total_correct / total_images * 100 if total_images > 0 else 0
    
    print(f"\nAuthentic: {authentic_correct}/{authentic_total} correct ({auth_acc:.2f}%)")
    print(f"  Mean score: {np.mean(authentic_scores):.4f} (lower is better)")
    
    print(f"\nForged: {forged_correct}/{forged_total} correct ({forge_acc:.2f}%)")
    print(f"  Mean score: {np.mean(forged_scores):.4f} (higher is better)")
    
    print(f"\nOverall: {total_correct}/{total_images} correct ({overall_acc:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()


