"""
Simple TruFor evaluation script for authentic/forged classification.
Runs TruFor on train_images and reports accuracy.
"""
import sys
import os

# Add TruFor source to path
TRUFOR_SRC = os.path.join(os.path.dirname(__file__), 'TruFor', 'test_docker', 'src')
sys.path.insert(0, TRUFOR_SRC)

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

# Change to TruFor src dir for config loading
os.chdir(TRUFOR_SRC)

from config import _C as config
from yacs.config import CfgNode

def load_model(weights_path, device):
    """Load TruFor model"""
    # Update config
    config.defrost()
    config.merge_from_file('trufor.yaml')
    config.TEST.MODEL_FILE = weights_path
    config.freeze()
    
    # Load model
    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    model = confcmx(cfg=config)
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict_score(model, image_path, device):
    """Get forgery score for a single image (0=authentic, 1=forged)"""
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
        rgb = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0) / 256.0
        rgb = rgb.to(device)
        
        with torch.no_grad():
            pred, conf, det, npp = model(rgb)
            score = torch.sigmoid(det).item()
        return score
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'TruFor', 'test_docker', 'weights', 'trufor.pth.tar')
    authentic_dir = os.path.join(script_dir, 'data', 'train_images', 'authentic')
    forged_dir = os.path.join(script_dir, 'data', 'train_images', 'forged')
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading TruFor model...")
    model = load_model(weights_path, device)
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
        score = predict_score(model, img_path, device)
        if score is not None:
            authentic_scores.append(score)
            authentic_total += 1
            if score < 0.5:  # Correctly classified as authentic
                authentic_correct += 1
    
    # Evaluate forged images (should predict high scores)
    print("\n--- Evaluating FORGED images ---")
    forged_correct = 0
    forged_total = 0
    forged_scores = []
    
    for img_path in tqdm(forged_images, desc="Forged"):
        score = predict_score(model, img_path, device)
        if score is not None:
            forged_scores.append(score)
            forged_total += 1
            if score >= 0.5:  # Correctly classified as forged
                forged_correct += 1
    
    # Results
    print("\n" + "="*50)
    print("RESULTS")
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

