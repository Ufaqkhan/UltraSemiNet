
import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from model import UNet
from metrics import compute_surface_metrics

import cv2

class HC18TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Find all images ending with _HC.png
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*_HC.png")))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.basename(img_path)
        annot_name = base_name.replace("_HC.png", "_HC_Annotation.png")
        annot_path = os.path.join(self.root_dir, annot_name)
        
        # Load Image (Grayscale)
        image = Image.open(img_path).convert('L')
        
        # Load Annotation (Grayscale)
        if os.path.exists(annot_path):
            mask = Image.open(annot_path).convert('L')
        else:
            print(f"Warning: Annotation not found for {base_name}")
            mask = Image.new('L', image.size, 0)
        
        image = self.transform(image)
        
        # Mask Processing: Fill BEFORE Resize to preserve thin boundaries
        mask_np = np.array(mask) # Full resolution
        mask_np = (mask_np > 0).astype(np.uint8) * 255
        
        # Fill the mask using Convex Hull to handle potential gaps
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask_filled = np.zeros_like(mask_np)
            # Assume the largest contour is the head
            largest_contour = max(contours, key=cv2.contourArea)
            # Use Convex Hull to ensure it's closed and filled
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(mask_filled, [hull], -1, 255, thickness=cv2.FILLED)
            mask_np = mask_filled
            
        # Resize mask to 224x224
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.resize((224, 224), Image.NEAREST)
        mask_np = np.array(mask_pil)
        
        mask_np = (mask_np > 0).astype(np.float32)
        mask = torch.from_numpy(mask_np).unsqueeze(0) # [1, H, W]

        return {'image': image, 'label': mask, 'name': base_name}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/HC18/test_set')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained UltraSemiNet model checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Loader
    test_ds = HC18TestDataset(args.root)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"Evaluting on HC18 Test Set: {len(test_ds)} images")

    # Model
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load Pixel Size Map
    import csv
    pixel_map = {}
    csv_path = os.path.join(args.root, 'test_set_pixel_size.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(args.root), 'test_set_pixel_size.csv')
        
    if os.path.exists(csv_path):
        print(f"Loading pixel sizes from {csv_path}...")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean keys just in case
                fname = row['filename'].strip()
                try:
                    ps = float(row['pixel size(mm)'])
                    pixel_map[fname] = ps
                except ValueError:
                    pass
    else:
        print("Warning: Pixel size CSV not found. Using default 1.0 spacing.")

    dice_scores = []
    asd_scores = []
    hd95_scores = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)
            name = batch['name'][0]
            
            logits = model(img)
            pred_prob = torch.sigmoid(logits)
            pred = (pred_prob > 0.5).float()
            
            # Dice
            inter = (pred * lbl).sum()
            union = pred.sum() + lbl.sum()
            dice = (2 * inter) / (union + 1e-6)
            dice_scores.append(dice.item())
            
            # Surface Metrics
            if lbl.sum() > 0 and pred.sum() > 0:
                try:
                    # Get pixel size
                    ps = pixel_map.get(name, 1.0)
                    metrics = compute_surface_metrics(pred, lbl, spacing=(ps, ps))
                    asd_scores.append(metrics['asd'])
                    hd95_scores.append(metrics['hd95'])
                except Exception as e:
                    print(f"Error computing surface metrics for {name}: {e}")
            
    print("-" * 30)
    print(f"HC18 Test Results ({len(test_ds)} samples) [Millimeters]")
    print(f"Mean DSC:  {np.mean(dice_scores):.4f}")
    print(f"Mean ASD:  {np.mean(asd_scores):.4f} mm")
    print(f"Mean HD95: {np.mean(hd95_scores):.4f} mm")
    print("-" * 30)

if __name__ == '__main__':
    main()
