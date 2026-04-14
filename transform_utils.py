import torch
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms import functional as TF

def weak_augment(sample): # Geometric
    image, label = sample['image'], sample['label']
    # Expects Tensors (C, H, W)
    
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)
        
    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)
        
    k = random.randint(0, 3)
    if k > 0:
        image = torch.rot90(image, k, [1, 2])
        label = torch.rot90(label, k, [1, 2])

    return {'image': image, 'label': label}

def strong_intensity_augment(sample): # Color/Noise only
    image, label = sample['image'], sample['label']
    
    # ColorJitter
    if random.random() > 0.2:
        try:
            # ColorJitter needs (C, H, W) or (B, C, H, W)
            # It works on Tensor in recent versions
            jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0)
            image = jitter(image)
        except:
            pass # Fallback if fails
        
    # Speckle Noise
    if random.random() > 0.2:
        noise = torch.randn_like(image) * 0.1
        image = image + image * noise
        image = torch.clamp(image, 0, 1)
        
    return {'image': image, 'label': label}

def elastic_transform(sample):
    # Simplified Elastic Transform
    # Getting full elastic correct on tensor without scipy/cv2 is hard. 
    # Skipping to avoid breaking batching/logic complexly, 
    # or implement simplified grid sample.
    return sample

def get_tta_views(image):
    # Image: (C, H, W) or (B, C, H, W)
    views = []
    views.append((image, lambda x: x))
    views.append((TF.hflip(image), lambda x: TF.hflip(x)))
    views.append((TF.vflip(image), lambda x: TF.vflip(x)))
    views.append((torch.rot90(image, 1, [-2, -1]), lambda x: torch.rot90(x, 3, [-2, -1])))
    return views

def to_tensor(sample):
    # No-op if already tensor
    return sample
