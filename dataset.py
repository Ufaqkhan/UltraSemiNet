import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import random
from transform_utils import weak_augment, strong_intensity_augment

class FBUIDataset(Dataset):
    def __init__(self, root_dir, split='train', fold=0, labeled_ratio=None, transform=None, mode='labeled', augment=True):
        self.root_dir = root_dir
        self.mode = mode
        self.split = split
        self.augment = augment
        
        # Load all image paths
        image_dir = os.path.join(root_dir, 'Images')
        mask_dir = os.path.join(root_dir, 'Masks')
        
        all_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(all_images) == 0:
            raise ValueError(f"No images found in {image_dir}")
            
        subject_map = {}
        for img_path in all_images:
            basename = os.path.basename(img_path)
            subject_id = basename.split('_')[0]
            if subject_id not in subject_map:
                subject_map[subject_id] = []
            subject_map[subject_id].append(img_path)
            
        unique_subjects = sorted(list(subject_map.keys()))
        
        # Split subjects: 5-Fold
        random.seed(42)
        random.shuffle(unique_subjects)
        
        n_subjects = len(unique_subjects)
        n_subjects = len(unique_subjects)
        fold_size = n_subjects // 5
        
        # Test Fold (Current Fold)
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < 4 else n_subjects
        test_subjects = unique_subjects[test_start:test_end]
        
        # Validation Fold (Next Fold)
        val_fold = (fold + 1) % 5
        val_start = val_fold * fold_size
        val_end = (val_fold + 1) * fold_size if val_fold < 4 else n_subjects
        
        # Handle wrap-around case for last fold
        if val_fold == 4: val_end = n_subjects
        
        val_subjects = unique_subjects[val_start:val_end]
        
        # Train Fold (Remaining)
        train_subjects = [s for s in unique_subjects if s not in test_subjects and s not in val_subjects]
        
        if split == 'test':
            self.subjects = test_subjects
        elif split == 'val':
            self.subjects = val_subjects
        else: # train
            self.subjects = train_subjects
        

            
        self.images = []
        for s in self.subjects:
            self.images.extend(subject_map[s])
            
        if split == 'train':
            random.seed(42 + fold)
            random.shuffle(self.images)
            n_labeled = 1500
            if len(self.images) <= n_labeled:
                print(f"Warning: Dataset size ({len(self.images)}) <= requested labeled. Using 50% split.")
                n_labeled = int(len(self.images) * 0.5)
            
            if mode == 'labeled':
                self.images = self.images[:n_labeled]
            elif mode == 'unlabeled':
                self.images = self.images[n_labeled:]
            
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        basename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, basename)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        
        # To Tensor (1, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        sample = {'image': image, 'label': mask}
        
        # Apply transforms if augment is True AND split is train
        if self.augment and self.split == 'train':
            # Apply Weak
            sample = weak_augment(sample)
            
            # Create Strong
            img_s = sample['image'].clone()
            sample_s = {'image': img_s, 'label': sample['label']}
            sample_s = strong_intensity_augment(sample_s)
            
            sample['image_strong'] = sample_s['image']
            
        return sample
