"""
PyTorch Data Loading and Preprocessing Module for Rice Variety Classification
Handles image loading with paired augmentation strategy where each original image
has 5 corresponding augmented versions at specific indices in the augmented dataset.
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from PIL import Image
import json
from torchvision.transforms import v2 # Recommended to use torchvision's v2
from config import *
from sklearn.model_selection import train_test_split


class RiceVarietyDataset(Dataset):
    """
    Custom PyTorch Dataset for rice variety classification.
    
    Implements paired augmentation strategy where:
    - Original image at index i in original dataset
    - Has augmented versions at indices: i, i+num_original, i+2*num_original, ...
    
    Example:
        Original 1_Subol_Lota: 232 images
        Augmented 1_Subol_Lota: has augmented versions at indices 1, 233, 465, 697, 929
    """
    
    def __init__(self, original_dir, augmented_dir=None, transform=None, 
                 use_augmented=True, include_file_info=False):
        """
        Initialize dataset.
        
        Args:
            original_dir: Path to original images directory (contains variety subdirectories)
            augmented_dir: Path to augmented images directory
            transform: PyTorch transforms to apply
            use_augmented: Whether to include augmented versions
            include_file_info: Whether to return file information for debugging
        """
        self.original_dir = Path(original_dir)
        self.augmented_dir = Path(augmented_dir) if augmented_dir else None
        self.transform = transform
        self.use_augmented = use_augmented and augmented_dir is not None
        self.include_file_info = include_file_info
        self.rice_varieties = RICE_VARIETIES
        self.label_to_idx = {variety: idx for idx, variety in enumerate(self.rice_varieties)}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.augmented_paths = []  # Will store augmented image paths for each original
        self.variety_counts = {}  # Track number of original images per variety
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load original and augmented image paths."""
        
        # First pass: load original images and count per variety
        for variety in self.rice_varieties:
            variety_dir = self.original_dir / variety
            if not variety_dir.exists():
                print(f"Warning: {variety} directory not found in original data")
                self.variety_counts[variety] = 0
                continue
            
            # Collect images for this variety
            variety_images = sorted([
                img_file for img_file in variety_dir.glob("*.*")
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ])
            
            self.variety_counts[variety] = len(variety_images)
            
            # Add original images
            for img_path in variety_images:
                self.image_paths.append(str(img_path))
                self.labels.append(self.label_to_idx[variety])
            
        
        # Second pass: load augmented images and map to originals
        if self.use_augmented:
            
            # First, collect all augmented images per variety
            augmented_variety_images = {}
            for variety in self.rice_varieties:
                variety_dir = self.augmented_dir / variety
                if not variety_dir.exists():
                    augmented_variety_images[variety] = []
                    continue
                
                variety_images = sorted([
                    img_file for img_file in variety_dir.glob("*.*")
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
                ])
                augmented_variety_images[variety] = variety_images
            
            # Now create mapping: for each original image, find its 5 augmented versions
            current_original_idx = 0
            for variety in self.rice_varieties:
                num_original = self.variety_counts[variety]
                augmented_images = augmented_variety_images[variety]
                
                if len(augmented_images) == 0:
                    # No augmented data for this variety
                    for _ in range(num_original):
                        self.augmented_paths.append([])
                    current_original_idx += num_original
                    continue
                
                # For each original image of this variety
                for orig_idx in range(num_original):
                    # Find its 5 augmented versions
                    # They are at indices: orig_idx, orig_idx+num_original, 
                    # orig_idx+2*num_original, orig_idx+3*num_original, orig_idx+4*num_original
                    aug_indices = [
                        orig_idx + k * num_original 
                        for k in range(NUM_AUGMENTATIONS_PER_ORIGINAL)
                    ]
                    
                    # Get the actual augmented image paths (only if index exists)
                    aug_paths = []
                    for aug_idx in aug_indices:
                        if aug_idx < len(augmented_images):
                            aug_paths.append(str(augmented_images[aug_idx]))
                    
                    self.augmented_paths.append(aug_paths)
                
                current_original_idx += num_original
    
    def __len__(self):
        """Return total number of samples (including augmented)."""
        if self.use_augmented:
            # Each original comes with its augmented versions
            total = sum(1 + len(aug_paths) for aug_paths in self.augmented_paths)
            return total
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        If use_augmented=True:
        - Returns original image or one of its augmented versions
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (image, label) or (image, label, file_info) if include_file_info=True
        """
        if self.use_augmented:
            return self._get_augmented_item(idx)
        else:
            return self._get_original_item(idx)
    
    def _get_original_item(self, idx):
        """Get original image at index."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if self.include_file_info:
            return image, label, img_path
        return image, label
    
    def _get_augmented_item(self, idx):
        """
        Get item with augmented pairing.
        
        Strategy: For each original image, create a group of (original + augmented)
        """
        # Calculate which original image this corresponds to
        current_idx = 0
        original_image_idx = None
        augmented_idx = None
        
        for orig_idx, aug_paths in enumerate(self.augmented_paths):
            # This original + its augmented images
            total_in_group = 1 + len(aug_paths)
            
            if current_idx + total_in_group > idx:
                # Found the right group
                offset = idx - current_idx
                
                if offset == 0:
                    # Return original image
                    original_image_idx = orig_idx
                    augmented_idx = None
                else:
                    # Return augmented image
                    original_image_idx = orig_idx
                    augmented_idx = offset - 1
                break
            
            current_idx += total_in_group
        
        # Load image
        if augmented_idx is None:
            # Load original
            img_path = self.image_paths[original_image_idx]
        else:
            # Load augmented
            aug_paths = self.augmented_paths[original_image_idx]
            if augmented_idx < len(aug_paths):
                img_path = aug_paths[augmented_idx]
            else:
                # Fallback to original if augmented not available
                img_path = self.image_paths[original_image_idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[original_image_idx]
        
        if self.include_file_info:
            return image, label, img_path
        return image, label
    
    def get_dataset_statistics(self):
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(self),
            'num_classes': len(self.rice_varieties),
            'images_per_variety': self.variety_counts.copy()
        }
        return stats


def get_transforms(image_size=(224, 224), training=False):
    if training:
        transform = v2.Compose([
            v2.Resize(image_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomRotation(degrees=15),
            
            # --- PROPOSED DISTORTION TRANSFORMS ---
            # 1. Perspective (Simulates angled camera shots)
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            
            # 2. Elastic Transform (Creates slight ripples on the rice grain edges)
            # alpha: distortion intensity. Should not be too high for rice grains.
            v2.RandomApply([v2.ElasticTransform(alpha=25.0, sigma=5.0)], p=0.2),
            
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = v2.Compose([
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def prepare_dataloaders(original_dir=ORIGINAL_DATA_DIR, augmented_dir=AUGMENTED_DATA_DIR, 
                        batch_size=BATCH_SIZE, num_workers=4, use_augmented=True):
    
    # 1. Initialize the full Dataset
    dataset = RiceVarietyDataset(
        original_dir=original_dir,
        augmented_dir=augmented_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=False),
        use_augmented=use_augmented
    )
    
    # 2. ONLY USE ORIGINAL IMAGE INFO FOR SPLITTING
    num_originals = len(dataset.image_paths) # Exactly 4730 images
    original_indices = list(range(num_originals))
    original_labels = dataset.labels 
    
    # 3. Split 10% for Test (Stratified by rice variety)
    train_val_orig_idx, test_orig_idx, train_val_labels, _ = train_test_split(
        original_indices, original_labels, 
        test_size=TEST_SPLIT, random_state=42, stratify=original_labels
    )
    
    # Split another for Validation
    val_ratio = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    train_orig_idx, val_orig_idx, _, _ = train_test_split(
        train_val_orig_idx, train_val_labels, 
        test_size=val_ratio, random_state=42, stratify=train_val_labels
    )
    
    # Cast to Set for extremely fast lookups in the loop below
    set_train_orig = set(train_orig_idx)
    set_val_orig = set(val_orig_idx)
    set_test_orig = set(test_orig_idx)
    
    # 4. MAP BACK TO FLAT INDICES OF THE ENTIRE DATASET
    train_flat_indices = []
    val_flat_indices = []
    test_flat_indices = []
    
    current_flat_idx = 0
    for orig_idx in range(num_originals):
        # Calculate how many images are in this group (1 original + N augmented)
        num_in_group = 1 + len(dataset.augmented_paths[orig_idx]) if use_augmented else 1
        
        if orig_idx in set_train_orig:
            # TO TRAIN: Take the whole group (Original + 5 augmented)
            train_flat_indices.extend(range(current_flat_idx, current_flat_idx + num_in_group))
        elif orig_idx in set_val_orig:
            # TO VAL: Only take the original image (the first current_flat_idx of the group)
            val_flat_indices.append(current_flat_idx)
        elif orig_idx in set_test_orig:
            # TO TEST: Only take the original image
            test_flat_indices.append(current_flat_idx)
            
        # Jump to the next original image group
        current_flat_idx += num_in_group

    # 5. Initialize Dataset with specialized Transform for Train (Includes Distortion)
    train_dataset = RiceVarietyDataset(
        original_dir=original_dir, augmented_dir=augmented_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=True), 
        use_augmented=use_augmented
    )
    
    # Initialize pure Dataset for Val and Test (No image distortion)
    val_test_dataset = RiceVarietyDataset(
        original_dir=original_dir, augmented_dir=augmented_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=False), 
        use_augmented=use_augmented
    )
    
    # 6. Create Subsets
    train_subset = Subset(train_dataset, train_flat_indices)
    val_subset = Subset(val_test_dataset, val_flat_indices)
    test_subset = Subset(val_test_dataset, test_flat_indices)
    
    # 7. Pass into DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'dataset': dataset
    }


def print_dataset_info(dataset):
    """
    Print information about the dataset.
    
    Args:
        dataset: RiceVarietyDataset object
    """
    stats = dataset.get_dataset_statistics()
    
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total samples (with augmentation): {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print("\nImages per variety (original):")
    print("-" * 60)
    
    for variety, count in stats['images_per_variety'].items():
        print(f"  {variety:.<40} {count:>6} images")
    
    print("="*60 + "\n")


