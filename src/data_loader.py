"""
PyTorch Data Loading and Preprocessing Module for Rice Variety Classification
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from config import *
from sklearn.model_selection import train_test_split


class RiceVarietyDataset(Dataset):
    """
    Custom PyTorch Dataset for rice variety classification.
    """
    
    def __init__(self, original_dir, transform=None, include_file_info=False):
        """
        Initialize dataset.
        
        Args:
            original_dir: Path to original images directory (contains variety subdirectories)
            transform: PyTorch transforms to apply
            include_file_info: Whether to return file information for debugging
        """
        self.original_dir = Path(original_dir)
        self.transform = transform
        self.include_file_info = include_file_info
        self.rice_varieties = RICE_VARIETIES
        self.label_to_idx = {variety: idx for idx, variety in enumerate(self.rice_varieties)}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.variety_counts = {}
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load original image paths."""
        
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
    
    def __len__(self):
        """Return total number of original samples."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (image, label) or (image, label, file_info) if include_file_info=True
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
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

def prepare_dataloaders(original_dir=ORIGINAL_DATA_DIR,
                        batch_size=BATCH_SIZE, num_workers=4):
    
    # 1. Initialize the full Dataset
    dataset = RiceVarietyDataset(
        original_dir=original_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=False)
    )
    
    # 2. ONLY USE ORIGINAL IMAGE INFO FOR SPLITTING
    num_originals = len(dataset.image_paths) # Exactly 4730 images
    original_indices = list(range(num_originals))
    original_labels = dataset.labels 
    
    # 3. Split for Test (stratified by rice variety)
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
    
    # 4. For original-only dataset, flat indices are the same as original indices
    train_flat_indices = list(train_orig_idx)
    val_flat_indices = list(val_orig_idx)
    test_flat_indices = list(test_orig_idx)

    # 5. Initialize Dataset with specialized Transform for Train
    train_dataset = RiceVarietyDataset(
        original_dir=original_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=True)
    )
    
    # Initialize pure Dataset for Val and Test (no random distortion)
    val_test_dataset = RiceVarietyDataset(
        original_dir=original_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=False)
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
    print(f"Total original samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print("\nImages per variety (original):")
    print("-" * 60)
    
    for variety, count in stats['images_per_variety'].items():
        print(f"  {variety:.<40} {count:>6} images")
    
    print("="*60 + "\n")