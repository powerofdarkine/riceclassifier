"""PyTorch data loading for rice variety images."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
try:
    from .config import *
except ImportError:
    from config import *
from sklearn.model_selection import train_test_split


class RiceVarietyDataset(Dataset):
    """Simple dataset that reads images from class subfolders."""

    def __init__(self, original_dir, transform=None):
        self.original_dir = Path(original_dir)
        self.transform = transform
        self.rice_varieties = RICE_VARIETIES
        self.label_to_idx = {variety: idx for idx, variety in enumerate(self.rice_varieties)}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load image paths and labels from disk."""
        for variety in self.rice_varieties:
            variety_dir = self.original_dir / variety
            if not variety_dir.exists():
                continue
            
            variety_images = sorted([
                img_file for img_file in variety_dir.glob("*.*")
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ])
            for img_path in variety_images:
                self.image_paths.append(str(img_path))
                self.labels.append(self.label_to_idx[variety])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size=(224, 224), training=False):
    """Basic transforms; normalization can be overridden by feature extraction code."""
    ops = [
        v2.Resize(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return v2.Compose(ops)

def prepare_dataloaders(original_dir=ORIGINAL_DATA_DIR,
                        batch_size=BATCH_SIZE, num_workers=4):
    
    # Initialize dataset (used for stratified split)
    dataset = RiceVarietyDataset(
        original_dir=original_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=False)
    )
    
    # Stratified split by class label
    num_originals = len(dataset.image_paths)
    original_indices = list(range(num_originals))
    original_labels = dataset.labels 
    
    train_val_orig_idx, test_orig_idx, train_val_labels, _ = train_test_split(
        original_indices, original_labels, 
        test_size=TEST_SPLIT, random_state=42, stratify=original_labels
    )
    
    val_ratio = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
    train_orig_idx, val_orig_idx, _, _ = train_test_split(
        train_val_orig_idx, train_val_labels, 
        test_size=val_ratio, random_state=42, stratify=train_val_labels
    )
    
    train_flat_indices = list(train_orig_idx)
    val_flat_indices = list(val_orig_idx)
    test_flat_indices = list(test_orig_idx)

    train_dataset = RiceVarietyDataset(
        original_dir=original_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=True)
    )
    
    val_test_dataset = RiceVarietyDataset(
        original_dir=original_dir,
        transform=get_transforms(image_size=IMAGE_SIZE, training=False)
    )
    
    train_subset = Subset(train_dataset, train_flat_indices)
    val_subset = Subset(val_test_dataset, val_flat_indices)
    test_subset = Subset(val_test_dataset, test_flat_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'dataset': dataset
    }
