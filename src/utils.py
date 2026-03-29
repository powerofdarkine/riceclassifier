"""
Utility functions for Rice Variety Classification
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle


class Logger:
    """Simple logging utility"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_file = self.log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")


def save_metrics(metrics_dict, save_path):
    """
    Save metrics to JSON file
    
    Args:
        metrics_dict: Dictionary with metrics
        save_path: Path to save file
    """
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics_dict)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)


def load_metrics(load_path):
    """
    Load metrics from JSON file
    
    Args:
        load_path: Path to load file
        
    Returns:
        Dictionary with metrics
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def get_dataset_statistics(data_dir):
    """
    Get statistics about the dataset
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_images': 0,
        'images_per_variety': {},
        'total_varieties': 0
    }
    
    data_dir = Path(data_dir)
    
    for variety_dir in data_dir.iterdir():
        if variety_dir.is_dir():
            image_count = len(list(variety_dir.glob("*.*")))
            stats['images_per_variety'][variety_dir.name] = image_count
            stats['total_images'] += image_count
    
    stats['total_varieties'] = len(stats['images_per_variety'])
    
    return stats


def print_dataset_statistics(data_dir):
    """
    Print dataset statistics
    
    Args:
        data_dir: Path to dataset directory
    """
    stats = get_dataset_statistics(data_dir)
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total Varieties: {stats['total_varieties']}")
    print(f"Total Images: {stats['total_images']}")
    print(f"Average Images per Variety: {stats['total_images'] / stats['total_varieties']:.1f}")
    
    print("\nImages per Variety:")
    for variety, count in sorted(stats['images_per_variety'].items()):
        print(f"  {variety}: {count}")
    
    print("="*60 + "\n")


def setup_project_dirs(project_root):
    """
    Create necessary project directories
    
    Args:
        project_root: Root directory of the project
    """
    dirs_to_create = [
        'models',
        'results',
        'logs',
        'src'
    ]
    
    project_root = Path(project_root)
    
    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✓ {dir_name}/ directory created/verified")


def verify_dataset(original_dir, rice_varieties):
    """
    Verify dataset integrity
    
    Args:
        original_dir: Path to original dataset
        rice_varieties: List of rice variety names
        
    Returns:
        Dictionary with verification results
    """
    results = {
        'original_complete': True,
        'missing_varieties': [],
        'original_stats': {}
    }
    
    original_dir = Path(original_dir)
    
    print("\nVerifying dataset...")
    
    # Check original dataset
    for variety in rice_varieties:
        variety_path = original_dir / variety
        if not variety_path.exists():
            results['original_complete'] = False
            results['missing_varieties'].append(f"Original/{variety}")
        else:
            image_count = len(list(variety_path.glob("*.*")))
            results['original_stats'][variety] = image_count
    
    return results


def create_project_summary(results_dict, output_path):
    """
    Create a summary report
    
    Args:
        results_dict: Dictionary with all results
        output_path: Path to save summary
    """
    summary = []
    summary.append("=" * 80)
    summary.append("RICE VARIETY CLASSIFICATION PROJECT - SUMMARY REPORT")
    summary.append("=" * 80)
    summary.append("")
    
    summary.append("PROJECT STRUCTURE:")
    summary.append("  ├── src/")
    summary.append("  │   ├── config.py          - Configuration and hyperparameters")
    summary.append("  │   ├── data_loader.py     - Data loading and preprocessing")
    summary.append("  │   ├── models.py          - Model definitions")
    summary.append("  │   ├── train.py           - Training script")
    summary.append("  │   ├── evaluate.py        - Evaluation script")
    summary.append("  │   └── utils.py           - Utility functions")
    summary.append("  ├── models/                - Trained models")
    summary.append("  ├── results/               - Evaluation results")
    summary.append("  ├── Original/              - Original dataset")
    summary.append("  ├── requirements.txt       - Dependencies")
    summary.append("  └── README.md              - Documentation")
    summary.append("")
    
    summary.append("QUICK START:")
    summary.append("  1. Install dependencies:  pip install -r requirements.txt")
    summary.append("  2. Train models:          python src/train.py")
    summary.append("  3. Evaluate models:       python src/evaluate.py")
    summary.append("  4. Make predictions:      python src/evaluate.py --predict image.jpg")
    summary.append("")
    
    if results_dict:
        summary.append("MODEL PERFORMANCE:")
        if 'ml_models' in results_dict:
            summary.append("  Machine Learning Models:")
            for model_name, metrics in results_dict['ml_models'].items():
                summary.append(f"    {model_name}: Accuracy={metrics['accuracy']:.4f}")
        
        if 'cnn_model' in results_dict:
            summary.append("  Deep Learning Model:")
            summary.append(f"    CNN: Accuracy={results_dict['cnn_model']['accuracy']:.4f}")
        
        summary.append("")
    
    summary.append("RICE VARIETIES (20 total):")
    if 'rice_varieties' in results_dict:
        for i, variety in enumerate(results_dict['rice_varieties'], 1):
            summary.append(f"  {i:2d}. {variety}")
    
    summary.append("")
    summary.append("=" * 80)
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 80)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write("\n".join(summary))
    
    # Print to console
    print("\n".join(summary))


if __name__ == "__main__":
    print("This module provides utility functions for the Rice Classification project.")
    print("Import it in your scripts to use the utility functions.")