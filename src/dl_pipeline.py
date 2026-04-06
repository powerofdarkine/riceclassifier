import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

# Import từ các file của bạn
from config import ORIGINAL_DATA_DIR, RICE_VARIETIES, RESULTS_DIR, NUM_CLASSES, BATCH_SIZE, IMAGE_SIZE
from data_loader import prepare_dataloaders

# =====================================================================
# 1. FOCAL LOSS (PYTORCH)
# =====================================================================
class FocalLoss(nn.Module):
    """Multi-class Focal Loss for classification."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()


# =====================================================================
# 2. KIẾN TRÚC MÔ HÌNH MULTI-SCALE (GAP/GMP)
# =====================================================================
class AdvancedRiceNet(nn.Module):
    """Multi-scale feature extraction using GAP and GMP pooling."""
    def __init__(self, backbone_name='resnet50', num_classes=20, num_layers=4):
        super(AdvancedRiceNet, self).__init__()
        import torchvision.models as models
        
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet50':
            base = models.resnet50(weights='DEFAULT')
            self.initial = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1  # 256 channels
            self.layer2 = base.layer2  # 512 channels
            self.layer3 = base.layer3  # 1024 channels
            self.layer4 = base.layer4  # 2048 channels
            # Concatenate 4 layers * (GAP + GMP) = (256 + 512 + 1024 + 2048) * 2
            input_dim = (256 + 512 + 1024 + 2048) * 2
            
        elif backbone_name == 'resnet18':
            base = models.resnet18(weights='DEFAULT')
            self.initial = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1  # 64 channels
            self.layer2 = base.layer2  # 128 channels
            self.layer3 = base.layer3  # 256 channels
            self.layer4 = base.layer4  # 512 channels
            input_dim = (64 + 128 + 256 + 512) * 2
            
        elif backbone_name == 'efficientnet_b0':
            base = models.efficientnet_b0(weights='DEFAULT')
            self.initial = nn.Sequential(base.features[0:2])
            self.layer1 = nn.Sequential(base.features[2:3])
            self.layer2 = nn.Sequential(base.features[3:4])
            self.layer3 = nn.Sequential(base.features[4:6])
            self.layer4 = nn.Sequential(base.features[6:9])
            input_dim = (40 + 80 + 112 + 320) * 2
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # Classifier head with dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Multi-scale pooling
        pooled = []
        for f in [f1, f2, f3, f4]:
            gap = self.gap(f).view(f.size(0), -1)
            gmp = self.gmp(f).view(f.size(0), -1)
            pooled.append(gap)
            pooled.append(gmp)
        
        combined = torch.cat(pooled, dim=1)
        return self.classifier(combined)


# =====================================================================
# 3. TRAINING UTILITIES
# =====================================================================
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=5, verbose=True, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def freeze_backbone(model):
    """Freeze all layers except classifier."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def set_bn_eval(model):
    """Set BatchNorm layers to evaluation mode (no weight updates)."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()


# =====================================================================
# 4. TRAINING LOOP
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch")
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == lbls).sum().item()
        total += lbls.size(0)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100*correct/total:.2f}%"
        })
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc


def validate_one_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds)


# =====================================================================
# 5. EVALUATION AND REPORTING
# =====================================================================
def evaluate_model(model, loader, device, run_folder):
    """Complete evaluation with metrics and visualizations."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(lbls.numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=RICE_VARIETIES, digits=4)
    print("\n" + "="*80)
    print("Classification Report:")
    print("="*80)
    print(report)
    
    with open(os.path.join(run_folder, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=RICE_VARIETIES, yticklabels=RICE_VARIETIES, cbar_kws={'label': 'Normalized Count'})
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(run_folder, 'confusion_matrix.png')}")
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    metrics_df = {
        'Class': RICE_VARIETIES,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    import pandas as pd
    df = pd.DataFrame(metrics_df)
    df.to_csv(os.path.join(run_folder, "metrics_per_class.csv"), index=False)
    print(f"✓ Saved: {os.path.join(run_folder, 'metrics_per_class.csv')}")
    
    return all_labels, all_preds


def plot_training_history(history, run_folder):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "training_history.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.join(run_folder, 'training_history.png')}")


def save_model_and_metadata(model, run_folder, exp_name):
    """Save model weights and metadata."""
    torch.save(model.state_dict(), os.path.join(run_folder, f"{exp_name}_best.pth"))
    
    # Class mapping
    label_map = {i: name for i, name in enumerate(RICE_VARIETIES)}
    with open(os.path.join(run_folder, "class_indices.json"), "w") as f:
        json.dump(label_map, f, indent=4)
    
    print(f"✓ Saved: {os.path.join(run_folder, f'{exp_name}_best.pth')}")


def print_exp_summary(exp_name, config, save_path):
    """Print experiment configuration summary."""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_name}")
    print("="*80)
    print(f"  Backbone        : {config['backbone']}")
    print(f"  Loss Function   : {config['loss_fn']}")
    print(f"  Phase 1 Epochs  : {config['p1_epochs']} (backbone frozen)")
    print(f"  Phase 1 LR      : {config['lr_p1']}")
    print(f"  Phase 2 Epochs  : {config['p2_epochs']} (full fine-tune)")
    print(f"  Phase 2 LR      : {config['lr_p2']}")
    print(f"  Batch Size      : {BATCH_SIZE}")
    print(f"  Image Size      : {IMAGE_SIZE}")
    print(f"  Save Path       : {save_path}")
    print("="*80 + "\n")


# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================
def run_pipeline():
    """Main training pipeline with multiple experiences."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Using device: {device}")
    print(f"{'='*80}\n")
    
    # Define multiple experiences
    EXPERIENCES = {
        "res50_focal_frozen": {
            "backbone": "resnet50",
            "loss_fn": "focal",
            "p1_epochs": 12,
            "lr_p1": 1e-4,
            "p2_epochs": 0,  # No fine-tuning
            "lr_p2": 1e-5,
        },
        "res50_focal_finetune": {
            "backbone": "resnet50",
            "loss_fn": "focal",
            "p1_epochs": 12,
            "lr_p1": 1e-4,
            "p2_epochs": 18,
            "lr_p2": 1e-5,
        },
        "res50_ce_finetune": {
            "backbone": "resnet50",
            "loss_fn": "crossentropy",
            "p1_epochs": 12,
            "lr_p1": 1e-4,
            "p2_epochs": 18,
            "lr_p2": 1e-5,
        },
        "res18_focal_finetune": {
            "backbone": "resnet18",
            "loss_fn": "focal",
            "p1_epochs": 12,
            "lr_p1": 1e-4,
            "p2_epochs": 18,
            "lr_p2": 1e-5,
        },
        "efficientnet_b0_ce": {
            "backbone": "efficientnet_b0",
            "loss_fn": "crossentropy",
            "p1_epochs": 15,
            "lr_p1": 1e-4,
            "p2_epochs": 15,
            "lr_p2": 1e-5,
        },
    }
    
    # Load data once
    print("Loading dataset...")
    loaders = prepare_dataloaders(ORIGINAL_DATA_DIR, batch_size=BATCH_SIZE)
    print(f"✓ Dataset loaded: Train={len(loaders['train'].dataset)}, Val={len(loaders['val'].dataset)}, Test={len(loaders['test'].dataset)}")
    
    # Run each experience
    for exp_name, config in EXPERIENCES.items():
        date_tag = datetime.now().strftime("%Y%m%d")
        run_folder = os.path.join(RESULTS_DIR, f"{exp_name}_{date_tag}")
        os.makedirs(run_folder, exist_ok=True)
        
        print_exp_summary(exp_name, config, run_folder)
        
        # Initialize model and loss
        model = AdvancedRiceNet(config['backbone'], NUM_CLASSES).to(device)
        criterion = FocalLoss() if config['loss_fn'] == 'focal' else nn.CrossEntropyLoss()
        
        # Initialize history tracking
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        best_val_loss = float('inf')
        early_stopping = EarlyStopping(patience=7, verbose=True)
        
        # PHASE 1: FROZEN BACKBONE
        print(f"\n{'─'*80}")
        print(f"PHASE 1: Training Head Only ({config['p1_epochs']} epochs)")
        print(f"{'─'*80}")
        
        freeze_backbone(model)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr_p1'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        for epoch in range(config['p1_epochs']):
            train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device, epoch, config['p1_epochs'])
            val_loss, val_acc, _, _ = validate_one_epoch(model, loaders['val'], criterion, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            print(f"  → Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(run_folder, f"{exp_name}_best.pth"))
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # PHASE 2: FULL FINE-TUNING
        if config['p2_epochs'] > 0:
            print(f"\n{'─'*80}")
            print(f"PHASE 2: Full Fine-tuning ({config['p2_epochs']} epochs)")
            print(f"{'─'*80}")
            
            unfreeze_backbone(model)
            optimizer = optim.Adam(model.parameters(), lr=config['lr_p2'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            early_stopping = EarlyStopping(patience=7, verbose=True)
            
            for epoch in range(config['p2_epochs']):
                train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device, epoch, config['p2_epochs'])
                val_loss, val_acc, _, _ = validate_one_epoch(model, loaders['val'], criterion, device)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_loss)
                
                print(f"  → Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(run_folder, f"{exp_name}_best.pth"))
                
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(run_folder, f"{exp_name}_best.pth")))
        
        # FINAL EVALUATION
        print(f"\n{'─'*80}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'─'*80}")
        evaluate_model(model, loaders['test'], device, run_folder)
        
        # Save visualizations and metadata
        plot_training_history(history, run_folder)
        save_model_and_metadata(model, run_folder, exp_name)
        
        # Save history as JSON
        with open(os.path.join(run_folder, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)
        
        print(f"\n✓ Experiment '{exp_name}' completed.")
        print(f"✓ Results saved to: {run_folder}\n")


if __name__ == "__main__":
    run_pipeline()