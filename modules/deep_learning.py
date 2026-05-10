import argparse
from pathlib import Path
from typing import Dict, List, Tuple
try:
    from .config import (
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        MINIMUM_DELTA,
        MODELS_DIR,
        RANDOM_SEED,
        RESULTS_DIR,
    )
except ImportError:
    from config import (
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        MINIMUM_DELTA,
        MODELS_DIR,
        RANDOM_SEED,
        RESULTS_DIR,
    )
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


class SoftmaxClassifierHead(nn.Module):
    """Two-FC classifier head with ReLU + Dropout before Softmax logits."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.net(x)


def load_split_features(feature_dir: Path) -> Dict[str, np.ndarray]:
    data = {}
    for split in ["train", "val", "test"]:
        data[f"{split}_features"] = np.load(feature_dir / f"{split}_features.npy")
        data[f"{split}_labels"] = np.load(feature_dir / f"{split}_labels.npy")
    return data


def to_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(features).float()
    y_tensor = torch.from_numpy(labels).long()
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)

            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true_np, y_pred_np)
    return avg_loss, acc, y_true_np, y_pred_np


def plot_curves(history_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Acc")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def get_backbone_name(extractor_name: str) -> str:
    if "_224x224" in extractor_name:
        return extractor_name.split("_224x224")[0]
    return extractor_name


def train(
    feature_dir: Path,
    output_root: Path,
    model_root: Path,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> Dict[str, float]:
    data = load_split_features(feature_dir)
    train_loader = to_loader(data["train_features"], data["train_labels"], batch_size=batch_size, shuffle=True)
    val_loader = to_loader(data["val_features"], data["val_labels"], batch_size=batch_size, shuffle=False)

    input_dim = data["train_features"].shape[1]
    num_classes = int(np.max(data["train_labels"])) + 1

    model = SoftmaxClassifierHead(input_dim=input_dim, num_classes=num_classes, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    history: List[Dict[str, float]] = []
    best_state = None
    best_val_acc = -1.0
    early_stopping_patience = 5
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        y_true_epoch: List[int] = []
        y_pred_epoch: List[int] = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true_epoch.extend(yb.cpu().numpy().tolist())
            y_pred_epoch.extend(preds.cpu().numpy().tolist())

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(y_true_epoch, y_pred_epoch)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

                
        if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if val_loss < best_val_loss - MINIMUM_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break


        print(
            f"[{feature_dir.name}] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

    extractor_out = output_root / feature_dir.name
    extractor_out.mkdir(parents=True, exist_ok=True)
    backbone_name = get_backbone_name(feature_dir.name)
    checkpoint_path = model_root / f"{backbone_name}_head.pth"

    # Required flow: train first, then save checkpoint. Test evaluation is done in main.
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), checkpoint_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(extractor_out / "training_history.csv", index=False)

    metrics = {
        "extractor": feature_dir.name,
        "backbone": backbone_name,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "best_val_acc": best_val_acc,
        "checkpoint_path": str(checkpoint_path),
    }

    plot_curves(
        history_df=history_df,
        out_path=extractor_out / "training_validation_curve.png",
        title=f"Training/Validation Curves - {feature_dir.name}",
    )

    return metrics

def evaluate_test_set(
    feature_dir: Path,
    output_root: Path,
    checkpoint_path: Path,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    data = load_split_features(feature_dir)
    test_loader = to_loader(data["test_features"], data["test_labels"], batch_size=batch_size, shuffle=False)

    model = SoftmaxClassifierHead(input_dim=input_dim, num_classes=num_classes, dropout=0.2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, y_true_test, y_pred_test = evaluate(model, test_loader, criterion, device)
    test_precision = precision_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
    test_recall = recall_score(y_true_test, y_pred_test, average="weighted", zero_division=0)
    test_f1 = f1_score(y_true_test, y_pred_test, average="weighted", zero_division=0)

    extractor_out = output_root / feature_dir.name
    extractor_out.mkdir(parents=True, exist_ok=True)
    plot_confusion(
        y_true=y_true_test,
        y_pred=y_pred_test,
        out_path=extractor_out/"confusion_matrix.png",
        title=f"Confusion Matrix - {feature_dir.name}",
    )

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision_weighted": test_precision,
        "test_recall_weighted": test_recall,
        "test_f1_weighted": test_f1,
    }


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    features_root = RESULTS_DIR / "features_all"
    result_root = RESULTS_DIR / "deep_learning"
    model_root = MODELS_DIR / "deep_learningmodel"
    result_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    feature_dirs = sorted([d for d in features_root.iterdir() if d.is_dir()])
    if len(feature_dirs) == 0:
        raise FileNotFoundError(f"No extractor folder found in {features_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Features root: {features_root}")
    print(f"Result root: {result_root}")
    print(f"Model root: {model_root}")

    all_metrics: List[Dict[str, float]] = []
    for feature_dir in feature_dirs:
        train_info = train(
            feature_dir=feature_dir,
            output_root=result_root,
            model_root=model_root,
            device=device,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
        )
        test_metrics = evaluate_test_set(
            feature_dir=feature_dir,
            output_root=result_root,
            checkpoint_path=Path(train_info["checkpoint_path"]),
            input_dim=int(train_info["input_dim"]),
            num_classes=int(train_info["num_classes"]),
            batch_size=BATCH_SIZE,
            device=device,
        )
        merged_metrics = {**train_info, **test_metrics}
        merged_metrics_csv = {
            k: v for k, v in merged_metrics.items() if k not in {"checkpoint_path", "input_dim", "num_classes"}
        }
        metrics_out_dir = result_root / feature_dir.name
        metrics_out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([merged_metrics_csv]).to_csv(metrics_out_dir / "classifier_metrics.csv", index=False)
        all_metrics.append(merged_metrics_csv)

    summary_df = pd.DataFrame(all_metrics).sort_values("test_acc", ascending=False)
    summary_path = result_root / "all_extractors_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
