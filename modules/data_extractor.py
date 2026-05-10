from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
from tqdm import tqdm

try:
    from .config import ORIGINAL_DATA_DIR, RESULTS_DIR
    from .data_loader import prepare_dataloaders
    from .models import ALL_BACKBONES, build_backbone, eval_transform_for_backbone
except ImportError:
    from config import ORIGINAL_DATA_DIR, RESULTS_DIR
    from data_loader import prepare_dataloaders
    from models import ALL_BACKBONES, build_backbone, eval_transform_for_backbone


@torch.no_grad()
def extract_features(backbone: torch.nn.Module, dataloader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (features, labels) for a single dataloader split.
    Kept as a simple helper to be reused by downstream scripts.
    """
    backbone.eval().to(device)
    feats, labels = [], []
    for images, y in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        out = backbone(images)
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        feats.append(out.cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(feats).astype(np.float32), np.concatenate(labels).astype(np.int64)


def _save_npy(out_dir: Path, split: str, feats: np.ndarray, labels: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_features.npy", feats)
    np.save(out_dir / f"{split}_labels.npy", labels)


def run_all_models(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """
    Main entrypoint (NO CLI): extract features for ALL backbones and save.

    cfg keys (all optional):
      - original_dir: Path
      - image_size: (H,W)
      - batch_size: int
      - num_workers: int
      - device: "cuda" | "cpu"
      - out_root: Path (default results/features_all)
      - backbones: list[str] (default ALL_BACKBONES)
    """
    cfg = cfg or {}

    original_dir: Path = Path(cfg.get("original_dir", ORIGINAL_DATA_DIR))
    image_size: Tuple[int, int] = tuple(cfg.get("image_size", (224, 224)))  # type: ignore[arg-type]
    batch_size: int = int(cfg.get("batch_size", 32))
    num_workers: int = int(cfg.get("num_workers", 4))
    device: str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_root: Path = Path(cfg.get("out_root", RESULTS_DIR / "features_all"))
    backbones: List[str] = list(cfg.get("backbones", ALL_BACKBONES))

    loaders = prepare_dataloaders(
        original_dir=original_dir, batch_size=batch_size, num_workers=num_workers
    )

    out_root.mkdir(parents=True, exist_ok=True)

    for bb in backbones:
        print("\n" + "=" * 70)
        print(f"[EXTRACT] backbone={bb} | image_size={image_size} | device={device}")
        print("=" * 70)

        tf = eval_transform_for_backbone(bb, image_size)
        for split in ("train", "val", "test"):
            subset = loaders[split].dataset
            subset.dataset.transform = tf

        backbone = build_backbone(bb)

        tag = f"{bb}_{image_size[0]}x{image_size[1]}"
        bb_dir = out_root / tag

        for split in ("train", "val", "test"):
            x, y = extract_features(backbone, loaders[split], device=device)
            _save_npy(bb_dir, split, x, y)
            print(f"  {split}: {x.shape} | labels={y.shape}")

        print(f"Saved features: {bb_dir}")

    return out_root


if __name__ == "__main__":
    run_all_models()
