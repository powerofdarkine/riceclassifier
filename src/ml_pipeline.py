from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    torch = importlib.import_module("torch")
except Exception:
    torch = None


@dataclass
class ArchitectureData:
    name: str
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


GPU_MODEL_NAMES = {"logreg", "linear_svm", "knn", "random_forest"}
TORCH_GPU_MODEL_NAMES = {"logreg", "linear_svm", "knn"}


class TorchLogRegClassifier:
    def __init__(self, device: str, lr: float = 0.05, epochs: int = 250, weight_decay: float = 1e-4):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.model = None
        self.classes_ = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchLogRegClassifier":
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is not installed.")

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)

        self.classes_ = torch.unique(y_t).detach().cpu().numpy()
        in_dim = int(x_t.shape[1])
        num_classes = int(torch.max(y_t).item() + 1)

        self.model = torch.nn.Linear(in_dim, num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(x_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if torch is None or self.model is None:
            raise RuntimeError("Model is not fitted.")

        self.model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            logits = self.model(x_t)
            pred = torch.argmax(logits, dim=1)
        return pred.detach().cpu().numpy()


class TorchLinearSVMClassifier:
    def __init__(self, device: str, c: float = 1.0, lr: float = 0.02, epochs: int = 300):
        self.device = device
        self.c = c
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchLinearSVMClassifier":
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is not installed.")

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)

        in_dim = int(x_t.shape[1])
        num_classes = int(torch.max(y_t).item() + 1)
        self.model = torch.nn.Linear(in_dim, num_classes).to(self.device)

        # C in SVM is inverse of regularization strength; approximate via weight decay.
        weight_decay = 1.0 / max(self.c, 1e-6)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)
        criterion = torch.nn.MultiMarginLoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            scores = self.model(x_t)
            loss = criterion(scores, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if torch is None or self.model is None:
            raise RuntimeError("Model is not fitted.")

        self.model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            scores = self.model(x_t)
            pred = torch.argmax(scores, dim=1)
        return pred.detach().cpu().numpy()


class TorchKNNClassifier:
    def __init__(self, device: str, n_neighbors: int = 5, weights: Literal["uniform", "distance"] = "distance"):
        self.device = device
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.x_train = None
        self.y_train = None
        self.n_classes = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchKNNClassifier":
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is not installed.")

        self.x_train = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.y_train = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.n_classes = int(torch.max(self.y_train).item() + 1)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if torch is None or self.x_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted.")

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        preds: List[np.ndarray] = []

        chunk_size = 512
        for start in range(0, x_t.shape[0], chunk_size):
            end = min(start + chunk_size, x_t.shape[0])
            chunk = x_t[start:end]

            dists = torch.cdist(chunk, self.x_train)
            top_dists, top_idx = torch.topk(dists, k=self.n_neighbors, largest=False, dim=1)
            neigh_labels = self.y_train[top_idx]

            votes = torch.zeros((chunk.shape[0], self.n_classes), device=self.device)
            if self.weights == "distance":
                weights = 1.0 / (top_dists + 1e-8)
            else:
                weights = torch.ones_like(top_dists)

            votes.scatter_add_(1, neigh_labels, weights)
            chunk_pred = torch.argmax(votes, dim=1)
            preds.append(chunk_pred.detach().cpu().numpy())

        return np.concatenate(preds)


def use_cuml_for_model(model_name: str, backend: Dict[str, Any]) -> bool:
    return bool(backend.get("use_cuml")) and model_name in GPU_MODEL_NAMES


def use_torch_gpu_for_model(model_name: str, backend: Dict[str, Any]) -> bool:
    return bool(backend.get("use_torch_gpu")) and model_name in TORCH_GPU_MODEL_NAMES


def resolve_device(requested: str) -> str:
    req = requested.lower()
    if req not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device '{requested}'. Use one of: auto, cpu, cuda, mps")

    if req == "cpu":
        return "cpu"

    if req == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    if req == "mps":
        if torch is not None and torch.backends.mps.is_available():
            return "mps"
        print("[WARN] MPS requested but not available. Falling back to CPU.")
        return "cpu"

    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_backend(device: str) -> Dict[str, Any]:
    backend: Dict[str, Any] = {
        "device": device,
        "use_cuml": False,
        "use_torch_gpu": False,
        "torch_device": "cpu",
        "cp": None,
        "label": "cpu_sklearn",
    }

    if device == "cuda":
        try:
            import cupy as cp  # type: ignore[import-not-found]
            from cuml.ensemble import RandomForestClassifier as CuRF  # type: ignore[import-not-found]
            from cuml.linear_model import LogisticRegression as CuLogReg  # type: ignore[import-not-found]
            from cuml.neighbors import KNeighborsClassifier as CuKNN  # type: ignore[import-not-found]
            from cuml.svm import SVC as CuSVC  # type: ignore[import-not-found]

            backend.update(
                {
                    "use_cuml": True,
                    "cp": cp,
                    "CuLogReg": CuLogReg,
                    "CuSVC": CuSVC,
                    "CuKNN": CuKNN,
                    "CuRF": CuRF,
                    "label": "cuda_cuml",
                }
            )
            print("[INFO] Using CUDA backend with cuML for supported models.")
        except Exception as exc:
            print(f"[WARN] CUDA detected but cuML/CuPy unavailable ({exc}).")
            if torch is not None and torch.cuda.is_available():
                backend.update(
                    {
                        "use_torch_gpu": True,
                        "torch_device": "cuda",
                        "label": "cuda_torch",
                    }
                )
                print("[INFO] Falling back to torch CUDA backend for supported models.")
            else:
                print("[WARN] Torch CUDA unavailable. Using CPU sklearn backend.")
                backend["device"] = "cpu"

    elif device == "mps":
        if torch is not None and torch.backends.mps.is_available():
            backend.update(
                {
                    "use_torch_gpu": True,
                    "torch_device": "mps",
                    "label": "mps_torch",
                }
            )
            print("[INFO] Using torch MPS backend for supported models.")
        else:
            print("[WARN] MPS requested but torch MPS unavailable. Using CPU sklearn backend.")
            backend["device"] = "cpu"

    else:
        print("[INFO] Using CPU sklearn backend.")

    return backend


def to_numpy(arr: Any, backend: Dict[str, Any]) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr

    cp = backend.get("cp")
    if cp is not None:
        try:
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        except Exception:
            pass

    if hasattr(arr, "get"):
        return np.asarray(arr.get())

    return np.asarray(arr)


def maybe_to_gpu(arr: np.ndarray, backend: Dict[str, Any]) -> Any:
    cp = backend.get("cp")
    if backend.get("use_cuml") and cp is not None:
        return cp.asarray(arr)
    return arr


def load_split(arch_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(arch_dir / f"{split}_features.npy")
    y = np.load(arch_dir / f"{split}_labels.npy").ravel().astype(np.int64)
    return x, y


def load_architecture_data(arch_dir: Path) -> ArchitectureData:
    train_x, train_y = load_split(arch_dir, "train")
    val_x, val_y = load_split(arch_dir, "val")
    test_x, test_y = load_split(arch_dir, "test")

    return ArchitectureData(
        name=arch_dir.name,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )


def get_param_grid() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "logreg": [
            {"C": 0.1},
            {"C": 1.0},
            {"C": 3.0},
        ],
        "linear_svm": [
            {"C": 0.1},
            {"C": 1.0},
            {"C": 3.0},
        ],
        "knn": [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "distance"},
            {"n_neighbors": 9, "weights": "distance"},
        ],
        "random_forest": [
            {"n_estimators": 200, "max_depth": 20, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": 30, "max_features": "sqrt"},
            {"n_estimators": 400, "max_depth": 40, "max_features": "sqrt"},
        ],
        "gradient_boosting": [
            {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 31},
            {"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 31},
            {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 63},
        ],
    }


def build_model(model_name: str, params: Dict[str, Any], random_state: int, backend: Dict[str, Any]) -> Any:
    if use_cuml_for_model(model_name, backend):
        if model_name == "logreg":
            return backend["CuLogReg"](
                C=float(params["C"]),
                max_iter=2000,
            )

        if model_name == "linear_svm":
            return backend["CuSVC"](
                C=float(params["C"]),
                kernel="linear",
            )

        if model_name == "knn":
            weights = cast(Literal["uniform", "distance"], params["weights"])
            return backend["CuKNN"](
                n_neighbors=int(params["n_neighbors"]),
                weights=weights,
            )

        if model_name == "random_forest":
            return backend["CuRF"](
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                random_state=random_state,
            )

    if use_torch_gpu_for_model(model_name, backend):
        torch_device = str(backend.get("torch_device", "cpu"))

        if model_name == "logreg":
            return TorchLogRegClassifier(
                device=torch_device,
                lr=0.05,
                epochs=250,
            )

        if model_name == "linear_svm":
            return TorchLinearSVMClassifier(
                device=torch_device,
                c=float(params["C"]),
                lr=0.02,
                epochs=300,
            )

        if model_name == "knn":
            weights = cast(Literal["uniform", "distance"], params["weights"])
            return TorchKNNClassifier(
                device=torch_device,
                n_neighbors=int(params["n_neighbors"]),
                weights=weights,
            )

    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                max_iter=2000,
                random_state=random_state,
            ),
        )

    if model_name == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=float(params["C"]),
                random_state=random_state,
                dual=False,
                max_iter=5000,
                tol=1e-3,
            ),
        )

    if model_name == "knn":
        weights = cast(Literal["uniform", "distance"], params["weights"])
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(params["n_neighbors"]),
                weights=weights,
            ),
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(params["max_iter"]),
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            random_state=random_state,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def tune_model_on_val(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    random_state: int,
    backend: Dict[str, Any],
) -> tuple[Any, Dict[str, Any], Dict[str, float], float, int]:
    grid = get_param_grid()[model_name]
    best_model: Any = None
    best_params: Dict[str, Any] = {}
    best_metrics: Dict[str, float] = {}
    best_score = -1.0

    tuning_start = time.perf_counter()

    for params in grid:
        use_gpu_model = use_cuml_for_model(model_name, backend)
        fit_train_x = maybe_to_gpu(train_x, backend) if use_gpu_model else train_x
        fit_train_y = maybe_to_gpu(train_y, backend) if use_gpu_model else train_y
        fit_val_x = maybe_to_gpu(val_x, backend) if use_gpu_model else val_x

        model = build_model(model_name, params, random_state, backend)
        model.fit(fit_train_x, fit_train_y)
        val_pred = model.predict(fit_val_x)
        val_metrics = evaluate_split(val_y, to_numpy(val_pred, backend))

        score = val_metrics["accuracy"]
        tie_break = val_metrics["f1_macro"]

        if (
            score > best_score
            or (score == best_score and tie_break > best_metrics.get("f1_macro", -1.0))
        ):
            best_score = score
            best_model = model
            best_params = params
            best_metrics = val_metrics

    tuning_time = time.perf_counter() - tuning_start

    if best_model is None:
        raise RuntimeError(f"No model candidate evaluated for: {model_name}")

    return best_model, best_params, best_metrics, tuning_time, len(grid)


def run_architecture_benchmark(
    arch_data: ArchitectureData,
    random_state: int,
    backend: Dict[str, Any],
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    model_names = ["logreg", "linear_svm", "knn", "random_forest", "gradient_boosting"]

    print(f"\n=== Architecture: {arch_data.name} ===")
    print(
        f"train={arch_data.train_x.shape}, val={arch_data.val_x.shape}, test={arch_data.test_x.shape}"
    )

    for model_name in model_names:
        model, best_params, val_metrics, tuning_time, n_trials = tune_model_on_val(
            model_name,
            arch_data.train_x,
            arch_data.train_y,
            arch_data.val_x,
            arch_data.val_y,
            random_state,
            backend,
        )

        use_gpu_model = use_cuml_for_model(model_name, backend)
        eval_test_x = maybe_to_gpu(arch_data.test_x, backend) if use_gpu_model else arch_data.test_x
        test_pred = model.predict(eval_test_x)
        test_metrics = evaluate_split(arch_data.test_y, to_numpy(test_pred, backend))

        row: Dict[str, float | str] = {
            "architecture": arch_data.name,
            "model": model_name,
            "device": str(backend.get("device", "cpu")),
            "backend": str(backend.get("label", "cpu_sklearn")),
            "best_params": json.dumps(best_params, sort_keys=True),
            "tuning_trials": int(n_trials),
            "fit_time_sec": round(tuning_time, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
        }
        rows.append(row)

        print(
            f"{model_name:>17} | val_acc={row['val_accuracy']:.4f} | "
            f"test_acc={row['test_accuracy']:.4f} | tune={row['fit_time_sec']}s | "
            f"best={row['best_params']}"
        )

    return rows


def summarize_best_models(results_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for arch_name, group in results_df.groupby("architecture"):
        best_idx = group["val_accuracy"].idxmax()
        best_rows.append(group.loc[best_idx])

    best_df = pd.DataFrame(best_rows).sort_values("test_accuracy", ascending=False)
    cols = [
        "architecture",
        "model",
        "val_accuracy",
        "val_f1_macro",
        "test_accuracy",
        "test_f1_macro",
        "fit_time_sec",
    ]
    return best_df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 5 ML classifiers for each extracted feature architecture."
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("results/features_all"),
        help="Root folder containing architecture feature subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ml_benchmarks"),
        help="Folder to save benchmark outputs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference: auto, cpu, cuda, or mps.",
    )
    args = parser.parse_args()

    selected_device = resolve_device(args.device)
    backend = init_backend(selected_device)
    print(f"[INFO] Selected device={selected_device}, backend={backend['label']}")

    if not args.features_root.exists():
        raise FileNotFoundError(f"Features root not found: {args.features_root}")

    arch_dirs = sorted([p for p in args.features_root.iterdir() if p.is_dir()])
    if not arch_dirs:
        raise RuntimeError(f"No architecture folders found in: {args.features_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, float | str]] = []
    for arch_dir in arch_dirs:
        arch_data = load_architecture_data(arch_dir)
        all_rows.extend(run_architecture_benchmark(arch_data, args.random_state, backend))

    results_df = pd.DataFrame(all_rows)
    comparison_csv = args.output_dir / "model_comparison_all.csv"
    results_df.to_csv(comparison_csv, index=False)

    best_df = summarize_best_models(results_df)
    best_csv = args.output_dir / "best_model_per_architecture.csv"
    best_df.to_csv(best_csv, index=False)

    json_path = args.output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "features_root": str(args.features_root),
                "num_architectures": int(results_df["architecture"].nunique()),
                "num_models_per_architecture": 5,
                "results": results_df.to_dict(orient="records"),
                "best_per_architecture": best_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print("\n=== Overall Comparison (all architecture x model rows) ===")
    print(
        results_df.sort_values(
            ["test_accuracy", "val_accuracy"], ascending=False
        )[["architecture", "model", "val_accuracy", "test_accuracy", "test_f1_macro", "fit_time_sec"]]
        .to_string(index=False)
    )

    print("\n=== Best Model Per Architecture (selected by val_accuracy) ===")
    print(best_df.to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {comparison_csv}")
    print(f"- {best_csv}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()
