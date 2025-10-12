"""Core pipeline primitives for the CSTH fault-detection KNN workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .metrics import ThresholdSweep

try:  # Optional GPU acceleration
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None

try:  # Optional DTW accelerators
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None

try:  # Optional ANN search
    from annoy import AnnoyIndex
except Exception:  # pragma: no cover - optional dependency
    AnnoyIndex = None

try:  # pragma: no cover - optional dependency
    from fastdtw import fastdtw
except Exception:  # pragma: no cover - optional dependency
    fastdtw = None

try:  # pragma: no cover - optional dependency
    from dtaidistance import dtw as dtaid_dtw
except Exception:  # pragma: no cover - optional dependency
    dtaid_dtw = None


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(slots=True)
class EvalResult:
    """Container returned by :func:`evaluate_knn`."""

    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray | None
    accuracy: float
    f1_weighted: float
    f1_macro: float
    roc_auc: float | None
    pr_auc: float | None
    per_class_f1: dict[str, float]
    report: str
    confusion_mat: np.ndarray
    threshold_sweep: ThresholdSweep
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None

    def __str__(self) -> str:  # pragma: no cover - presentation helper
        lines = [
            f"Accuracy: {self.accuracy:.3f}",
            f"F1 (weighted): {self.f1_weighted:.3f}",
            f"F1 (macro): {self.f1_macro:.3f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"ROC-AUC: {self.roc_auc:.3f}")
        if self.pr_auc is not None:
            lines.append(f"PR-AUC: {self.pr_auc:.3f}")
        total_time = self.timing.get("total")
        if total_time is not None:
            lines.append(f"Time: {total_time:.2f}s")
        return "\n".join(lines)


# ============================================================================
# PCA Time Compressor
# ============================================================================


class PCATimeCompressor:
    """Fit independent PCA models per time step while enforcing common shape."""

    def __init__(
        self,
        n_components: int | None = None,
        variance_threshold: float = 0.95,
    ) -> None:
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pcas: list[PCA] = []
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_per_timestep_: list[int] | None = None
        self.min_components_: int | None = None

    def fit(self, X: np.ndarray) -> PCATimeCompressor:
        """Fit PCA transformers for each time step."""

        n_samples, n_timesteps, n_features = X.shape
        self.pcas = []
        variance_ratios = []
        components = []

        for idx in range(n_timesteps):
            X_t = X[:, idx, :]
            if self.n_components is None:
                pca = PCA(n_components=self.variance_threshold, svd_solver="full")
            else:
                pca = PCA(n_components=min(self.n_components, n_features))
            pca.fit(X_t)
            self.pcas.append(pca)
            variance_ratios.append(float(np.sum(pca.explained_variance_ratio_)))
            components.append(pca.n_components_)

        self.explained_variance_ratio_ = np.asarray(variance_ratios)
        self.n_components_per_timestep_ = components
        self.min_components_ = int(min(components))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform samples using the fitted PCA models."""

        if self.min_components_ is None:
            raise RuntimeError("PCATimeCompressor must be fitted before use")

        transformed = []
        for idx, pca in enumerate(self.pcas):
            X_t = pca.transform(X[:, idx, :])
            transformed.append(X_t[:, : self.min_components_])
        return np.stack(transformed, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and immediately transform the same data."""

        return self.fit(X).transform(X)


# ============================================================================
# Distance Computation
# ============================================================================


def _python_dtw(s1: np.ndarray, s2: np.ndarray, window: int | None) -> float:
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    if window is None:
        window = max(n, m)

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = float(np.linalg.norm(s1[i - 1] - s2[j - 1]))
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )
    return float(dtw_matrix[n, m])


if njit is not None:  # pragma: no cover - exercised if numba available

    @njit(cache=True)
    def _numba_dtw(s1: np.ndarray, s2: np.ndarray, window: int) -> float:
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0
        for i in range(1, n + 1):
            j_start = 1 if i - window < 1 else i - window
            j_end = m + 1 if i + window + 1 > m + 1 else i + window + 1
            for j in range(j_start, j_end):
                cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )
        return float(dtw_matrix[n, m])

else:  # pragma: no cover - ensures symbol exists

    def _numba_dtw(s1: np.ndarray, s2: np.ndarray, window: int) -> float:
        raise RuntimeError("Numba backend requested but numba is not installed")


def dtw_distance(
    s1: np.ndarray,
    s2: np.ndarray,
    window: int | None = None,
    backend: str = "python",
) -> float:
    """Compute DTW distance with several optional backends."""

    if backend == "python":
        return _python_dtw(s1, s2, window)
    if backend == "numba":
        if window is None:
            window = max(len(s1), len(s2))
        return _numba_dtw(s1, s2, window)
    if backend == "fastdtw":
        if fastdtw is None:
            raise RuntimeError("fastdtw backend requested but fastdtw is missing")
        distance, _ = fastdtw(s1, s2, radius=window or 1)
        return float(distance)
    if backend == "dtaidistance":
        if dtaid_dtw is None:
            raise RuntimeError("dtaidistance backend requested but not installed")
        return float(dtaid_dtw.distance_fast(s1, s2, window=window))
    raise ValueError(f"Unknown DTW backend: {backend}")


def pairwise_dtw_matrix(
    X_test: np.ndarray,
    X_train: np.ndarray,
    window: float | None = None,
    batch_size: int = 64,
    backend: str = "python",
    verbose: bool = False,
) -> np.ndarray:
    """Compute pairwise DTW distances between test and train samples."""

    n_test, n_timesteps, _ = X_test.shape
    window_size = None
    if window is not None:
        window_size = max(1, int(window * n_timesteps))

    distances = np.zeros((n_test, X_train.shape[0]), dtype=float)
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        if verbose:  # pragma: no cover - logging only
            print(f"DTW batch {start}:{end} of {n_test}")
        for i in range(start, end):
            for j in range(X_train.shape[0]):
                distances[i, j] = dtw_distance(
                    X_test[i], X_train[j], window=window_size, backend=backend
                )
    return distances


def _euclidean_numpy(block_a: np.ndarray, block_b: np.ndarray) -> np.ndarray:
    a_sq = np.sum(block_a**2, axis=1, keepdims=True)
    b_sq = np.sum(block_b**2, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(a_sq + b_sq - 2.0 * block_a @ block_b.T, 0.0))


def _euclidean_torch(
    block_a: np.ndarray,
    block_b: np.ndarray,
    use_gpu: bool,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch euclidean backend")
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        tensor_a = torch.from_numpy(block_a).to(device=device, dtype=torch.float32)
        tensor_b = torch.from_numpy(block_b).to(device=device, dtype=torch.float32)
        dist = torch.cdist(tensor_a, tensor_b)
        return dist.cpu().numpy()


def pairwise_euclidean_matrix(
    X_test: np.ndarray,
    X_train: np.ndarray,
    backend: str = "numpy",
    use_gpu: bool = False,
) -> np.ndarray:
    """Compute pairwise Euclidean distances with selectable backend."""

    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    if backend == "numpy":
        return _euclidean_numpy(X_test_flat, X_train_flat)
    if backend == "torch":
        return _euclidean_torch(X_test_flat, X_train_flat, use_gpu)
    raise ValueError(f"Unknown euclidean backend: {backend}")


def batched_lower_triangular_euclidean(
    X: np.ndarray,
    batch_size: int = 256,
    backend: str = "numpy",
    use_gpu: bool = False,
    memmap_path: Path | None = None,
) -> np.ndarray:
    """Compute a full pairwise distance matrix using lower-triangular batching."""

    flattened = X.reshape(X.shape[0], -1)
    n_samples = flattened.shape[0]
    if memmap_path is not None:
        memmap_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.memmap(
            memmap_path, mode="w+", dtype=np.float32, shape=(n_samples, n_samples)
        )
    else:
        matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        block = flattened[start:end]
        if backend == "numpy":
            distances = _euclidean_numpy(block, flattened[:end])
        elif backend == "torch":
            distances = _euclidean_torch(block, flattened[:end], use_gpu)
        else:
            raise ValueError(f"Unknown euclidean backend: {backend}")
        matrix[start:end, :end] = distances
        matrix[:end, start:end] = distances.T
    return matrix


# ============================================================================
# ANN search (optional)
# ============================================================================


def compute_ann_neighbors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
    backend: str | None,
    n_trees: int = 50,
    search_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate nearest neighbours for the Euclidean metric."""

    if backend is None:
        raise ValueError("ANN backend must be provided")

    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    if backend == "annoy":
        if AnnoyIndex is None:
            raise RuntimeError("Annoy backend requested but annoy is not installed")
        index = AnnoyIndex(X_train_flat.shape[1], metric="euclidean")
        for idx, vector in enumerate(X_train_flat):
            index.add_item(idx, vector.tolist())
        index.build(n_trees)
        indices = np.zeros((X_test_flat.shape[0], k), dtype=int)
        distances = np.zeros((X_test_flat.shape[0], k), dtype=float)
        query_search_k = search_k or (n_trees * k)
        for i, vector in enumerate(X_test_flat):
            neighbours, dists = index.get_nns_by_vector(
                vector.tolist(), k, include_distances=True, search_k=query_search_k
            )
            indices[i, :] = neighbours
            distances[i, :] = dists
        return distances, indices
    if backend == "faiss":  # pragma: no cover - faiss rarely available in CI
        import faiss  # type: ignore

        d = X_train_flat.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(X_train_flat)
        distances, indices = index.search(X_test_flat, k)
        return np.sqrt(distances), indices
    raise ValueError(f"Unknown ANN backend: {backend}")


# ============================================================================
# KNN Voting
# ============================================================================


def knn_vote_from_dist(
    distances: np.ndarray,
    y_train: np.ndarray,
    k: int,
    weighted: bool = True,
    neighbour_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert distances to predictions and probabilities."""

    if neighbour_indices is None and distances.shape[1] < k:
        raise ValueError("Distance matrix must have at least k columns")

    classes = np.unique(y_train)
    n_classes = len(classes)
    if neighbour_indices is None:
        # Use the first k columns (after partial sort for efficiency)
        sorted_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        neighbour_indices = sorted_indices
        neighbour_distances = np.take_along_axis(distances, sorted_indices, axis=1)
    else:
        neighbour_distances = distances[:, :k]
        neighbour_indices = neighbour_indices[:, :k]

    predictions = np.zeros(distances.shape[0], dtype=y_train.dtype)
    probabilities = np.zeros((distances.shape[0], n_classes), dtype=float)

    for row in range(distances.shape[0]):
        labels = y_train[neighbour_indices[row]]
        dists = neighbour_distances[row]
        if weighted:
            weights = 1.0 / (dists + 1e-8)
        else:
            weights = np.ones_like(dists)
        class_weights = np.zeros(n_classes, dtype=float)
        for idx, cls in enumerate(classes):
            mask = labels == cls
            class_weights[idx] = float(np.sum(weights[mask]))
        class_weights_sum = class_weights.sum()
        if class_weights_sum > 0:
            class_weights /= class_weights_sum
        predictions[row] = classes[int(np.argmax(class_weights))]
        probabilities[row] = class_weights
    return predictions, probabilities


# ============================================================================
# Preprocessing
# ============================================================================


class TimeSeriesPreprocessor:
    """Standardisation and PCA pipeline for time-series arrays."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.scaler: StandardScaler | None = None
        self.pca: PCATimeCompressor | None = None

    def fit(self, X: np.ndarray) -> TimeSeriesPreprocessor:
        X_processed = X.copy()
        if self.config.standardize:
            self.scaler = StandardScaler()
            n_samples, _, n_features = X_processed.shape
            X_flat = X_processed.reshape(-1, n_features)
            X_flat = self.scaler.fit_transform(X_flat)
            X_processed = X_flat.reshape(n_samples, -1, n_features)
        if self.config.use_pca:
            self.pca = PCATimeCompressor(
                n_components=self.config.n_components,
                variance_threshold=self.config.pca_variance_threshold,
            )
            self.pca.fit(X_processed)
            if self.config.verbose >= 1:
                if self.pca.explained_variance_ratio_ is not None:
                    retained = float(np.mean(self.pca.explained_variance_ratio_))
                else:
                    retained = 0.0
                print(f"PCA retained {retained:.1%} variance on average")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_processed = X.copy()
        if self.scaler is not None:
            n_samples, _, n_features = X_processed.shape
            X_flat = X_processed.reshape(-1, n_features)
            X_flat = self.scaler.transform(X_flat)
            X_processed = X_flat.reshape(n_samples, -1, n_features)
        if self.pca is not None:
            X_processed = self.pca.transform(X_processed)
        return X_processed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================================
# Utility helpers
# ============================================================================


def pairwise_distance_matrix(
    X_test: np.ndarray,
    X_train: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute the distance matrix respecting configuration options."""

    if config.distance_metric == "euclidean":
        if config.ann_backend is not None:
            distances, indices = compute_ann_neighbors(
                X_train,
                X_test,
                k=max(config.k, 1),
                backend=config.ann_backend,
                n_trees=config.ann_n_trees,
                search_k=config.ann_search_k,
            )
            return distances, indices
        distances = pairwise_euclidean_matrix(
            X_test, X_train, backend=config.euclidean_backend, use_gpu=config.use_gpu
        )
        return distances, None
    if config.distance_metric == "dtw":
        distances = pairwise_dtw_matrix(
            X_test,
            X_train,
            window=config.dtw_window,
            batch_size=config.dtw_batch_size,
            backend=config.dtw_backend,
            verbose=config.verbose >= 2,
        )
        return distances, None
    raise ValueError(f"Unsupported distance metric: {config.distance_metric}")


def _data_signature(X: np.ndarray, max_samples: int = 512) -> str:
    """Create a deterministic hash for a dataset sample."""

    subset = X[:max_samples].astype(np.float32, copy=False)
    hasher = hashlib.sha256()
    hasher.update(subset.tobytes())
    return hasher.hexdigest()


def _normalise_for_json(value: Any) -> Any:
    """Recursively convert objects into JSON-serialisable types."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalise_for_json(val) for key, val in value.items()}
    if isinstance(value, list | tuple | set):
        return [_normalise_for_json(item) for item in value]
    return value


def compute_run_manifest(
    config: PipelineConfig,
    metrics: dict[str, Any],
    timing: dict[str, float],
    data_signature: str,
    manifest_path: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a JSON manifest describing a single evaluation run."""

    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd())
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - git not always available in CI
        git_sha = "unknown"

    manifest: dict[str, Any] = {
        "git_sha": git_sha,
        "config": _normalise_for_json(asdict(config)),
        "metrics": _normalise_for_json(metrics),
        "timing": _normalise_for_json(timing),
        "data_signature": data_signature,
    }
    if extra_metadata:
        manifest.update(extra_metadata)

    target_dir = manifest_path.parent if manifest_path else config.output_dir
    if target_dir is None:
        target_dir = Path("results")
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = (
        manifest_path
        if manifest_path is not None
        else target_dir / f"manifest_{int(time.time())}.json"
    )
    with open(manifest_file, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_file


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_knn(
    X_seq: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: PipelineConfig | None = None,
) -> EvalResult:
    """Evaluate the configured pipeline using GroupKFold cross-validation."""

    if config is None:
        config = PipelineConfig()

    start_time = time.time()
    if X_seq.shape[0] != len(y) or len(y) != len(groups):
        raise ValueError("Mismatched sample counts")
    if X_seq.ndim != 3:
        raise ValueError("X_seq must be a 3D array")

    n_groups = len(np.unique(groups))
    n_splits = min(config.n_splits, n_groups)
    if n_groups < config.n_splits:
        warnings.warn(
            f"Only {n_groups} groups available. Using {n_splits} folds instead of "
            f"{config.n_splits}",
            stacklevel=2,
        )

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: list[dict[str, Any]] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_proba_all: list[np.ndarray] = []

    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X_seq, y, groups)):
        fold_start = time.time()
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            warnings.warn(
                f"Skipping fold {fold_id}: only one class present", stacklevel=2
            )
            continue

        preprocessor = TimeSeriesPreprocessor(config)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        distances, neighbour_idx = pairwise_distance_matrix(
            X_test_proc, X_train_proc, config
        )
        y_pred_fold, y_proba_fold = knn_vote_from_dist(
            distances,
            y_train,
            k=config.k,
            weighted=config.distance_weighting,
            neighbour_indices=neighbour_idx,
        )

        y_true_all.append(y_test)
        y_pred_all.append(y_pred_fold)
        y_proba_all.append(y_proba_fold)

        fold_time = time.time() - fold_start
        fold_metrics.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, y_pred_fold)),
                "f1_weighted": float(
                    f1_score(y_test, y_pred_fold, average="weighted", zero_division=0)
                ),
                "time": fold_time,
            }
        )

    if not y_true_all:
        raise ValueError("Cross-validation produced no valid folds")

    y_true_final = np.concatenate(y_true_all)
    y_pred_final = np.concatenate(y_pred_all)
    y_proba_final = np.concatenate(y_proba_all)

    metrics = _compute_summary_metrics(
        y_true_final, y_pred_final, y_proba_final, config.threshold_grid
    )

    total_time = time.time() - start_time
    timing = {"total": total_time}

    manifest_path: Path | None = None
    if config.record_manifest:
        data_sig = _data_signature(X_seq)
        manifest_path = compute_run_manifest(
            config,
            metrics={
                key: float(value)
                for key, value in metrics.items()
                if key in {"accuracy", "f1_weighted", "f1_macro", "roc_auc", "pr_auc"}
                and value is not None
            },
            timing=timing,
            data_signature=data_sig,
            manifest_path=config.manifest_path,
            extra_metadata=config.metadata,
        )

    return EvalResult(
        y_true=y_true_final,
        y_pred=y_pred_final,
        probabilities=y_proba_final,
        accuracy=float(metrics["accuracy"]),
        f1_weighted=float(metrics["f1_weighted"]),
        f1_macro=float(metrics["f1_macro"]),
        roc_auc=metrics.get("roc_auc"),
        pr_auc=metrics.get("pr_auc"),
        per_class_f1=metrics["per_class_f1"],
        report=metrics["report"],
        confusion_mat=metrics["confusion_matrix"],
        threshold_sweep=metrics["threshold_sweep"],
        fold_metrics=fold_metrics,
        timing=timing,
        metadata=config.metadata,
        manifest_path=manifest_path,
    )


def _compute_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    unique_labels = np.unique(y_true)
    per_class_f1 = {
        str(cls): float(score)
        for cls, score in zip(unique_labels, per_class, strict=False)
    }

    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        positive_scores = y_proba[:, 1]
    else:
        positive_scores = y_proba.ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, positive_scores))
    except ValueError:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, positive_scores))
    except ValueError:
        pr_auc = None

    sweep = ThresholdSweep.from_probabilities(y_true, y_proba, thresholds)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "per_class_f1": per_class_f1,
        "threshold_sweep": sweep,
        "report": report,
        "confusion_matrix": conf_mat,
    }


# ============================================================================
# Grid search
# ============================================================================


def grid_search_k(
    X_seq: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k_values: Sequence[int],
    base_config: PipelineConfig | None = None,
) -> dict[int, EvalResult]:
    """Evaluate multiple ``k`` values."""

    if base_config is None:
        base_config = PipelineConfig()

    results: dict[int, EvalResult] = {}
    for k in k_values:
        config = replace(base_config, k=int(k))
        config.metadata = {**base_config.metadata, "k": int(k)}
        result = evaluate_knn(X_seq, y, groups, config)
        results[int(k)] = result
    return results


# ============================================================================
# Windowing utility and synthetic demo
# ============================================================================


def make_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    window: int,
    stride: int,
    group_col: str = "run_id",
    time_col: str = "time",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Create sliding windows suitable for the pipeline input."""

    X_windows: list[np.ndarray] = []
    y_windows: list[int] = []
    groups: list[Any] = []

    for group_id, df_group in df.groupby(group_col):
        df_sorted = df_group.sort_values(time_col)
        features = df_sorted[list(feature_cols)].to_numpy()
        labels = df_sorted[label_col].to_numpy()
        for start in range(0, len(df_sorted) - window + 1, stride):
            end = start + window
            X_windows.append(features[start:end])
            window_labels = labels[start:end]
            y_windows.append(int(np.bincount(window_labels).argmax()))
            groups.append(group_id)
    return (
        np.stack(X_windows),
        np.asarray(y_windows, dtype=int),
        np.asarray(groups),
        list(feature_cols),
    )


def demo_synthetic(
    n_runs: int = 8,
    length: int = 400,
    n_features: int = 6,
    window: int = 60,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate a synthetic dataset for smoke testing."""

    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []
    for run in range(n_runs):
        base = rng.normal(0.0, 1.0, size=(length, n_features)).cumsum(axis=0)
        label = np.zeros(length, dtype=int)
        if run % 2 == 1:
            switch = length // 2
            base[switch:] += rng.normal(3.0, 0.5, size=(1, n_features))
            label[switch:] = 1
        df = pd.DataFrame(base, columns=[f"x{idx}" for idx in range(n_features)])
        df["run_id"] = run
        df["time"] = np.arange(length)
        df["label"] = label
        frames.append(df)
    dataset = pd.concat(frames, ignore_index=True)
    feature_cols = [col for col in dataset.columns if col.startswith("x")]
    return make_windows(dataset, feature_cols, "label", window=window, stride=stride)


__all__ = [
    "EvalResult",
    "PCATimeCompressor",
    "TimeSeriesPreprocessor",
    "batched_lower_triangular_euclidean",
    "compute_ann_neighbors",
    "compute_run_manifest",
    "demo_synthetic",
    "dtw_distance",
    "evaluate_knn",
    "grid_search_k",
    "knn_vote_from_dist",
    "make_windows",
    "pairwise_distance_matrix",
]
