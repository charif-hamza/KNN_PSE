"""Command line utilities for running the CSTH KNN pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .config import PipelineConfig
from .pipeline import (
    EvalResult,
    TimeSeriesPreprocessor,
    batched_lower_triangular_euclidean,
    compute_run_manifest,
    evaluate_knn,
    grid_search_k,
    knn_vote_from_dist,
    pairwise_distance_matrix,
)


def load_csth_split(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSTH split from disk and validate its contents."""

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        data = torch.load(filepath, map_location="cpu", weights_only=True)
    except TypeError:  # weights_only introduced in torch 2.0
        data = torch.load(filepath, map_location="cpu")

    if not isinstance(data, dict) or "samples" not in data or "labels" not in data:
        raise ValueError(
            f"File {filepath} does not contain the expected dictionary format"
        )

    X = data["samples"]
    y = data["labels"]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    if X.ndim != 3:
        raise ValueError(f"Expected shape (N, T, F) for samples, got {X.shape}")
    y = y.reshape(-1).astype(np.int64)
    if len(X) != len(y):
        raise ValueError(f"Mismatch between samples and labels: {len(X)} vs {len(y)}")

    unique = np.unique(y)
    if not np.all((unique >= 0) & (unique <= 1)):
        raise ValueError(f"Labels are expected to be binary (0/1); observed {unique}")
    return X, y


def load_csth_dataset(data_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load train/validation/test splits for the CSTH dataset."""

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        X, y = load_csth_split(data_dir / f"{split}.pt")
        splits[split] = (X, y)
    return splits


def create_groups_for_split(
    n_samples: int,
    n_groups: int | None = None,
    group_size: int | None = None,
) -> np.ndarray:
    """Create synthetic group labels for GroupKFold evaluation."""

    if n_groups is not None:
        target_groups = max(2, min(n_groups, n_samples))
    elif group_size is not None:
        target_groups = max(2, n_samples // max(group_size, 1))
    else:
        target_groups = max(5, min(10, n_samples // 50 or 1))

    groups = np.repeat(np.arange(target_groups), n_samples // target_groups + 1)[
        :n_samples
    ]
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_samples)
    return groups[np.argsort(perm)]


def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    n_groups: int | None = None,
) -> EvalResult:
    """Run cross-validation using automatically generated groups."""

    groups = create_groups_for_split(len(y), n_groups=n_groups)
    return evaluate_knn(X, y, groups, config)


def evaluate_train_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Train on the train split and evaluate on the test split."""

    start = time.time()
    preprocessor = TimeSeriesPreprocessor(config)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if config.precompute_train_distances and config.distance_metric == "euclidean":
        batched_lower_triangular_euclidean(
            X_train_proc,
            batch_size=max(64, config.dtw_batch_size),
            backend=config.euclidean_backend,
            use_gpu=config.use_gpu,
            memmap_path=config.memmap_path,
        )

    distances, neighbour_idx = pairwise_distance_matrix(
        X_test_proc, X_train_proc, config
    )
    y_pred, y_proba = knn_vote_from_dist(
        distances,
        y_train,
        k=config.k,
        weighted=config.distance_weighting,
        neighbour_indices=neighbour_idx,
    )

    metrics: dict[str, Any] = {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "time": time.time() - start,
    }

    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["f1_weighted"] = float(
        f1_score(y_test, y_pred, average="weighted", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_test, y_pred, average="macro", zero_division=0)
    )

    positive_scores = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, positive_scores))
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["pr_auc"] = float(average_precision_score(y_test, positive_scores))
    except ValueError:
        metrics["pr_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, digits=3, zero_division=0
    )

    return metrics


def _data_signature(array: np.ndarray, max_samples: int = 512) -> str:
    """Create a lightweight hash for provenance tracking."""

    subset = array.reshape(array.shape[0], -1)[:max_samples].astype(
        np.float32, copy=False
    )
    hasher = hashlib.sha256()
    hasher.update(subset.tobytes())
    return hasher.hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run CSTH KNN experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--mode", choices=["quick", "search", "final", "all"], default="quick"
    )
    parser.add_argument("--best-k", type=int, default=25)
    parser.add_argument("--distance", choices=["euclidean", "dtw"], default="euclidean")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use PyTorch GPU backend when available"
    )
    parser.add_argument(
        "--ann", choices=["annoy", "faiss"], help="Use ANN backend for Euclidean KNN"
    )
    parser.add_argument(
        "--dtw-backend",
        choices=["python", "numba", "fastdtw", "dtaidistance"],
        default="python",
    )
    parser.add_argument("--threshold-grid", nargs="*", type=float, default=None)
    return parser


def configure_pipeline(args: argparse.Namespace) -> PipelineConfig:
    """Construct a :class:`PipelineConfig` from CLI arguments."""

    threshold_grid = args.threshold_grid or [i / 20 for i in range(1, 20)]
    return PipelineConfig(
        distance_metric=args.distance,
        use_gpu=args.use_gpu,
        ann_backend=args.ann,
        dtw_backend=args.dtw_backend,
        threshold_grid=threshold_grid,
    )


def summarise_result(result: EvalResult) -> None:
    """Pretty-print a cross-validation result."""

    print("\n=== Aggregate Metrics ===")
    print(result)
    print("\nPer-class F1:")
    for cls, score in result.per_class_f1.items():
        print(f"  Class {cls}: {score:.3f}")
    if result.roc_auc is not None:
        print(f"ROC-AUC: {result.roc_auc:.3f}")
    if result.pr_auc is not None:
        print(f"PR-AUC: {result.pr_auc:.3f}")
    print("\nConfusion Matrix:\n", result.confusion_mat)


def run_mode_quick(args: argparse.Namespace) -> None:
    """Run the fast sanity-check cross-validation."""

    splits = load_csth_dataset(args.data_dir)
    X_val, y_val = splits["val"]
    config = configure_pipeline(args)
    result = evaluate_with_cv(X_val, y_val, config, n_groups=10)
    summarise_result(result)


def run_mode_search(args: argparse.Namespace) -> None:
    """Grid search over several ``k`` values."""

    splits = load_csth_dataset(args.data_dir)
    X_val, y_val = splits["val"]
    config = configure_pipeline(args)
    groups = create_groups_for_split(len(y_val))
    results = grid_search_k(X_val, y_val, groups, [3, 5, 10, 15, 25], config)
    for k, result in results.items():
        print(f"k={k}: accuracy={result.accuracy:.3f} | F1w={result.f1_weighted:.3f}")


def run_mode_final(args: argparse.Namespace) -> None:
    """Train on train+val and evaluate on test."""

    splits = load_csth_dataset(args.data_dir)
    X_train = np.concatenate([splits["train"][0], splits["val"][0]])
    y_train = np.concatenate([splits["train"][1], splits["val"][1]])
    X_test, y_test = splits["test"]

    config = configure_pipeline(args)
    config.k = args.best_k
    config.output_dir = args.output_dir

    result = evaluate_train_test(X_train, y_train, X_test, y_test, config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                k: v
                for k, v in result.items()
                if k not in {"y_true", "y_pred", "y_proba"}
            },
            handle,
            indent=2,
        )
    np.save(output_dir / "test_predictions.npy", result["y_pred"])
    np.save(output_dir / "test_probabilities.npy", result["y_proba"])
    timing = {"total": result["time"]}
    data_signature = _data_signature(X_train)
    compute_run_manifest(
        config,
        metrics={
            k: v
            for k, v in result.items()
            if isinstance(v, int | float) and v is not None
        },
        timing=timing,
        data_signature=data_signature,
        manifest_path=config.manifest_path,
    )


def run_mode_all(args: argparse.Namespace) -> None:
    """Run quick validation, grid search, and final evaluation sequentially."""

    run_mode_quick(args)
    run_mode_search(args)
    run_mode_final(args)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the console script."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "quick":
        run_mode_quick(args)
    elif args.mode == "search":
        run_mode_search(args)
    elif args.mode == "final":
        run_mode_final(args)
    elif args.mode == "all":
        run_mode_all(args)
    else:  # pragma: no cover - argparse prevents this path
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main(sys.argv[1:])
