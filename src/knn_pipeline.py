"""
Production-quality KNN pipeline for time series classification.

- Proper preprocessing pipeline with scaling
- Comprehensive error handling and validation
- Configurable hyperparameters with sensible defaults
- Detailed logging and diagnostics
- Memory-efficient batch processing
- Support for multiple distance metrics
- Per-fold metrics for debugging
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Literal
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class EvalResult:
    """Comprehensive evaluation results."""
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    f1_weighted: float
    f1_macro: float
    report: str
    confusion_mat: np.ndarray
    fold_metrics: List[Dict] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.3f}\n"
            f"F1 (weighted): {self.f1_weighted:.3f}\n"
            f"F1 (macro): {self.f1_macro:.3f}\n"
            f"Time: {self.timing.get('total', 0):.2f}s\n"
        )


@dataclass
class PipelineConfig:
    """Configuration for the KNN pipeline."""
    # Cross-validation
    n_splits: int = 5
    
    # Preprocessing
    standardize: bool = True
    use_pca: bool = True
    n_components: Optional[int] = None  # None = auto-select
    pca_variance_threshold: float = 0.95
    
    # KNN
    k: int = 10
    distance_metric: Literal["euclidean", "dtw"] = "euclidean"
    distance_weighting: bool = True
    
    # DTW-specific
    dtw_window: Optional[float] = 0.1  # Sakoe-Chiba band (fraction)
    dtw_batch_size: int = 100  # For memory efficiency
    
    # Computation
    n_jobs: int = -1
    verbose: int = 1
    
    # Output
    save_predictions: bool = False
    output_dir: Optional[Path] = None


# ============================================================================
# PCA Time Compressor
# ============================================================================


class PCATimeCompressor:
    """Apply PCA independently to each time step."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
    ):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pcas: List[PCA] = []
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.n_components_per_timestep_: Optional[List[int]] = None

    def fit(self, X: np.ndarray) -> "PCATimeCompressor":
        """
        Fit PCA on each time step.

        Args:
            X: Shape (n_samples, n_timesteps, n_features)
        """
        n_samples, n_timesteps, n_features = X.shape
        self.pcas = []
        variance_ratios = []
        n_components_list = []

        for t in range(n_timesteps):
            X_t = X[:, t, :]  # (n_samples, n_features)

            if self.n_components is None:
                # Auto-select to preserve variance
                pca = PCA(n_components=self.variance_threshold, svd_solver='full')
            else:
                pca = PCA(n_components=min(self.n_components, n_features))

            pca.fit(X_t)
            self.pcas.append(pca)
            variance_ratios.append(np.sum(pca.explained_variance_ratio_))
            n_components_list.append(pca.n_components_)

        self.explained_variance_ratio_ = np.array(variance_ratios)
        self.n_components_per_timestep_ = n_components_list
        
        # Use the minimum number of components across all timesteps
        # This ensures consistent output shape
        self.min_components_ = min(n_components_list)
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform using fitted PCAs.

        Args:
            X: Shape (n_samples, n_timesteps, n_features)

        Returns:
            X_transformed: Shape (n_samples, n_timesteps, min_components)
        """
        n_samples, n_timesteps, _ = X.shape
        X_transformed = []

        for t in range(n_timesteps):
            X_t = X[:, t, :]
            X_t_transformed = self.pcas[t].transform(X_t)
            
            # Truncate to min_components to ensure consistent shape
            X_t_transformed = X_t_transformed[:, :self.min_components_]
            X_transformed.append(X_t_transformed)

        return np.stack(X_transformed, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
# ============================================================================
# Distance Computation
# ============================================================================


def dtw_distance(s1: np.ndarray, s2: np.ndarray, window: Optional[int] = None) -> float:
    """
    Compute DTW distance between two sequences with optional Sakoe-Chiba band.
    
    Args:
        s1, s2: Shape (n_timesteps, n_features)
        window: Absolute window size (None = no constraint)
    """
    n, m = len(s1), len(s2)
    
    # Initialize cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Determine window boundaries
    if window is None:
        window = max(n, m)
    
    for i in range(1, n + 1):
        # Sakoe-Chiba band
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        
        for j in range(j_start, j_end):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )
    
    return dtw_matrix[n, m]


def pairwise_dtw_matrix(
    X_test: np.ndarray,
    X_train: np.ndarray,
    window: Optional[float] = None,
    n_jobs: int = 1,
    batch_size: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute pairwise DTW distance matrix with batching for memory efficiency.
    
    Args:
        X_test: Shape (n_test, n_timesteps, n_features)
        X_train: Shape (n_train, n_timesteps, n_features)
        window: Window size as fraction of sequence length
        batch_size: Process test samples in batches
    """
    n_test, n_timesteps, _ = X_test.shape
    n_train = X_train.shape[0]
    
    # Convert window from fraction to absolute
    window_size = None
    if window is not None:
        window_size = max(1, int(window * n_timesteps))
    
    D = np.zeros((n_test, n_train))
    
    # Process in batches to manage memory
    for batch_start in range(0, n_test, batch_size):
        batch_end = min(batch_start + batch_size, n_test)
        
        if verbose:
            print(f"Processing batch {batch_start}-{batch_end}/{n_test}")
        
        for i in range(batch_start, batch_end):
            for j in range(n_train):
                D[i, j] = dtw_distance(X_test[i], X_train[j], window=window_size)
    
    return D


def pairwise_euclidean_matrix(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance on flattened sequences.
    
    Args:
        X_test: Shape (n_test, n_timesteps, n_features)
        X_train: Shape (n_train, n_timesteps, n_features)
    """
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    # Efficient vectorized computation
    D = np.sqrt(
        np.sum(X_test_flat**2, axis=1, keepdims=True) +
        np.sum(X_train_flat**2, axis=1, keepdims=True).T -
        2 * X_test_flat @ X_train_flat.T
    )
    
    return D


# ============================================================================
# KNN Voting
# ============================================================================


def knn_vote_from_dist(
    D: np.ndarray,
    y_train: np.ndarray,
    k: int,
    weighted: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KNN classification from precomputed distance matrix.
    
    Args:
        D: Distance matrix (n_test, n_train)
        y_train: Training labels (n_train,)
        k: Number of neighbors
        weighted: Use distance weighting
    
    Returns:
        predictions: (n_test,)
        probabilities: (n_test, n_classes)
    """
    n_test = D.shape[0]
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    predictions = np.zeros(n_test, dtype=y_train.dtype)
    probabilities = np.zeros((n_test, n_classes))
    
    for i in range(n_test):
        # Find k nearest neighbors
        knn_idx = np.argpartition(D[i], k)[:k]
        knn_distances = D[i, knn_idx]
        knn_labels = y_train[knn_idx]
        
        # Compute weights
        if weighted:
            # Inverse distance weighting (add epsilon to avoid division by zero)
            weights = 1.0 / (knn_distances + 1e-8)
        else:
            weights = np.ones(k)
        
        # Weighted vote
        class_weights = np.zeros(n_classes)
        for cls_idx, cls in enumerate(classes):
            mask = knn_labels == cls
            class_weights[cls_idx] = np.sum(weights[mask])
        
        predictions[i] = classes[np.argmax(class_weights)]
        probabilities[i] = class_weights / np.sum(class_weights)
    
    return predictions, probabilities


# ============================================================================
# Preprocessing Pipeline
# ============================================================================


class TimeSeriesPreprocessor:
    """Preprocessing pipeline for time series data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCATimeCompressor] = None
    
    def fit(self, X: np.ndarray) -> "TimeSeriesPreprocessor":
        """Fit preprocessing on training data."""
        X_processed = X.copy()
        
        # Standardization
        if self.config.standardize:
            self.scaler = StandardScaler()
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X_processed.reshape(-1, n_features)
            X_flat = self.scaler.fit_transform(X_flat)
            X_processed = X_flat.reshape(n_samples, n_timesteps, n_features)
        
        # PCA compression
        if self.config.use_pca:
            self.pca = PCATimeCompressor(
                n_components=self.config.n_components,
                variance_threshold=self.config.pca_variance_threshold,
            )
            self.pca.fit(X_processed)
            
            if self.config.verbose >= 1:
                avg_variance = np.mean(self.pca.explained_variance_ratio_)
                print(f"PCA: Avg {avg_variance:.1%} variance retained per timestep")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessing."""
        X_processed = X.copy()
        
        if self.scaler is not None:
            n_samples, n_timesteps, n_features = X.shape
            X_flat = X_processed.reshape(-1, n_features)
            X_flat = self.scaler.transform(X_flat)
            X_processed = X_flat.reshape(n_samples, n_timesteps, n_features)
        
        if self.pca is not None:
            X_processed = self.pca.transform(X_processed)
        
        return X_processed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================================
# Main Evaluation Function
# ============================================================================


def evaluate_knn(
    X_seq: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Optional[PipelineConfig] = None,
) -> EvalResult:
    """
    Evaluate KNN with cross-validation and comprehensive metrics.
    
    Args:
        X_seq: Time series data (n_samples, n_timesteps, n_features)
        y: Labels (n_samples,)
        groups: Group identifiers for GroupKFold (n_samples,)
        config: Pipeline configuration
    
    Returns:
        EvalResult with detailed metrics
    """
    if config is None:
        config = PipelineConfig()
    
    start_time = time.time()
    
    # Validate inputs
    assert X_seq.shape[0] == len(y) == len(groups), "Mismatched sample sizes"
    assert X_seq.ndim == 3, f"Expected 3D input, got {X_seq.ndim}D"
    
    n_groups = len(np.unique(groups))
    n_splits = min(config.n_splits, n_groups)
    
    if n_groups < config.n_splits:
        warnings.warn(
            f"Only {n_groups} groups available, using {n_splits}-fold CV "
            f"instead of {config.n_splits}"
        )
    
    if config.verbose >= 1:
        print(f"Starting {config.distance_metric.upper()}-KNN evaluation")
        print(f"Data: {X_seq.shape[0]} samples, {n_groups} groups, {n_splits} folds")
        print(f"Config: k={config.k}, standardize={config.standardize}, "
              f"PCA={config.use_pca}")
    
    # Initialize cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    fold_metrics: List[Dict] = []
    
    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_seq, y, groups)):
        fold_start = time.time()
        
        if config.verbose >= 1:
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check for single-class splits
        if len(np.unique(y_train)) < 2:
            warnings.warn(f"Fold {fold}: Single class in training set, skipping")
            continue
        
        # Preprocessing
        preprocessor = TimeSeriesPreprocessor(config)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Compute distances
        if config.distance_metric == "euclidean":
            D = pairwise_euclidean_matrix(X_test_processed, X_train_processed)
        elif config.distance_metric == "dtw":
            D = pairwise_dtw_matrix(
                X_test_processed,
                X_train_processed,
                window=config.dtw_window,
                n_jobs=config.n_jobs,
                batch_size=config.dtw_batch_size,
                verbose=config.verbose >= 2,
            )
        else:
            raise ValueError(f"Unknown distance metric: {config.distance_metric}")
        
        # KNN voting
        y_pred, _ = knn_vote_from_dist(
            D, y_train, k=config.k, weighted=config.distance_weighting
        )
        
        # Store predictions
        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        
        # Fold metrics
        fold_acc = accuracy_score(y_test, y_pred)
        fold_f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        fold_time = time.time() - fold_start
        
        fold_metrics.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": fold_acc,
            "f1_weighted": fold_f1w,
            "time": fold_time,
        })
        
        if config.verbose >= 1:
            print(f"Fold accuracy: {fold_acc:.3f}, F1: {fold_f1w:.3f}, "
                  f"Time: {fold_time:.1f}s")
    
    # Aggregate results
    if not y_true_all:
        raise ValueError("No valid folds - check your data and groups")
    
    y_true_final = np.concatenate(y_true_all)
    y_pred_final = np.concatenate(y_pred_all)
    
    # Compute final metrics
    accuracy = float(accuracy_score(y_true_final, y_pred_final))
    f1_weighted = float(f1_score(y_true_final, y_pred_final, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true_final, y_pred_final, average="macro", zero_division=0))
    report = classification_report(y_true_final, y_pred_final, digits=3, zero_division=0)
    conf_mat = confusion_matrix(y_true_final, y_pred_final)
    
    total_time = time.time() - start_time
    
    if config.verbose >= 1:
        print(f"\n{'='*60}")
        print(f"Final Results ({config.distance_metric.upper()}-KNN, k={config.k})")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"Total time: {total_time:.1f}s")
    
    # Save predictions if requested
    if config.save_predictions and config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({
            "y_true": y_true_final,
            "y_pred": y_pred_final,
        })
        pred_df.to_csv(
            config.output_dir / f"predictions_{config.distance_metric}_k{config.k}.csv",
            index=False
        )
    
    return EvalResult(
        y_true=y_true_final,
        y_pred=y_pred_final,
        accuracy=accuracy,
        f1_weighted=f1_weighted,
        f1_macro=f1_macro,
        report=report,
        confusion_mat=conf_mat,
        fold_metrics=fold_metrics,
        timing={"total": total_time},
        metadata={
            "config": config,
            "n_folds": len(fold_metrics),
            "n_samples": len(y_true_final),
        },
    )


# ============================================================================
# Hyperparameter Search
# ============================================================================


def grid_search_k(
    X_seq: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k_values: List[int],
    base_config: Optional[PipelineConfig] = None,
) -> Dict[int, EvalResult]:
    """
    Grid search over k values.
    
    Returns:
        Dictionary mapping k -> EvalResult
    """
    if base_config is None:
        base_config = PipelineConfig()
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Grid Search: k = {k_values}")
    print(f"{'='*60}")
    
    for k in k_values:
        config = PipelineConfig(**{
            **base_config.__dict__,
            "k": k,
            "verbose": 0,  # Suppress per-fold output
        })
        
        result = evaluate_knn(X_seq, y, groups, config)
        results[k] = result
        
        print(f"\nk={k:2d} | Acc: {result.accuracy:.3f} | "
              f"F1w: {result.f1_weighted:.3f} | F1m: {result.f1_macro:.3f}")
    
    # Find best k
    best_k = max(results.keys(), key=lambda k: results[k].f1_weighted)
    print(f"\nBest k: {best_k} (F1-weighted: {results[best_k].f1_weighted:.3f})")
    
    return results


# ============================================================================
# Windowing Utility
# ============================================================================


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    window: int,
    stride: int,
    group_col: str = "run_id",
    time_col: str = "time",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create sliding windows from time series data.
    
    Args:
        df: DataFrame with columns [feature_cols, label_col, group_col]
        window: Window size
        stride: Stride between windows
        
    Returns:
        X: (n_windows, window, n_features)
        y: (n_windows,) - mode of labels in each window
        groups: (n_windows,) - group identifier
        feature_cols: List of feature column names
    """
    X_windows = []
    y_windows = []
    group_windows = []
    
    for group_id in df[group_col].unique():
        df_group = df[df[group_col] == group_id].sort_values(time_col)
        
        features = df_group[feature_cols].values
        labels = df_group[label_col].values
        
        n_timesteps = len(df_group)
        
        for start in range(0, n_timesteps - window + 1, stride):
            end = start + window
            
            X_win = features[start:end]
            y_win = labels[start:end]
            
            # Use mode of labels in window
            y_mode = np.bincount(y_win).argmax()
            
            X_windows.append(X_win)
            y_windows.append(y_mode)
            group_windows.append(group_id)
    
    return (
        np.array(X_windows),
        np.array(y_windows),
        np.array(group_windows),
        feature_cols,
    )


# ============================================================================
# Demo / Testing
# ============================================================================


def demo_synthetic(
    n_runs: int = 8,
    length: int = 400,
    d: int = 6,
    window: int = 60,
    stride: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic time series data for testing."""
    rng = np.random.default_rng(42)
    records = []
    
    for r in range(n_runs):
        # Generate random walk
        base = rng.normal(0, 1, size=(length, d)).cumsum(axis=0)
        label = np.zeros(length, dtype=int)
        
        # Inject change point for odd runs
        if r % 2 == 1:
            t0 = length // 2
            base[t0:] += rng.normal(3, 0.5, size=(1, d))
            label[t0:] = 1
        
        df_r = pd.DataFrame(base, columns=[f"x{j}" for j in range(d)])
        df_r["run_id"] = r
        df_r["time"] = np.arange(length)
        df_r["label"] = label
        records.append(df_r)
    
    df = pd.concat(records, ignore_index=True)
    feature_cols = [c for c in df.columns if c.startswith("x")]
    
    return make_windows(df, feature_cols, "label", window=window, stride=stride)


if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    X_seq, y, groups, features = demo_synthetic()
    
    print(f"\nData shape: {X_seq.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    print(f"Groups: {len(np.unique(groups))} runs")
    
    # Test 1: Euclidean KNN with grid search
    print("\n" + "="*60)
    print("TEST 1: Euclidean KNN - Grid Search")
    print("="*60)
    
    euclidean_config = PipelineConfig(
        distance_metric="euclidean",
        use_pca=True,
        n_components=4,
        verbose=1,
    )
    
    results_euclidean = grid_search_k(
        X_seq, y, groups,
        k_values=[3, 5, 10, 15],
        base_config=euclidean_config,
    )
    
    # Test 2: DTW KNN with single k
    print("\n" + "="*60)
    print("TEST 2: DTW KNN - Single Run")
    print("="*60)
    
    dtw_config = PipelineConfig(
        distance_metric="dtw",
        use_pca=True,
        n_components=4,
        k=10,
        dtw_window=0.1,
        dtw_batch_size=50,
        verbose=1,
    )
    
    result_dtw = evaluate_knn(X_seq, y, groups, dtw_config)
    
    print("\n" + "="*60)
    print("Classification Report (DTW):")
    print("="*60)
    print(result_dtw.report)
    
    print("\nConfusion Matrix (DTW):")
    print(result_dtw.confusion_mat)
    
    # Test 3: Compare preprocessing options
    print("\n" + "="*60)
    print("TEST 3: Ablation Study - Preprocessing")
    print("="*60)
    
    configs_to_test = [
        ("No preprocessing", PipelineConfig(
            standardize=False, use_pca=False, k=10, verbose=0
        )),
        ("Standardize only", PipelineConfig(
            standardize=True, use_pca=False, k=10, verbose=0
        )),
        ("PCA only", PipelineConfig(
            standardize=False, use_pca=True, n_components=4, k=10, verbose=0
        )),
        ("Full pipeline", PipelineConfig(
            standardize=True, use_pca=True, n_components=4, k=10, verbose=0
        )),
    ]
    
    for name, config in configs_to_test:
        result = evaluate_knn(X_seq, y, groups, config)
        print(f"{name:20s} | Acc: {result.accuracy:.3f} | "
              f"F1w: {result.f1_weighted:.3f} | Time: {result.timing['total']:.1f}s")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
