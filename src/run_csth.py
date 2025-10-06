"""
Runner script for CSTH (Continuous Stirred Tank Heater) dataset with KNN pipeline.

Dataset details:
- Source: CSTH simulation model (https://zenodo.org/records/10093059)
- Application: Time series classification and fault detection/diagnosis (FDD)
- Process: Heating system with hot/cold water mixing, steam heating, closed-loop control
- Train: 6300 samples (binary classification)
- Val:   900 samples
- Test:  1800 samples
- Actual shape: (N, T=3, F=200) where T=timesteps (3), F=features (200)
  * 3 timesteps representing key process states
  * 200 features derived from process variables (cold water flow, tank level, temperature)
- Data is normalized to [0, 1]
- Binary classification: 
  * Y=0: Normal operating conditions (50% of data)
  * Y=1: Faulty scenarios - instrumentation errors (50% of data)

Usage:
    python src/run_csth.py --mode quick
    python src/run_csth.py --mode search
    python src/run_csth.py --mode final --best-k 10
    python src/run_csth.py --mode all
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import time
import warnings

import numpy as np
import torch
import pandas as pd

# Import from sibling module
try:
    from knn_pipeline import (
        evaluate_knn,
        grid_search_k,
        PipelineConfig,
        EvalResult,
        TimeSeriesPreprocessor,
        pairwise_euclidean_matrix,
        pairwise_dtw_matrix,
        knn_vote_from_dist,
    )
except ImportError:
    # Fallback: add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    from knn_pipeline import (
        evaluate_knn,
        grid_search_k,
        PipelineConfig,
        EvalResult,
        TimeSeriesPreprocessor,
        pairwise_euclidean_matrix,
        pairwise_dtw_matrix,
        knn_vote_from_dist,
    )

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================================
# Data Loading with Validation
# ============================================================================


def load_csth_split(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CSTH data split with validation.
    
    Args:
        filepath: Path to .pt file
        
    Returns:
        X: (N, T, F) array - N samples, T timesteps, F features (process variables)
        y: (N,) array - 0=normal operation, 1=fault condition
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data has unexpected format
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=True)
    except Exception as e:
        # Fallback for older PyTorch versions
        data = torch.load(filepath, map_location='cpu')
    
    # Extract samples and labels
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")
    
    if 'samples' not in data or 'labels' not in data:
        raise ValueError(f"Missing 'samples' or 'labels' keys. Found: {data.keys()}")
    
    X = data['samples']
    y = data['labels']
    
    # Convert to numpy
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    
    # Validate shape
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N, T, F), got shape {X.shape}")
    
    if len(y.shape) != 1:
        y = y.flatten()
    
    if len(X) != len(y):
        raise ValueError(f"Sample count mismatch: X={len(X)}, y={len(y)}")
    
    # Ensure labels are integers
    y = y.astype(np.int64)
    
    # Validate label range (binary: 0=normal, 1=fault)
    unique_labels = np.unique(y)
    if not np.all((unique_labels >= 0) & (unique_labels <= 1)):
        warnings.warn(f"Expected binary labels (0=normal, 1=fault), found: {unique_labels}")
    
    return X, y


def load_csth_dataset(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all CSTH splits with comprehensive validation.
    
    Returns:
        Dictionary with keys 'train', 'val', 'test'
        Each value is (X, y) tuple where:
        - X: (N, T, F) - time series data
        - y: (N,) - labels (0=normal, 1=fault)
    """
    print("\n" + "="*70)
    print("Loading CSTH Dataset (Continuous Stirred Tank Heater)")
    print("="*70)
    
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        filepath = data_dir / f'{split_name}.pt'
        
        print(f"\nLoading {split_name}...")
        X, y = load_csth_split(filepath)
        splits[split_name] = (X, y)
        
        # Report statistics
        class_counts = np.bincount(y)
        class_labels = {0: 'normal', 1: 'fault'}
        print(f"  Shape: {X.shape} (samples x timesteps x features)")
        print(f"  Classes: {class_counts[0]} normal, {class_counts[1]} fault")
        print(f"  Data range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Data mean: {X.mean():.3f}, std: {X.std():.3f}")
        
        # Verify expected dimensionality
        n_samples, n_timesteps, n_features = X.shape
        if n_features != 3:
            warnings.warn(f"Expected 3 features (cold water flow, tank level, temperature), "
                        f"got {n_features}")
        if n_timesteps != 200:
            warnings.warn(f"Expected 200 timesteps, got {n_timesteps}")
    
    # Cross-split validation
    shapes = [splits[s][0].shape for s in ['train', 'val', 'test']]
    if not all(s[1:] == shapes[0][1:] for s in shapes):
        warnings.warn(f"Inconsistent shapes across splits: {shapes}")
    
    print("\n" + "="*70)
    
    return splits


def create_groups_for_split(
    n_samples: int,
    n_groups: Optional[int] = None,
    group_size: Optional[int] = None
) -> np.ndarray:
    """
    Create artificial groups for cross-validation.
    
    Since CSTH doesn't have explicit run_ids, we create groups by
    dividing samples into chunks. This simulates different experimental
    runs or operating conditions.
    
    Args:
        n_samples: Number of samples
        n_groups: Target number of groups (overrides group_size)
        group_size: Approximate size of each group
        
    Returns:
        groups: (n_samples,) array of group IDs
    """
    if n_groups is not None:
        # Use specified number of groups
        target_groups = max(5, min(n_groups, n_samples // 5))
    elif group_size is not None:
        # Calculate groups from size
        target_groups = max(5, n_samples // group_size)
    else:
        # Default: aim for ~10 groups for balanced CV
        target_groups = max(5, min(10, n_samples // 50))
    
    # Create evenly distributed groups
    groups = np.repeat(np.arange(target_groups), n_samples // target_groups + 1)[:n_samples]
    
    # Shuffle to avoid sequential bias
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    groups = groups[np.argsort(perm)]
    
    return groups


# ============================================================================
# Evaluation Modes
# ============================================================================


def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    n_groups: Optional[int] = None,
) -> EvalResult:
    """
    Evaluate using cross-validation on a single split.
    Useful for hyperparameter tuning on train or val sets.
    """
    groups = create_groups_for_split(len(y), n_groups=n_groups)
    
    print(f"\nCross-validation setup:")
    print(f"  Samples: {len(y)}")
    print(f"  Groups: {len(np.unique(groups))}")
    print(f"  Folds: {config.n_splits}")
    print(f"  Group distribution: min={np.bincount(groups).min()}, "
          f"max={np.bincount(groups).max()}, mean={np.bincount(groups).mean():.1f}")
    
    return evaluate_knn(X, y, groups, config)


def evaluate_train_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: PipelineConfig,
) -> Dict[str, Any]:
    """
    Train on full training set and evaluate on test set.
    This is the final evaluation mode for fault detection performance.
    """
    print(f"\nTrain-test evaluation:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    start_time = time.time()
    
    # Preprocessing
    print("  Preprocessing...")
    preprocessor = TimeSeriesPreprocessor(config)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    if config.verbose >= 1:
        print(f"  After preprocessing: train={X_train_proc.shape}, test={X_test_proc.shape}")
    
    # Compute distances
    print(f"  Computing {config.distance_metric} distances...")
    if config.distance_metric == "euclidean":
        D = pairwise_euclidean_matrix(X_test_proc, X_train_proc)
    elif config.distance_metric == "dtw":
        D = pairwise_dtw_matrix(
            X_test_proc,
            X_train_proc,
            window=config.dtw_window,
            n_jobs=config.n_jobs,
            batch_size=config.dtw_batch_size,
            verbose=config.verbose >= 2,
        )
    else:
        raise ValueError(f"Unknown metric: {config.distance_metric}")
    
    # KNN prediction
    print(f"  Running KNN (k={config.k})...")
    y_pred, y_proba = knn_vote_from_dist(
        D, y_train, k=config.k, weighted=config.distance_weighting
    )
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Custom classification report with meaningful labels
    target_names = ['normal', 'fault']
    report = classification_report(y_test, y_pred, target_names=target_names, 
                                   digits=3, zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    total_time = time.time() - start_time
    
    return {
        'y_true': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'report': report,
        'confusion_matrix': conf_mat,
        'time': total_time,
    }


# ============================================================================
# Experiment Runners
# ============================================================================


def run_hyperparameter_search(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> Tuple[Dict[int, EvalResult], int]:
    """
    Run hyperparameter search on validation set.
    
    Returns:
        Tuple of (results_dict, best_k)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Hyperparameter Search (k)")
    print("="*70)
    
    X_val, y_val = splits['val']
    groups = create_groups_for_split(len(y_val), n_groups=10)
    
    base_config = PipelineConfig(
        distance_metric='euclidean',
        standardize=True,
        use_pca=True,
        n_components=None,  # Auto-select
        pca_variance_threshold=0.95,
        n_splits=5,
        verbose=1,
    )
    
    k_values = [1, 3, 5, 7, 10, 15, 20, 25, 30]
    
    results = grid_search_k(X_val, y_val, groups, k_values, base_config)
    
    # Find best k
    best_k = max(results.keys(), key=lambda k: results[k].f1_weighted)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([
        {
            'k': k,
            'accuracy': res.accuracy,
            'f1_weighted': res.f1_weighted,
            'f1_macro': res.f1_macro,
            'time': res.timing['total'],
        }
        for k, res in results.items()
    ])
    
    results_df.to_csv(output_dir / 'hyperparam_search_k.csv', index=False)
    print(f"\nResults saved to {output_dir / 'hyperparam_search_k.csv'}")
    print(f"\nBest k: {best_k} (F1-weighted: {results[best_k].f1_weighted:.4f})")
    
    return results, best_k


def run_final_evaluation(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    best_k: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Final evaluation: train on train+val, test on test set.
    This evaluates the fault detection/diagnosis performance.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Final Test Set Evaluation (Fault Detection)")
    print("="*70)
    
    # Combine train and val
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    
    print(f"\nCombined training set: {X_train_full.shape}")
    print(f"  Normal samples: {np.sum(y_train_full == 0)}")
    print(f"  Fault samples: {np.sum(y_train_full == 1)}")
    print(f"Test set: {X_test.shape}")
    print(f"  Normal samples: {np.sum(y_test == 0)}")
    print(f"  Fault samples: {np.sum(y_test == 1)}")
    
    config = PipelineConfig(
        k=best_k,
        distance_metric='euclidean',
        standardize=True,
        use_pca=True,
        n_components=None,
        pca_variance_threshold=0.95,
        distance_weighting=True,
        verbose=1,
    )
    
    results = evaluate_train_test(
        X_train_full, y_train_full,
        X_test, y_test,
        config
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("FINAL TEST RESULTS - Fault Detection Performance")
    print(f"{'='*70}")
    print(f"Accuracy:      {results['accuracy']:.4f}")
    print(f"F1 (weighted): {results['f1_weighted']:.4f}")
    print(f"F1 (macro):    {results['f1_macro']:.4f}")
    print(f"Time:          {results['time']:.2f}s")
    print(f"\nClassification Report (0=normal, 1=fault):")
    print(results['report'])
    print(f"\nConfusion Matrix:")
    print("                Predicted")
    print("              Normal  Fault")
    print(f"Actual Normal  {results['confusion_matrix'][0,0]:5d}  {results['confusion_matrix'][0,1]:5d}")
    print(f"       Fault   {results['confusion_matrix'][1,0]:5d}  {results['confusion_matrix'][1,1]:5d}")
    
    # Calculate and display fault detection metrics
    tn, fp, fn, tp = results['confusion_matrix'].ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nFault Detection Metrics:")
    print(f"  Detection Rate (Recall):    {detection_rate:.4f}")
    print(f"  False Alarm Rate:           {false_alarm_rate:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'y_true': results['y_true'],
        'y_true_label': ['normal' if y == 0 else 'fault' for y in results['y_true']],
        'y_pred': results['y_pred'],
        'y_pred_label': ['normal' if y == 0 else 'fault' for y in results['y_pred']],
        'prob_normal': results['y_proba'][:, 0],
        'prob_fault': results['y_proba'][:, 1],
    })
    pred_df.to_csv(output_dir / f'test_predictions_k{best_k}.csv', index=False)
    
    # Save summary
    summary = {
        'dataset': 'CSTH (Continuous Stirred Tank Heater)',
        'task': 'Fault Detection and Diagnosis',
        'config': {
            'k': best_k,
            'distance_metric': 'euclidean',
            'use_pca': True,
            'pca_variance_threshold': 0.95,
        },
        'results': {
            'accuracy': float(results['accuracy']),
            'f1_weighted': float(results['f1_weighted']),
            'f1_macro': float(results['f1_macro']),
            'detection_rate': float(detection_rate),
            'false_alarm_rate': float(false_alarm_rate),
            'time': float(results['time']),
        },
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
        },
        'n_train': len(y_train_full),
        'n_test': len(results['y_true']),
    }
    
    with open(output_dir / f'test_results_k{best_k}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return results


def run_ablation_study(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Ablation study on preprocessing choices for fault detection.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Ablation Study - Preprocessing Impact on Fault Detection")
    print("="*70)
    
    X_val, y_val = splits['val']
    groups = create_groups_for_split(len(y_val), n_groups=10)
    
    configs = [
        ("Baseline (no preprocessing)", PipelineConfig(
            standardize=False, use_pca=False, k=10, n_splits=5, verbose=0
        )),
        ("Standardization only", PipelineConfig(
            standardize=True, use_pca=False, k=10, n_splits=5, verbose=0
        )),
        ("PCA only (95% var)", PipelineConfig(
            standardize=False, use_pca=True, n_components=None,
            pca_variance_threshold=0.95, k=10, n_splits=5, verbose=0
        )),
        ("PCA (50 components)", PipelineConfig(
            standardize=False, use_pca=True, n_components=50,
            k=10, n_splits=5, verbose=0
        )),
        ("Full pipeline (std+PCA)", PipelineConfig(
            standardize=True, use_pca=True, n_components=None,
            pca_variance_threshold=0.95, k=10, n_splits=5, verbose=0
        )),
    ]
    
    results = []
    
    for name, config in configs:
        print(f"\nTesting: {name}")
        result = evaluate_knn(X_val, y_val, groups, config)
        
        results.append({
            'configuration': name,
            'accuracy': result.accuracy,
            'f1_weighted': result.f1_weighted,
            'f1_macro': result.f1_macro,
            'time': result.timing['total'],
        })
        
        print(f"  Acc: {result.accuracy:.4f} | "
              f"F1w: {result.f1_weighted:.4f} | "
              f"Time: {result.timing['total']:.1f}s")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'ablation_study.csv', index=False)
    
    print(f"\n{'='*70}")
    print("Ablation Study Results:")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {output_dir / 'ablation_study.csv'}")
    
    return results_df


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main execution pipeline for CSTH fault detection experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run KNN pipeline on CSTH (Continuous Stirred Tank Heater) dataset for fault detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on validation set
  python src/run_csth.py --mode quick
  
  # Full hyperparameter search
  python src/run_csth.py --mode search
  
  # Final test evaluation with best k
  python src/run_csth.py --mode final --best-k 10
  
  # Run all experiments
  python src/run_csth.py --mode all

Dataset Info:
  CSTH = Continuous Stirred Tank Heater (process control simulation)
  Task: Fault detection and diagnosis
  Classes: 0=normal operation, 1=fault (instrumentation errors)
  Features: cold water flow, tank level, temperature
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/raw'),
        help='Directory containing train.pt, val.pt, test.pt'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results'),
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'search', 'final', 'ablation', 'all'],
        default='quick',
        help='Execution mode (default: quick)'
    )
    
    parser.add_argument(
        '--best-k',
        type=int,
        default=10,
        help='Best k value for final evaluation (use after search)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please ensure data/raw/ contains train.pt, val.pt, test.pt")
        sys.exit(1)
    
    # Load data
    try:
        splits = load_csth_dataset(args.data_dir)
    except Exception as e:
        print(f"\nError loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run experiments based on mode
    best_k = args.best_k
    
    try:
        if args.mode in ['quick', 'all']:
            print("\n" + "="*70)
            print("QUICK TEST: Validation Set with CV")
            print("="*70)
            X_val, y_val = splits['val']
            config = PipelineConfig(
                k=10,
                distance_metric='euclidean',
                standardize=True,
                use_pca=True,
                n_splits=5,
                verbose=1,
            )
            result = evaluate_with_cv(X_val, y_val, config, n_groups=10)
            print("\n" + str(result))
        
        if args.mode in ['search', 'all']:
            hp_results, best_k = run_hyperparameter_search(splits, args.output_dir)
        
        if args.mode in ['ablation', 'all']:
            run_ablation_study(splits, args.output_dir)
        
        if args.mode in ['final', 'all']:
            run_final_evaluation(splits, best_k, args.output_dir)
        
        print("\n" + "="*70)
        print("All experiments completed successfully!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
