"""KNN Pipeline for CSTH fault detection."""

from .cli import main
from .config import PipelineConfig
from .metrics import ThresholdSweep
from .pipeline import (
    EvalResult,
    PCATimeCompressor,
    TimeSeriesPreprocessor,
    batched_lower_triangular_euclidean,
    compute_ann_neighbors,
    compute_run_manifest,
    demo_synthetic,
    evaluate_knn,
    grid_search_k,
    knn_vote_from_dist,
    make_windows,
    pairwise_distance_matrix,
)

__all__ = [
    "EvalResult",
    "PCATimeCompressor",
    "PipelineConfig",
    "TimeSeriesPreprocessor",
    "batched_lower_triangular_euclidean",
    "compute_ann_neighbors",
    "compute_run_manifest",
    "demo_synthetic",
    "evaluate_knn",
    "grid_search_k",
    "knn_vote_from_dist",
    "make_windows",
    "main",
    "pairwise_distance_matrix",
    "ThresholdSweep",
]
