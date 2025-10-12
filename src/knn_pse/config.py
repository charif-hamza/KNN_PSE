"""Configuration objects for the KNN CSTH pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DistanceMetric = Literal["euclidean", "dtw"]
DTWBackend = Literal["python", "numba", "fastdtw", "dtaidistance"]
EuclideanBackend = Literal["numpy", "torch"]
AnnBackend = Literal["annoy", "faiss"]
FeatureExtraction = Literal["none", "statistical", "temporal", "both"]
DimensionalityReduction = Literal["pca", "nmf", "ica", "isomap", "factor"]


@dataclass(slots=True)
class PipelineConfig:
    """Configuration options for the time-series KNN pipeline."""

    n_splits: int = 5
    standardize: bool = True
    use_pca: bool = True
    n_components: int | None = None
    pca_variance_threshold: float = 0.95
    dim_reduction_method: DimensionalityReduction = "pca"
    isomap_neighbors: int = 5
    k: int = 10
    distance_metric: DistanceMetric = "euclidean"
    distance_weighting: bool = True
    dtw_window: float | None = 0.1
    dtw_batch_size: int = 64
    dtw_backend: DTWBackend = "python"
    n_jobs: int = -1
    verbose: int = 1
    save_predictions: bool = False
    output_dir: Path | None = None
    euclidean_backend: EuclideanBackend = "numpy"
    use_gpu: bool = False
    precompute_train_distances: bool = False
    memmap_path: Path | None = None
    ann_backend: AnnBackend | None = None
    ann_n_trees: int = 50
    ann_search_k: int | None = None
    threshold_grid: Sequence[float] = field(
        default_factory=lambda: [i / 20 for i in range(1, 20)]
    )
    record_manifest: bool = True
    manifest_path: Path | None = None
    metadata: dict = field(default_factory=dict)
    feature_extraction: FeatureExtraction = "none"
