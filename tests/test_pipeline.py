"""Unit tests for the KNN CSTH pipeline."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from knn_pse.config import PipelineConfig
from knn_pse.pipeline import (
    TimeSeriesPreprocessor,
    batched_lower_triangular_euclidean,
    demo_synthetic,
    evaluate_knn,
    make_windows,
    pairwise_distance_matrix,
)


@pytest.fixture()
def toy_dataframe() -> pd.DataFrame:
    data = {
        "run_id": np.repeat([0, 1], 10),
        "time": list(range(10)) * 2,
        "x0": np.arange(20, dtype=float),
        "x1": np.linspace(0.0, 1.0, 20),
        "label": [0] * 10 + [1] * 10,
    }
    return pd.DataFrame(data)


def test_make_windows_shapes(toy_dataframe: pd.DataFrame) -> None:
    X, y, groups, features = make_windows(
        toy_dataframe,
        feature_cols=["x0", "x1"],
        label_col="label",
        window=4,
        stride=2,
    )
    assert X.shape == (7, 4, 2)
    assert y.tolist()[:3] == [0, 0, 0]
    assert list(groups[:3]) == [0, 0, 0]
    assert features == ["x0", "x1"]


def test_preprocessor_shape_and_pca() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5, 3))
    config = PipelineConfig(use_pca=True, n_components=2)
    preprocessor = TimeSeriesPreprocessor(config)
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape == (12, 5, 2)
    X_again = preprocessor.transform(X)
    np.testing.assert_allclose(X_transformed, X_again, atol=1e-6)


def test_distance_matrix_symmetry() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3, 4))
    matrix = batched_lower_triangular_euclidean(X)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-6)


def test_pairwise_distance_matches_direct() -> None:
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(5, 3, 2))
    X_test = rng.normal(size=(4, 3, 2))
    config = PipelineConfig(distance_metric="euclidean")
    distances, _ = pairwise_distance_matrix(X_test, X_train, config)
    manual = []
    for sample in X_test:
        dists = [np.linalg.norm(sample - ref) for ref in X_train]
        manual.append(dists)
    np.testing.assert_allclose(distances, np.asarray(manual))


def test_synthetic_end_to_end() -> None:
    X, y, groups, _ = demo_synthetic(
        n_runs=4, length=120, n_features=3, window=20, stride=10
    )
    config = PipelineConfig(k=3, n_splits=3, threshold_grid=[0.3, 0.5, 0.7])
    result = evaluate_knn(X, y, groups, config)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.probabilities is not None
    assert result.threshold_sweep.metrics
    assert result.confusion_mat.shape[0] == result.confusion_mat.shape[1]
