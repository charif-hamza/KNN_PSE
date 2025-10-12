"""Tests for the statistical feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from knn_pse.features import StatisticalFeatureExtractor


def test_statistical_extractor_shape() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 10, 3))
    extractor = StatisticalFeatureExtractor()
    features = extractor.fit_transform(X)
    expected_features = 3 * 13
    assert features.shape == (4, expected_features)


def test_statistical_extractor_repeatability() -> None:
    X = np.arange(24, dtype=float).reshape(2, 4, 3)
    extractor = StatisticalFeatureExtractor(n_entropy_bins=5)
    fitted = extractor.fit_transform(X)
    transformed = extractor.transform(X)
    np.testing.assert_allclose(fitted, transformed)


def test_statistical_extractor_known_values() -> None:
    X = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ]
    )
    extractor = StatisticalFeatureExtractor(n_entropy_bins=4)
    features = extractor.fit_transform(X)
    means = features[0, :2]
    assert np.allclose(means, [4.0, 5.0])
    mins = features[0, 4:6]
    assert np.allclose(mins, [1.0, 2.0])
    maxs = features[0, 6:8]
    assert np.allclose(maxs, [7.0, 8.0])


def test_statistical_extractor_invalid_input() -> None:
    extractor = StatisticalFeatureExtractor()
    with pytest.raises(ValueError):
        extractor.fit(np.ones((3, 4)))
