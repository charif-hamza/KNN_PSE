"""Statistical feature extraction for multivariate time-series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks  # type: ignore[import-untyped]
from scipy.stats import entropy, kurtosis, skew  # type: ignore[import-untyped]

FloatArray = NDArray[np.float_]


@dataclass
class StatisticalFeatureExtractor:
    """Compute descriptive statistics for each channel in a sequence.

    The extractor follows the scikit-learn estimator contract via
    :meth:`fit`, :meth:`transform`, and :meth:`fit_transform`.
    """

    n_entropy_bins: int = 10

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> StatisticalFeatureExtractor:
        self._validate_input(X)
        self.n_channels_ = X.shape[2]
        return self

    def transform(self, X: np.ndarray) -> FloatArray:
        self._validate_input(X)
        features = [
            self.extract_per_channel(X),
            self.extract_distribution(X),
            self.extract_peaks(X),
        ]
        combined = np.concatenate(features, axis=1)
        return cast(FloatArray, combined.astype(float, copy=False))

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> FloatArray:
        return self.fit(X, y).transform(X)

    def extract_per_channel(self, X: np.ndarray) -> FloatArray:
        data = np.asarray(X)
        means = np.mean(data, axis=1)
        stds = np.std(data, axis=1, ddof=0)
        minimums = np.min(data, axis=1)
        maximums = np.max(data, axis=1)
        ranges = maximums - minimums
        medians = np.median(data, axis=1)
        q75 = np.percentile(data, 75, axis=1)
        q25 = np.percentile(data, 25, axis=1)
        iqr_values = q75 - q25
        stacked = np.stack(
            [
                means,
                stds,
                minimums,
                maximums,
                ranges,
                medians,
                iqr_values,
            ],
            axis=1,
        )
        reshaped = stacked.reshape(data.shape[0], -1)
        return cast(FloatArray, reshaped.astype(float, copy=False))

    def extract_distribution(self, X: np.ndarray) -> FloatArray:
        data = np.asarray(X)
        skewness = skew(data, axis=1, bias=False, nan_policy="omit")
        kurt = kurtosis(data, axis=1, fisher=True, bias=False, nan_policy="omit")
        entropies = []
        for sample in data:
            sample_entropy = []
            for channel in sample.T:
                hist, _ = np.histogram(channel, bins=self.n_entropy_bins, density=True)
                hist = np.clip(hist, 1e-12, None)
                sample_entropy.append(float(entropy(hist)))
            entropies.append(sample_entropy)
        entropies_arr = np.asarray(entropies)
        stacked = np.stack([skewness, kurt, entropies_arr], axis=1)
        flattened = stacked.reshape(data.shape[0], -1)
        cleaned = np.nan_to_num(flattened, copy=False)
        return cast(FloatArray, cleaned.astype(float, copy=False))

    def extract_peaks(self, X: np.ndarray) -> FloatArray:
        data = np.asarray(X)
        n_samples, n_timesteps, n_channels = data.shape
        counts = np.zeros((n_samples, n_channels), dtype=float)
        amplitudes = np.zeros((n_samples, n_channels), dtype=float)
        locations = np.zeros((n_samples, n_channels), dtype=float)

        denom = max(n_timesteps - 1, 1)
        for sample_idx in range(n_samples):
            for channel_idx in range(n_channels):
                series = data[sample_idx, :, channel_idx]
                peaks, _ = find_peaks(series)
                counts[sample_idx, channel_idx] = float(len(peaks))
                if peaks.size > 0:
                    amplitudes[sample_idx, channel_idx] = float(np.max(series[peaks]))
                    norm_positions = peaks / denom
                    locations[sample_idx, channel_idx] = float(np.mean(norm_positions))
        stacked = np.stack([counts, amplitudes, locations], axis=1)
        reshaped = stacked.reshape(n_samples, -1)
        return cast(FloatArray, reshaped.astype(float, copy=False))

    def _validate_input(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if X.ndim != 3:
            msg = (
                "Expected a 3D array of shape (n_samples, n_timesteps, n_channels); "
                f"got {X.shape}"
            )
            raise ValueError(msg)
        fitted_channels = getattr(self, "n_channels_", None)
        if fitted_channels is not None and X.shape[2] != fitted_channels:
            msg = (
                "Number of channels differs from fitted data: "
                f"expected {fitted_channels}, got {X.shape[2]}"
            )
            raise ValueError(msg)
