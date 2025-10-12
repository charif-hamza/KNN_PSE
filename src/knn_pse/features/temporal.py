"""Temporal feature extraction for multivariate time-series."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float_]


@dataclass
class TemporalFeatureExtractor:
    """Derive temporal dynamics features following the sklearn API."""

    autocorr_lags: tuple[int, ...] = (1, 5, 10)

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> TemporalFeatureExtractor:
        self._validate_input(X)
        self.n_channels_ = X.shape[2]
        return self

    def transform(self, X: np.ndarray) -> FloatArray:
        self._validate_input(X)
        features = [
            self.extract_trend(X),
            self.extract_autocorrelation(X),
            self.extract_crosscorr(X),
            self.extract_stationarity(X),
        ]
        combined = np.concatenate(features, axis=1)
        return cast(FloatArray, combined.astype(float, copy=False))

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> FloatArray:
        return self.fit(X, y).transform(X)

    def extract_trend(self, X: np.ndarray) -> FloatArray:
        data = np.asarray(X)
        n_samples, n_timesteps, n_channels = data.shape
        slopes = np.zeros((n_samples, n_channels), dtype=float)
        time = np.arange(n_timesteps, dtype=float)
        for idx in range(n_samples):
            for ch in range(n_channels):
                series = data[idx, :, ch]
                if np.all(series == series[0]):
                    slopes[idx, ch] = 0.0
                    continue
                coeffs = np.polyfit(time, series, deg=1)
                slopes[idx, ch] = float(coeffs[0])
        slopes_reshaped = slopes.reshape(n_samples, -1)
        return cast(FloatArray, slopes_reshaped.astype(float, copy=False))

    def extract_autocorrelation(self, X: np.ndarray) -> FloatArray:
        data = np.asarray(X)
        n_samples, _, n_channels = data.shape
        n_features = n_channels * len(self.autocorr_lags)
        autocorr = np.zeros((n_samples, n_features), dtype=float)
        for sample_idx in range(n_samples):
            row_features: list[float] = []
            for ch in range(n_channels):
                series = data[sample_idx, :, ch]
                for lag in self.autocorr_lags:
                    row_features.append(self._lag_autocorr(series, lag))
            autocorr[sample_idx] = row_features
        return cast(FloatArray, autocorr.astype(float, copy=False))

    def extract_crosscorr(self, X: np.ndarray) -> FloatArray:
        data = np.asarray(X)
        n_samples, _, n_channels = data.shape
        if n_channels == 1:
            return cast(FloatArray, np.zeros((n_samples, 1), dtype=float))
        n_pairs = n_channels * (n_channels - 1) // 2
        cross = np.zeros((n_samples, n_pairs), dtype=float)
        for sample_idx in range(n_samples):
            series = data[sample_idx]
            row_features: list[float] = []
            for ch_a, ch_b in combinations(range(n_channels), 2):
                corr = np.corrcoef(series[:, ch_a], series[:, ch_b])[0, 1]
                if not np.isfinite(corr):
                    corr = 0.0
                row_features.append(float(corr))
            cross[sample_idx] = row_features
        return cast(FloatArray, cross.astype(float, copy=False))

    def extract_stationarity(self, X: np.ndarray) -> FloatArray:
        data = np.asarray(X)
        n_samples, _, n_channels = data.shape
        stats = np.zeros((n_samples, n_channels), dtype=float)
        for sample_idx in range(n_samples):
            for ch in range(n_channels):
                series = data[sample_idx, :, ch]
                stat = self._adf_statistic(series)
                stats[sample_idx, ch] = float(stat)
        stats_reshaped = stats.reshape(n_samples, -1)
        return cast(FloatArray, stats_reshaped.astype(float, copy=False))

    def _lag_autocorr(self, series: np.ndarray, lag: int) -> float:
        if lag <= 0 or lag >= len(series):
            return 0.0
        x = series[lag:]
        y = series[:-lag]
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(corr)

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

    def _adf_statistic(self, series: np.ndarray) -> float:
        data = np.asarray(series, dtype=float)
        if data.size < 3:
            return 0.0
        diff = np.diff(data)
        y_lag = data[:-1]
        X = np.column_stack([np.ones_like(y_lag), y_lag])
        try:
            coeffs, residuals, rank, _ = np.linalg.lstsq(X, diff, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        if rank < X.shape[1]:
            return 0.0
        fitted = X @ coeffs
        errors = diff - fitted
        dof = max(1, len(diff) - X.shape[1])
        sigma2 = float(np.dot(errors, errors) / dof)
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return 0.0
        denom = np.sqrt(max(cov[1, 1], 1e-12))
        return float(coeffs[1] / denom)
