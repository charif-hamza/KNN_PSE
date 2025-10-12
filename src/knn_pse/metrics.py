"""Metrics utilities for the CSTH KNN pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


@dataclass(slots=True)
class ThresholdMetrics:
    """Metric snapshot at a specific decision threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict:
        """Return a serialisable representation."""

        return {
            "threshold": float(self.threshold),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "accuracy": float(self.accuracy),
        }


@dataclass(slots=True)
class ThresholdSweep:
    """A collection of threshold-dependent metrics."""

    metrics: list[ThresholdMetrics]

    @classmethod
    def from_probabilities(
        cls,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Iterable[float],
    ) -> ThresholdSweep:
        """Compute metrics across a grid of thresholds."""

        metrics: list[ThresholdMetrics] = []
        if y_proba.ndim == 2 and y_proba.shape[1] > 1:
            positive_scores = y_proba[:, 1]
        else:
            positive_scores = y_proba.ravel()

        for threshold in thresholds:
            y_pred = (positive_scores >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            accuracy = float(np.mean(y_pred == y_true))
            metrics.append(
                ThresholdMetrics(
                    threshold=threshold,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    accuracy=accuracy,
                )
            )

        return cls(metrics=metrics)

    def to_dict(self) -> list[dict]:
        """Serialise sweep to JSON-compatible list."""

        return [metric.to_dict() for metric in self.metrics]
