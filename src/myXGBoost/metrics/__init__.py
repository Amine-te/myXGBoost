"""Evaluation metrics."""

from myXGBoost.metrics.regression import r2_score
from myXGBoost.metrics.classification import accuracy_score

__all__ = [
    "r2_score",
    "accuracy_score",
]

