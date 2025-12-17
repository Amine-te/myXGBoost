"""Regression metrics (RMSE, MAE, etc.)."""

import numpy as np
from myXGBoost.metrics.base import Metric


def rmse(y_true, y_pred, sample_weight=None):
    """
    Root Mean Squared Error (RMSE).
    
    Lower is better.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    Returns
    -------
    score : float
        RMSE score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        mse = np.average((y_true - y_pred) ** 2, weights=sample_weight)
    else:
        mse = np.mean((y_true - y_pred) ** 2)
    
    return np.sqrt(mse)


def mae(y_true, y_pred, sample_weight=None):
    """
    Mean Absolute Error (MAE).
    
    Lower is better.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    Returns
    -------
    score : float
        MAE score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        return np.average(np.abs(y_true - y_pred), weights=sample_weight)
    else:
        return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred, sample_weight=None):
    """
    R^2 (coefficient of determination) regression score function.
    
    Higher is better. Range: (-∞, 1], where 1 is perfect prediction.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    Returns
    -------
    score : float
        R^2 score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        numerator = np.sum(sample_weight * (y_true - y_pred) ** 2)
        denominator = np.sum(sample_weight * (y_true - np.average(y_true, weights=sample_weight)) ** 2)
    else:
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if denominator == 0:
        return 0.0 if numerator == 0 else -np.inf
    
    return 1 - (numerator / denominator)


# Metric Classes
class RMSE(Metric):
    """Root Mean Squared Error metric."""
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE."""
        return rmse(y_true, y_pred)
    
    def is_higher_better(self) -> bool:
        """Lower is better for RMSE."""
        return False
    
    @property
    def name(self) -> str:
        """Metric name."""
        return "rmse"


class MAE(Metric):
    """Mean Absolute Error metric."""
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAE."""
        return mae(y_true, y_pred)
    
    def is_higher_better(self) -> bool:
        """Lower is better for MAE."""
        return False
    
    @property
    def name(self) -> str:
        """Metric name."""
        return "mae"


class R2Score(Metric):
    """R^2 (Coefficient of Determination) metric."""
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R^2."""
        return r2_score(y_true, y_pred)
    
    def is_higher_better(self) -> bool:
        """Higher is better for R^2."""
        return True
    
    @property
    def name(self) -> str:
        """Metric name."""
        return "r2"

