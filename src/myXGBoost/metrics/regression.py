"""Regression metrics (RMSE, MAE, etc.)."""

import numpy as np


def r2_score(y_true, y_pred, sample_weight=None):
    """
    R^2 (coefficient of determination) regression score function.
    
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        numerator = np.sum(sample_weight * (y_true - y_pred) ** 2)
        denominator = np.sum(sample_weight * (y_true - np.average(y_true, weights=sample_weight)) ** 2)
    else:
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if denominator == 0:
        return 0.0 if numerator == 0 else -np.inf
    
    return 1 - (numerator / denominator)
