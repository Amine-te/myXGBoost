"""Classification metrics (accuracy, log loss, etc.)."""

import numpy as np


def accuracy_score(y_true, y_pred, sample_weight=None, normalize=True):
    """
    Accuracy classification score.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : bool, default=True
        If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
        
    Returns
    -------
    score : float
        If normalize == True, return the fraction of correctly
        classified samples (float), else returns the number of correctly
        classified samples (int).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        correct = np.sum(sample_weight * (y_true == y_pred))
        total = np.sum(sample_weight)
    else:
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
    
    if normalize:
        return correct / total if total > 0 else 0.0
    else:
        return correct
