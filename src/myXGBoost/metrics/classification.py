"""Classification metrics (accuracy, log loss, AUC, etc.)."""

import numpy as np
from myXGBoost.metrics.base import Metric


def accuracy_score(y_true, y_pred, sample_weight=None, normalize=True):
    """
    Accuracy classification score.
    
    Higher is better.
    
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


def log_loss(y_true, y_pred, eps=1e-15, sample_weight=None):
    """
    Log loss (cross-entropy loss) for binary classification.
    
    Lower is better. Works with probability predictions.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities (must be in [0, 1]).
    eps : float, default=1e-15
        Small value to clip predictions to avoid log(0).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    Returns
    -------
    score : float
        Log loss score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Clip predictions to avoid log(0) and log(inf)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Binary cross-entropy: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        return np.average(loss, weights=sample_weight)
    else:
        return np.mean(loss)


def auc_score(y_true, y_pred, sample_weight=None):
    """
    Area Under the ROC Curve (AUC) for binary classification.
    
    Higher is better. Range: [0, 1], where 1 is perfect prediction.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Target scores (probability estimates, confidence values,
        or non-thresholded decision function values).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    Returns
    -------
    score : float
        AUC score.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    
    # Get unique threshold values (predictions)
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    if sample_weight is not None:
        weights_sorted = sample_weight[sorted_indices]
    else:
        weights_sorted = np.ones_like(y_true_sorted, dtype=np.float64)
    
    # Calculate true positives and false positives at each threshold
    n_pos = np.sum(y_true_sorted * weights_sorted)
    n_neg = np.sum((1 - y_true_sorted) * weights_sorted)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Return 0.5 for undefined cases
    
    tp = np.cumsum(y_true_sorted * weights_sorted)
    fp = np.cumsum((1 - y_true_sorted) * weights_sorted)
    
    # Normalize to get rates
    tpr = tp / n_pos
    fpr = fp / n_neg
    
    # Add origin and endpoint
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Calculate AUC using trapezoidal rule
    try:
        # Use trapezoid if available (numpy >= 1.23)
        auc = np.trapezoid(tpr, fpr)
    except AttributeError:
        # Fall back to trapz for older numpy
        auc = np.trapz(tpr, fpr)
    
    return float(auc)


# Metric Classes
class Accuracy(Metric):
    """Accuracy metric for classification."""
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return accuracy_score(y_true, y_pred)
    
    def is_higher_better(self) -> bool:
        """Higher is better for accuracy."""
        return True
    
    @property
    def name(self) -> str:
        """Metric name."""
        return "accuracy"


class LogLoss(Metric):
    """Log loss (cross-entropy) metric for classification."""
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate log loss."""
        return log_loss(y_true, y_pred)
    
    def is_higher_better(self) -> bool:
        """Lower is better for log loss."""
        return False
    
    @property
    def name(self) -> str:
        """Metric name."""
        return "logloss"


class AUC(Metric):
    """Area Under the ROC Curve metric for binary classification."""
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUC."""
        return auc_score(y_true, y_pred)
    
    def is_higher_better(self) -> bool:
        """Higher is better for AUC."""
        return True
    
    @property
    def name(self) -> str:
        """Metric name."""
        return "auc"

