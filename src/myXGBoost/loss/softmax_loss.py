"""Softmax loss function for multiclass classification."""

import numpy as np
from myXGBoost.loss.base import ClassificationLoss


def softmax(x):
    """
    Compute softmax values for each row of x.
    
    Uses the log-sum-exp trick for numerical stability:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_classes)
        Input logits (raw predictions).
        
    Returns
    -------
    probs : ndarray of shape (n_samples, n_classes)
        Softmax probabilities.
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -500, 500))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class SoftmaxLoss(ClassificationLoss):
    """
    Softmax loss (categorical cross-entropy) for multiclass classification.
    
    Loss: L = -sum(y_true * log(softmax(y_pred)))
    
    For multiclass classification with K classes, we compute:
    - Gradient: grad_k = p_k - y_k where p = softmax(y_pred)
    - Hessian: hess_k = p_k * (1 - p_k)
    
    Parameters
    ----------
    n_classes : int
        Number of classes for classification.
        
    Notes
    -----
    This loss function is used for multiclass classification tasks.
    The target values y_true should be class indices in {0, 1, ..., n_classes-1}.
    
    The predictions y_pred should be a 2D array of shape (n_samples, n_classes)
    containing the raw logits for each class.
    """
    
    def __init__(self, n_classes):
        """
        Initialize SoftmaxLoss.
        
        Parameters
        ----------
        n_classes : int
            Number of classes.
        """
        self.n_classes = n_classes
    
    def loss(self, y_true, y_pred):
        """
        Compute categorical cross-entropy loss.
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels (integers from 0 to n_classes-1).
        y_pred : ndarray of shape (n_samples, n_classes)
            Raw predicted logits for each class.
            
        Returns
        -------
        loss : float
            Average categorical cross-entropy loss.
        """
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        n_samples = y_pred.shape[0]
        
        # Compute softmax probabilities
        probs = softmax(y_pred)
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        
        # Compute cross-entropy: -log(p[true_class])
        # Use advanced indexing to get probability of true class
        true_class_probs = probs[np.arange(n_samples), y_true]
        loss = -np.log(true_class_probs)
        
        return np.mean(loss)
    
    def grad(self, y_true, y_pred):
        """
        Compute first-order gradients.
        
        Gradient formula: grad = softmax(y_pred) - one_hot(y_true)
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels (integers from 0 to n_classes-1).
        y_pred : ndarray of shape (n_samples, n_classes)
            Raw predicted logits for each class.
            
        Returns
        -------
        grad : ndarray of shape (n_samples, n_classes)
            First-order gradients for each class.
        """
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        n_samples = y_pred.shape[0]
        
        # Compute softmax probabilities
        probs = softmax(y_pred)
        
        # Create one-hot encoding of y_true
        y_one_hot = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        y_one_hot[np.arange(n_samples), y_true] = 1.0
        
        # Gradient: p - y_true (one-hot)
        grad = probs - y_one_hot
        
        return grad
    
    def hess(self, y_true, y_pred):
        """
        Compute second-order gradients (hessians).
        
        Hessian formula: hess = p * (1 - p) where p = softmax(y_pred)
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels (not used, kept for interface consistency).
        y_pred : ndarray of shape (n_samples, n_classes)
            Raw predicted logits for each class.
            
        Returns
        -------
        hess : ndarray of shape (n_samples, n_classes)
            Second-order gradients (hessians) for each class.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Compute softmax probabilities
        probs = softmax(y_pred)
        
        # Hessian: p * (1 - p) for each class
        hess = probs * (1.0 - probs)
        
        # Add small constant for numerical stability
        hess = np.maximum(hess, 1e-16)
        
        return hess
    
    def grad_hess(self, y_true, y_pred):
        """
        Compute both gradient and hessian simultaneously.
        
        This is more efficient than calling grad() and hess() separately
        since both require computing softmax(y_pred).
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels (integers from 0 to n_classes-1).
        y_pred : ndarray of shape (n_samples, n_classes)
            Raw predicted logits for each class.
            
        Returns
        -------
        grad : ndarray of shape (n_samples, n_classes)
            First-order gradients.
        hess : ndarray of shape (n_samples, n_classes)
            Second-order gradients (hessians).
        """
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        n_samples = y_pred.shape[0]
        
        # Compute softmax once and reuse
        probs = softmax(y_pred)
        
        # Gradient computation
        y_one_hot = np.zeros((n_samples, self.n_classes), dtype=np.float64)
        y_one_hot[np.arange(n_samples), y_true] = 1.0
        grad = probs - y_one_hot
        
        # Hessian computation
        hess = probs * (1.0 - probs)
        hess = np.maximum(hess, 1e-16)
        
        return grad, hess

