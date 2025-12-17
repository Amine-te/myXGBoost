"""Classification loss functions (logistic, softmax, etc.)."""

import numpy as np
from myXGBoost.loss.base import ClassificationLoss


def sigmoid(x):
    """
    Sigmoid function: 1 / (1 + exp(-x)).
    
    Parameters
    ----------
    x : array-like
        Input values.
        
    Returns
    -------
    sigmoid : ndarray
        Sigmoid of input values.
    """
    # Clip x to avoid overflow
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class LogisticLoss(ClassificationLoss):
    """
    Logistic loss (binary cross-entropy) for binary classification.
    
    Loss: L = -[y * log(p) + (1-y) * log(1-p)]
    where p = sigmoid(y_pred)
    
    Gradient: grad = sigmoid(y_pred) - y = p - y
    Hessian: hess = p * (1 - p) = sigmoid(y_pred) * (1 - sigmoid(y_pred))
    
    Parameters
    ----------
    None
    
    Notes
    -----
    This loss function is used for binary classification tasks.
    The target values y_true should be in {0, 1}.
    
    The gradient and hessian are computed from the raw predictions
    (before sigmoid transformation), which is the standard approach
    in gradient boosting frameworks like XGBoost.
    """
    
    def loss(self, y_true, y_pred):
        """
        Compute logistic loss.
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values (should be in {0, 1}).
        y_pred : ndarray of shape (n_samples,)
            Raw predicted values (before sigmoid).
            
        Returns
        -------
        loss : float
            Average logistic loss.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Compute probabilities
        p = sigmoid(y_pred)
        
        # Avoid log(0) by clipping probabilities
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        loss = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        return np.mean(loss)
    
    def grad(self, y_true, y_pred):
        """
        Compute first-order gradient.
        
        Gradient formula: grad = sigmoid(y_pred) - y_true
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values (should be in {0, 1}).
        y_pred : ndarray of shape (n_samples,)
            Raw predicted values (before sigmoid).
            
        Returns
        -------
        grad : ndarray of shape (n_samples,)
            First-order gradients.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Gradient: sigmoid(pred) - y_true
        p = sigmoid(y_pred)
        return p - y_true
    
    def hess(self, y_true, y_pred):
        """
        Compute second-order gradient (hessian).
        
        Hessian formula: hess = sigmoid(y_pred) * (1 - sigmoid(y_pred))
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values (not used, but kept for interface consistency).
        y_pred : ndarray of shape (n_samples,)
            Raw predicted values (before sigmoid).
            
        Returns
        -------
        hess : ndarray of shape (n_samples,)
            Second-order gradients (hessians).
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Hessian: p * (1 - p) where p = sigmoid(y_pred)
        p = sigmoid(y_pred)
        return p * (1 - p)
    
    def grad_hess(self, y_true, y_pred):
        """
        Compute both gradient and hessian simultaneously.
        
        This is more efficient than calling grad() and hess() separately
        since both require computing sigmoid(y_pred).
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values (should be in {0, 1}).
        y_pred : ndarray of shape (n_samples,)
            Raw predicted values (before sigmoid).
            
        Returns
        -------
        grad : ndarray of shape (n_samples,)
            First-order gradients.
        hess : ndarray of shape (n_samples,)
            Second-order gradients (hessians).
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Compute sigmoid once and reuse
        p = sigmoid(y_pred)
        grad = p - y_true
        hess = p * (1 - p)
        
        return grad, hess

