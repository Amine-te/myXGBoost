"""Regression loss functions (squared error, etc.)."""

import numpy as np
from myXGBoost.loss.base import RegressionLoss


class MSELoss(RegressionLoss):
    """
    Mean Squared Error (MSE) loss for regression.
    
    Loss: L = (y_pred - y_true)^2 / 2
    Gradient: grad = y_pred - y_true
    Hessian: hess = 1
    
    Parameters
    ----------
    None
    
    Notes
    -----
    The MSE loss is the standard loss function for regression tasks.
    It provides constant hessian values, making it computationally efficient.
    """
    
    def loss(self, y_true, y_pred):
        """
        Compute MSE loss.
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values.
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
            
        Returns
        -------
        loss : float
            Average MSE loss.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean((y_pred - y_true) ** 2) / 2.0
    
    def grad(self, y_true, y_pred):
        """
        Compute first-order gradient.
        
        Gradient formula: grad = y_pred - y_true
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values.
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
            
        Returns
        -------
        grad : ndarray of shape (n_samples,)
            First-order gradients.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return y_pred - y_true
    
    def hess(self, y_true, y_pred):
        """
        Compute second-order gradient (hessian).
        
        Hessian formula: hess = 1 (constant)
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values (not used, but kept for interface consistency).
        y_pred : ndarray of shape (n_samples,)
            Predicted values (not used, but kept for interface consistency).
            
        Returns
        -------
        hess : ndarray of shape (n_samples,)
            Second-order gradients (all ones).
        """
        y_pred = np.asarray(y_pred)
        return np.ones_like(y_pred, dtype=np.float64)
    
    def grad_hess(self, y_true, y_pred):
        """
        Compute both gradient and hessian simultaneously.
        
        This is more efficient than calling grad() and hess() separately.
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values.
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
            
        Returns
        -------
        grad : ndarray of shape (n_samples,)
            First-order gradients.
        hess : ndarray of shape (n_samples,)
            Second-order gradients (all ones).
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        grad = y_pred - y_true
        hess = np.ones_like(y_pred, dtype=np.float64)
        return grad, hess


