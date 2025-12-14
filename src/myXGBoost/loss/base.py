"""Base loss function interface."""

from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    """
    Abstract base class for loss functions.
    
    All loss functions must implement methods to compute:
    - Loss value
    - First-order gradient (grad)
    - Second-order gradient (hessian/hess)
    """
    
    @abstractmethod
    def loss(self, y_true, y_pred):
        """
        Compute the loss value.
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values.
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
            
        Returns
        -------
        loss : float
            Average loss value.
        """
        pass
    
    @abstractmethod
    def grad(self, y_true, y_pred):
        """
        Compute first-order gradient (gradient).
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values.
        y_pred : ndarray of shape (n_samples,)
            Predicted values (raw predictions, before transformation).
            
        Returns
        -------
        grad : ndarray of shape (n_samples,)
            First-order gradients.
        """
        pass
    
    @abstractmethod
    def hess(self, y_true, y_pred):
        """
        Compute second-order gradient (hessian).
        
        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True target values.
        y_pred : ndarray of shape (n_samples,)
            Predicted values (raw predictions, before transformation).
            
        Returns
        -------
        hess : ndarray of shape (n_samples,)
            Second-order gradients (hessians).
        """
        pass
    
    def grad_hess(self, y_true, y_pred):
        """
        Compute both gradient and hessian simultaneously.
        
        This method can be overridden for efficiency when both
        grad and hess can be computed together.
        
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
            Second-order gradients (hessians).
        """
        return self.grad(y_true, y_pred), self.hess(y_true, y_pred)


class RegressionLoss(LossFunction):
    """
    Base class for regression loss functions.
    
    Regression losses operate on continuous target values.
    """
    pass


class ClassificationLoss(LossFunction):
    """
    Base class for classification loss functions.
    
    Classification losses operate on discrete target values (class labels).
    """
    pass
