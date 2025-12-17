"""Base metric interface."""

from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    All metrics should inherit from this class and implement
    the score method.
    """
    
    @abstractmethod
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate metric score.
        
        Parameters
        ----------
        y_true : ndarray
            True values.
        y_pred : ndarray
            Predicted values.
            
        Returns
        -------
        score : float
            Metric score.
        """
        pass
    
    @abstractmethod
    def is_higher_better(self) -> bool:
        """
        Whether higher metric values are better.
        
        Returns
        -------
        higher_better : bool
            True if higher is better, False if lower is better.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass
        
