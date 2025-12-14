"""Gradient boosting core implementation."""

from abc import ABC, abstractmethod
import numpy as np


class BoosterBase(ABC):
    """
    Abstract base class for gradient boosting models.
    
    This class defines the interface that all boosting implementations
    must follow, including fit, predict, and evaluation methods.
    """
    
    @abstractmethod
    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None, verbose=False):
        """
        Fit the gradient boosting model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        eval_set : list of tuples (X, y), default=None
            List of (X, y) tuple pairs to use as validation sets for
            early stopping and evaluation.
        eval_metric : str or callable, default=None
            Metric to use for evaluation. If None, uses default metric
            for the task (regression/classification).
        verbose : bool, default=False
            If True, prints progress information during training.
            
        Returns
        -------
        self : object
            Returns self.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict target values for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Note: This method is only applicable for classification tasks.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities of the input samples.
        """
        pass
    
    def eval_metrics(self, X, y, metrics=None):
        """
        Evaluate the model on given data using specified metrics.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,)
            True target values.
        metrics : list of str or callable, default=None
            List of metric names or callable functions to evaluate.
            If None, uses default metrics for the task.
            
        Returns
        -------
        results : dict
            Dictionary mapping metric names to their values.
        """
        y_pred = self.predict(X)
        
        if metrics is None:
            # Default metrics will be set by subclasses
            return {}
        
        results = {}
        for metric in metrics:
            if isinstance(metric, str):
                # Metric name - will be implemented in metrics module
                results[metric] = self._compute_metric(metric, y, y_pred)
            elif callable(metric):
                # Custom metric function
                results[metric.__name__] = metric(y, y_pred)
        
        return results
    
    def _compute_metric(self, metric_name, y_true, y_pred):
        """
        Compute a metric by name.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to compute.
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.
            
        Returns
        -------
        score : float
            Metric value.
        """
        # This will be implemented to dispatch to appropriate metric module
        raise NotImplementedError(
            f"Metric computation for '{metric_name}' not yet implemented"
        )
