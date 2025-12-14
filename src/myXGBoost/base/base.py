"""Base estimator classes following sklearn conventions."""

from abc import ABC, abstractmethod
import numpy as np


class BaseEstimator(ABC):
    """
    Base class for all estimators in myXGBoost.
    
    This class provides basic functionality for sklearn-compatible estimators
    including get_params, set_params, and basic validation.
    """
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                value = getattr(self, key)
                if deep and hasattr(value, 'get_params'):
                    deep_items = value.get_params().items()
                    out.update((key + '__' + k, val) for k, val in deep_items)
                else:
                    out[key] = value
        return out
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            return self
        
        valid_params = self.get_params(deep=False)
        
        nested_params = {}
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params)!r}."
                )
            
            if delim:
                nested_params.setdefault(key, {})[sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        
        return self
    
    def __repr__(self):
        """String representation of the estimator."""
        class_name = self.__class__.__name__
        params = self.get_params(deep=False)
        params_str = ', '.join(f'{k}={v!r}' for k, v in sorted(params.items()))
        return f"{class_name}({params_str})"


class RegressorMixin:
    """
    Mixin class for all regression estimators.
    
    Provides score method for regression.
    """
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from myXGBoost.metrics.regression import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class ClassifierMixin:
    """
    Mixin class for all classification estimators.
    
    Provides score method for classification.
    """
    
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from myXGBoost.metrics.classification import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
