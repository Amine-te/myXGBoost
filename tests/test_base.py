"""Tests for base classes."""

import pytest
import numpy as np
from myXGBoost.base.base import BaseEstimator, RegressorMixin, ClassifierMixin


class DummyEstimator(BaseEstimator):
    """Dummy estimator for testing BaseEstimator."""
    
    def __init__(self, param1=1, param2=2.0):
        self.param1 = param1
        self.param2 = param2


class DummyRegressor(BaseEstimator, RegressorMixin):
    """Dummy regressor for testing RegressorMixin."""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, X, y):
        self.fitted = True
        return self
    
    def predict(self, X):
        return np.zeros(X.shape[0])


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier for testing ClassifierMixin."""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, X, y):
        self.fitted = True
        return self
    
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class TestBaseEstimator:
    """Tests for BaseEstimator."""
    
    def test_get_params(self):
        """Test get_params method."""
        est = DummyEstimator(param1=10, param2=20.0)
        params = est.get_params()
        
        assert 'param1' in params
        assert 'param2' in params
        assert params['param1'] == 10
        assert params['param2'] == 20.0
    
    def test_set_params(self):
        """Test set_params method."""
        est = DummyEstimator()
        est.set_params(param1=5, param2=10.0)
        
        assert est.param1 == 5
        assert est.param2 == 10.0
    
    def test_set_params_invalid(self):
        """Test set_params with invalid parameter."""
        est = DummyEstimator()
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(invalid_param=10)
    
    def test_repr(self):
        """Test string representation."""
        est = DummyEstimator(param1=1, param2=2.0)
        repr_str = repr(est)
        
        assert 'DummyEstimator' in repr_str
        assert 'param1' in repr_str
        assert 'param2' in repr_str


class TestRegressorMixin:
    """Tests for RegressorMixin."""
    
    def test_score(self):
        """Test score method for regression."""
        reg = DummyRegressor()
        reg.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))
        
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        
        # Since predict returns zeros, score should be low
        score = reg.score(X, y)
        assert isinstance(score, float)


class TestClassifierMixin:
    """Tests for ClassifierMixin."""
    
    def test_score(self):
        """Test score method for classification."""
        clf = DummyClassifier()
        clf.fit(np.array([[1], [2], [3]]), np.array([0, 1, 0]))
        
        X = np.array([[1], [2], [3]])
        y = np.array([0, 1, 0])
        
        # Since predict returns zeros, score should be low
        score = clf.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
