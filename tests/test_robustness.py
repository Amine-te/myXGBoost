"""Tests for model robustness and edge cases."""

import pytest
import numpy as np
from myXGBoost.estimators import XGBRegressor, XGBClassifier

def test_nan_handling():
    """Test that NaNs raise appropriate errors."""
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0])
    
    model = XGBClassifier()
    
    # Should raise ValueError due to NaNs
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit(X, y)

def test_infinity_handling():
    """Test that Infinity raises appropriate errors."""
    X = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0])
    
    model = XGBClassifier()
    
    with pytest.raises(ValueError, match="Input contains NaN, infinity"):
        model.fit(X, y)

def test_empty_input():
    """Test handling of empty inputs."""
    X = np.array([]).reshape(0, 5)
    y = np.array([])
    
    model = XGBRegressor()
    
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_single_sample():
    """Test training with a single sample."""
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([1.0])
    
    model = XGBRegressor(n_estimators=2)
    model.fit(X, y)
    
    pred = model.predict(X)
    assert len(pred) == 1
    assert isinstance(pred[0], float)

def test_feature_importances_unfitted():
    """Test accessing feature importances before fitting."""
    model = XGBRegressor()
    
    with pytest.raises(AttributeError, match="is not fitted yet"):
        _ = model.feature_importances_
