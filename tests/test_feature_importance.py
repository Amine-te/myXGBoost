import numpy as np
import pytest
from myXGBoost import XGBRegressor, XGBClassifier

def test_feature_importance_regression():
    # Feature 0 is very informative, Feature 1 is noise
    X = np.random.rand(100, 2)
    y = X[:, 0] * 10 + np.random.normal(0, 0.1, 100)
    
    model = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    assert importances.shape == (2,)
    assert np.sum(importances) > 0
    assert np.isclose(np.sum(importances), 1.0)
    assert importances[0] > importances[1], "Feature 0 should be more important than Feature 1"

def test_feature_importance_classification():
    # Feature 0 is very informative, Feature 1 is noise
    X = np.random.rand(100, 2)
    # Class 1 if Feature 0 > 0.5
    y = (X[:, 0] > 0.5).astype(int)
    
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    assert importances.shape == (2,)
    assert np.sum(importances) > 0
    assert np.isclose(np.sum(importances), 1.0)
    assert importances[0] > importances[1], "Feature 0 should be more important than Feature 1"

def test_feature_importance_multiclass():
    # 3 classes, Feature 0 determines class
    X = np.random.rand(150, 2)
    y = np.zeros(150, dtype=int)
    y[X[:, 0] < 0.33] = 0
    y[(X[:, 0] >= 0.33) & (X[:, 0] < 0.66)] = 1
    y[X[:, 0] >= 0.66] = 2
    
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    importances = model.feature_importances_
    
    assert importances.shape == (2,)
    assert np.sum(importances) > 0
    assert np.isclose(np.sum(importances), 1.0)
    assert importances[0] > importances[1], "Feature 0 should be more important than Feature 1"
