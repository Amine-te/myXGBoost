"""Tests for model serialization (pickle/joblib)."""

import pytest
import numpy as np
import pickle
import tempfile
import os
from myXGBoost.estimators import XGBRegressor, XGBClassifier

def test_pickle_regressor():
    """Test pickling of XGBRegressor."""
    # Create random data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    
    # Train model
    model = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Get predictions
    pred_original = model.predict(X)
    
    # Pickle and unpickle
    serialized = pickle.dumps(model)
    model_loaded = pickle.loads(serialized)
    
    # Check predictions match
    pred_loaded = model_loaded.predict(X)
    np.testing.assert_allclose(pred_original, pred_loaded)
    
    # Check attributes preserved
    assert model.n_estimators == model_loaded.n_estimators
    assert model.max_depth == model_loaded.max_depth

def test_pickle_classifier():
    """Test pickling of XGBClassifier."""
    # Create random data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Train model
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Get predictions
    pred_original = model.predict(X)
    proba_original = model.predict_proba(X)
    
    # Pickle and unpickle
    serialized = pickle.dumps(model)
    model_loaded = pickle.loads(serialized)
    
    # Check predictions match
    pred_loaded = model_loaded.predict(X)
    proba_loaded = model_loaded.predict_proba(X)
    
    np.testing.assert_array_equal(pred_original, pred_loaded)
    np.testing.assert_allclose(proba_original, proba_loaded)

def test_pickle_file_io():
    """Test saving and loading from file."""
    # Create random data
    X = np.random.rand(50, 3)
    y = np.random.rand(50)
    
    model = XGBRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)
    pred_orig = model.predict(X)
    
    # Use temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name
        
    try:
        with open(tmp_path, 'rb') as f:
            model_loaded = pickle.load(f)
            
        pred_loaded = model_loaded.predict(X)
        np.testing.assert_allclose(pred_orig, pred_loaded)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
