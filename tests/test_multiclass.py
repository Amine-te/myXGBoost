"""Tests for multiclass classification functionality."""

import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from myXGBoost import XGBClassifier


class TestMulticlassClassification:
    """Tests for multiclass classification."""
    
    def test_iris_3_class(self):
        """Test XGBClassifier on Iris dataset (3 classes)."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        clf = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
        clf.fit(X, y)
        
        # Check that it recognizes 3 classes
        assert clf.n_classes_ == 3
        assert len(clf.classes_) == 3
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
        
        # Check predictions
        y_pred = clf.predict(X)
        assert y_pred.shape == (150,)
        assert all(pred in [0, 1, 2] for pred in y_pred)
        
        # Check probabilities
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (150, 3)
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(y_proba.sum(axis=1), np.ones(150), decimal=6)
        
        # Probabilities should be in [0, 1]
        assert (y_proba >= 0).all()
        assert (y_proba <= 1).all()
        
        # Verify predicted class matches argmax of probabilities
        pred_from_proba = np.argmax(y_proba, axis=1)
        np.testing.assert_array_equal(pred_from_proba, y_pred)
        
        # Check accuracy is reasonable (should be high on training data)
        accuracy = np.mean(y_pred == y)
        assert accuracy > 0.8, f"Accuracy {accuracy} is too low"
    
    def test_5_class(self):
        """Test XGBClassifier with 5 classes."""
        X, y = make_classification(
            n_samples=300, n_features=10, n_classes=5,
            n_informative=5, n_redundant=2, random_state=42
        )
        
        clf = XGBClassifier(n_estimators=30, max_depth=4, random_state=42)
        clf.fit(X, y)
        
        # Check that it recognizes 5 classes
        assert clf.n_classes_ == 5
        assert len(clf.classes_) == 5
        
        # Check predictions
        y_pred = clf.predict(X)
        assert y_pred.shape == (300,)
        
        # Check probabilities
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (300, 5)
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(y_proba.sum(axis=1), np.ones(300), decimal=6)
    
    def test_10_class(self):
        """Test XGBClassifier with 10 classes."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_classes=10,
            n_informative=10, n_redundant=5, random_state=42
        )
        
        clf = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
        clf.fit(X, y)
        
        # Check that it recognizes 10 classes
        assert clf.n_classes_ == 10
        assert len(clf.classes_) == 10
        
        # Check predictions
        y_pred = clf.predict(X)
        assert y_pred.shape == (500,)
        
        # Check probabilities
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (500, 10)
        
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(y_proba.sum(axis=1), np.ones(500), decimal=6)
    
    def test_multiclass_predict_before_fit(self):
        """Test that predict before fit raises error for multiclass."""
        clf = XGBClassifier()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="not fitted"):
            clf.predict(X)
    
    def test_multiclass_predict_proba_before_fit(self):
        """Test that predict_proba before fit raises error for multiclass."""
        clf = XGBClassifier()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="not fitted"):
            clf.predict_proba(X)
    
    def test_multiclass_different_label_encoding(self):
        """Test multiclass with non-sequential class labels."""
        # Classes: 1, 3, 7 (non-sequential)
        X = np.random.randn(100, 5)
        y = np.random.choice([1, 3, 7], size=100)
        
        clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)
        
        # Check classes are correctly identified
        assert clf.n_classes_ == 3
        np.testing.assert_array_equal(clf.classes_, [1, 3, 7])
        
        # Check predictions are in the correct set
        y_pred = clf.predict(X)
        assert all(pred in [1, 3, 7] for pred in y_pred)
        
        # Check probabilities
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (100, 3)
        np.testing.assert_array_almost_equal(y_proba.sum(axis=1), np.ones(100), decimal=6)
    
    def test_multiclass_single_sample_prediction(self):
        """Test that single sample prediction works for multiclass."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        clf = XGBClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        # Predict single sample
        single_pred = clf.predict(X[[0]])
        assert single_pred.shape == (1,)
        
        single_proba = clf.predict_proba(X[[0]])
        assert single_proba.shape == (1, 3)
        np.testing.assert_almost_equal(single_proba.sum(), 1.0, decimal=6)
    
    def test_multiclass_consistency(self):
        """Test that multiclass gives consistent results with same random state."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        clf1 = XGBClassifier(n_estimators=20, random_state=42)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)
        
        clf2 = XGBClassifier(n_estimators=20, random_state=42)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)
        
        # Results should be identical with same random state
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_multiclass_probability_ordering(self):
        """Test that probabilities correspond to correct classes."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        clf = XGBClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)
        
        # For each sample, the predicted class should have the highest probability
        for i in range(len(X)):
            predicted_class = y_pred[i]
            class_idx = np.where(clf.classes_ == predicted_class)[0][0]
            assert class_idx == np.argmax(y_proba[i])
    
    def test_binary_vs_multiclass(self):
        """Test that binary and multiclass paths work correctly for 2 classes."""
        # Binary dataset
        X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
        
        clf = XGBClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        # Should detect as binary (2 classes)
        assert clf.n_classes_ == 2
        
        # Predictions should work
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)
        
        assert y_proba.shape == (100, 2)
        np.testing.assert_array_almost_equal(y_proba.sum(axis=1), np.ones(100), decimal=6)
