"""Tests for XGBRegressor and XGBClassifier."""

import pytest
import numpy as np
from myXGBoost import XGBRegressor, XGBClassifier


class TestXGBRegressor:
    """Tests for XGBRegressor."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        reg = XGBRegressor()
        
        assert reg.learning_rate == 0.1
        assert reg.n_estimators == 100
        assert reg.max_depth == 6
        assert reg.min_child_weight == 1.0
        assert reg.gamma == 0.0
        assert reg.subsample == 1.0
        assert reg.colsample_bytree == 1.0
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        reg = XGBRegressor(
            learning_rate=0.05,
            n_estimators=50,
            max_depth=3,
            min_child_weight=2.0,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            verbose=True,
        )
        
        assert reg.learning_rate == 0.05
        assert reg.n_estimators == 50
        assert reg.max_depth == 3
        assert reg.min_child_weight == 2.0
        assert reg.gamma == 0.1
        assert reg.subsample == 0.8
        assert reg.colsample_bytree == 0.9
        assert reg.random_state == 42
        assert reg.verbose is True
    
    def test_fit(self):
        """Test fit method."""
        reg = XGBRegressor(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        # Should not raise an error
        reg.fit(X, y)
        
        assert hasattr(reg, 'n_features_in_')
        assert reg.n_features_in_ == 2
    
    def test_fit_invalid_params(self):
        """Test fit with invalid hyperparameters."""
        reg = XGBRegressor(learning_rate=-0.1)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        
        with pytest.raises(ValueError, match="learning_rate"):
            reg.fit(X, y)
    
    def test_predict_before_fit(self):
        """Test predict before fit raises error."""
        reg = XGBRegressor()
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="not fitted"):
            reg.predict(X)
    
    def test_predict_after_fit(self):
        """Test predict after fit."""
        reg = XGBRegressor(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        
        reg.fit(X, y)
        predictions = reg.predict(X)
        
        assert predictions.shape == (3,)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_wrong_features(self):
        """Test predict with wrong number of features."""
        reg = XGBRegressor(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        
        reg.fit(X, y)
        
        X_wrong = np.array([[1, 2, 3]])  # Wrong number of features
        
        with pytest.raises(ValueError, match="features"):
            reg.predict(X_wrong)
    
    def test_predict_proba_raises_error(self):
        """Test that predict_proba raises NotImplementedError for regression."""
        reg = XGBRegressor(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        
        reg.fit(X, y)
        
        with pytest.raises(NotImplementedError):
            reg.predict_proba(X)
    
    def test_get_params(self):
        """Test get_params method."""
        reg = XGBRegressor(learning_rate=0.05, n_estimators=50)
        params = reg.get_params()
        
        assert 'learning_rate' in params
        assert 'n_estimators' in params
        assert params['learning_rate'] == 0.05
        assert params['n_estimators'] == 50
    
    def test_set_params(self):
        """Test set_params method."""
        reg = XGBRegressor()
        reg.set_params(learning_rate=0.05, n_estimators=50)
        
        assert reg.learning_rate == 0.05
        assert reg.n_estimators == 50


class TestXGBClassifier:
    """Tests for XGBClassifier."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        clf = XGBClassifier()
        
        assert clf.learning_rate == 0.1
        assert clf.n_estimators == 100
        assert clf.max_depth == 6
        assert clf.min_child_weight == 1.0
        assert clf.gamma == 0.0
        assert clf.subsample == 1.0
        assert clf.colsample_bytree == 1.0
    
    def test_fit(self):
        """Test fit method."""
        clf = XGBClassifier(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        # Should not raise an error
        clf.fit(X, y)
        
        assert hasattr(clf, 'n_features_in_')
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'n_classes_')
        assert clf.n_features_in_ == 2
        assert clf.n_classes_ == 2
        assert len(clf.classes_) == 2
    
    def test_predict_before_fit(self):
        """Test predict before fit raises error."""
        clf = XGBClassifier()
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="not fitted"):
            clf.predict(X)
    
    def test_predict_after_fit(self):
        """Test predict after fit."""
        clf = XGBClassifier(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        clf.fit(X, y)
        predictions = clf.predict(X)
        
        assert predictions.shape == (3,)
        assert isinstance(predictions, np.ndarray)
        # Predictions should be in the classes
        assert all(pred in clf.classes_ for pred in predictions)
    
    def test_predict_proba(self):
        """Test predict_proba method."""
        clf = XGBClassifier(n_estimators=10, random_state=42)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        
        assert proba.shape == (3, 2)  # 3 samples, 2 classes
        assert isinstance(proba, np.ndarray)
        # Probabilities should sum to 1 for each sample
        np.testing.assert_almost_equal(proba.sum(axis=1), np.ones(3))
        # Probabilities should be between 0 and 1
        assert (proba >= 0).all()
        assert (proba <= 1).all()
    
    def test_fit_invalid_params(self):
        """Test fit with invalid hyperparameters."""
        clf = XGBClassifier(subsample=1.5)  # Invalid: > 1
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        with pytest.raises(ValueError, match="subsample"):
            clf.fit(X, y)


class TestBoosterBase:
    """Tests for BoosterBase interface."""
    
    def test_booster_base_is_abstract(self):
        """Test that BoosterBase cannot be instantiated."""
        from myXGBoost.booster.gradient_booster import BoosterBase
        
        with pytest.raises(TypeError):
            BoosterBase()
    
    def test_xgb_regressor_inherits_booster_base(self):
        """Test that XGBRegressor inherits from BoosterBase."""
        reg = XGBRegressor()
        from myXGBoost.booster.gradient_booster import BoosterBase
        
        assert isinstance(reg, BoosterBase)
    
    def test_xgb_classifier_inherits_booster_base(self):
        """Test that XGBClassifier inherits from BoosterBase."""
        clf = XGBClassifier()
        from myXGBoost.booster.gradient_booster import BoosterBase
        
        assert isinstance(clf, BoosterBase)
