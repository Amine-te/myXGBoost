"""Tests for gradient boosting implementation."""

import pytest
import numpy as np
from myXGBoost.booster.gradient_booster import GradientBooster
from myXGBoost.loss.regression import MSELoss
from myXGBoost.loss.classification import LogisticLoss


class TestGradientBoosterRegression:
    """Tests for GradientBooster with regression."""
    
    def test_initial_prediction_regression(self):
        """Test initial prediction for regression (mean)."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=1,
            learning_rate=0.1
        )
        
        y = np.array([1.0, 2.0, 3.0, 4.0])
        initial_pred = booster._calculate_initial_prediction(y, is_classification=False)
        
        assert abs(initial_pred - np.mean(y)) < 1e-10
    
    def test_fit_simple_regression(self):
        """Test fitting on simple regression data."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=5,
            learning_rate=0.1,
            max_depth=2,
            random_state=42
        )
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        booster.fit(X, y)
        
        assert len(booster.trees) == 5
        assert booster.initial_prediction is not None
        assert booster.n_features_ == 1
    
    def test_predict_regression(self):
        """Test prediction for regression."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=10,
            learning_rate=0.1,
            max_depth=2,
            random_state=42
        )
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        booster.fit(X, y)
        predictions = booster.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_row_subsampling(self):
        """Test row subsampling."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=1,
            learning_rate=0.1,
            subsample=0.5,
            random_state=42
        )
        
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        booster.fit(X, y)
        
        # Should have built a tree (even with subsampling)
        assert len(booster.trees) == 1
    
    def test_column_subsampling(self):
        """Test column subsampling."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=1,
            learning_rate=0.1,
            colsample_bytree=0.5,
            random_state=42
        )
        
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        booster.fit(X, y)
        
        # Should have built a tree
        assert len(booster.trees) == 1


class TestGradientBoosterClassification:
    """Tests for GradientBooster with classification."""
    
    def test_initial_prediction_classification(self):
        """Test initial prediction for classification (log-odds)."""
        booster = GradientBooster(
            loss_function=LogisticLoss(),
            n_estimators=1,
            learning_rate=0.1
        )
        
        y = np.array([0.0, 0.0, 1.0, 1.0])
        initial_pred = booster._calculate_initial_prediction(y, is_classification=True)
        
        # p = 0.5, log-odds = log(0.5 / 0.5) = log(1) = 0
        assert abs(initial_pred) < 1e-10
    
    def test_fit_simple_classification(self):
        """Test fitting on simple classification data."""
        booster = GradientBooster(
            loss_function=LogisticLoss(),
            n_estimators=5,
            learning_rate=0.1,
            max_depth=2,
            random_state=42
        )
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        
        booster.fit(X, y)
        
        assert len(booster.trees) == 5
        assert booster.initial_prediction is not None
    
    def test_predict_proba_classification(self):
        """Test probability prediction for classification."""
        booster = GradientBooster(
            loss_function=LogisticLoss(),
            n_estimators=10,
            learning_rate=0.1,
            max_depth=2,
            random_state=42
        )
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        
        booster.fit(X, y)
        proba = booster.predict_proba(X)
        
        assert proba.shape == (4, 2)
        # Probabilities should sum to 1
        np.testing.assert_almost_equal(proba.sum(axis=1), np.ones(4))
        # Probabilities should be in [0, 1]
        assert (proba >= 0).all()
        assert (proba <= 1).all()


class TestEarlyStopping:
    """Tests for early stopping."""
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=100,
            learning_rate=0.5,  # Higher learning rate to cause overfitting faster
            max_depth=5,  # Deeper trees to capture noise
            random_state=42
        )
        
        # Create training data with noise
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + 0.1 * np.random.randn(50)
        
        # Create validation data (same signal, different noise)
        X_val = np.random.randn(20, 3)
        y_val = X_val[:, 0] + 0.5 * X_val[:, 1] + 0.5 * np.random.randn(20)
        
        # Simple metric: mean squared error
        def mse_metric(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)
        
        booster.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=mse_metric,
            early_stopping_rounds=5
        )
        
        # Should have stopped early (before 100 iterations)
        assert len(booster.trees) < 100
        assert booster.best_iteration_ is not None
    
    def test_eval_results(self):
        """Test evaluation results storage."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=10,
            learning_rate=0.1,
            max_depth=2,
            random_state=42
        )
        
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([1.0, 2.0, 3.0, 4.0])
        
        X_val = np.array([[1.5], [2.5]])
        y_val = np.array([1.5, 2.5])
        
        def mse_metric(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)
        
        booster.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=mse_metric
        )
        
        # Should have evaluation results for each iteration
        assert len(booster.eval_results) == len(booster.trees)
        assert 'eval_0' in booster.eval_results[0]


class TestAdditiveModel:
    """Tests for additive model behavior."""
    
    def test_prediction_update(self):
        """Test that predictions are updated additively."""
        booster = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=3,
            learning_rate=0.1,
            max_depth=1,
            random_state=42
        )
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        booster.fit(X, y)
        
        # Predictions should start from initial prediction
        predictions = booster._predict_raw(X)
        
        # Should be different from initial prediction (trees have contributed)
        assert not np.allclose(predictions, booster.initial_prediction)
    
    def test_learning_rate_effect(self):
        """Test that learning rate affects predictions."""
        booster1 = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=5,
            learning_rate=0.01,
            max_depth=2,
            random_state=42
        )
        
        booster2 = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=5,
            learning_rate=0.5,
            max_depth=2,
            random_state=42
        )
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        booster1.fit(X, y)
        booster2.fit(X, y)
        
        pred1 = booster1.predict(X)
        pred2 = booster2.predict(X)
        
        # Higher learning rate should lead to different predictions
        assert not np.allclose(pred1, pred2)
