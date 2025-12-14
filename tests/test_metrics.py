"""Tests for evaluation metrics."""

import numpy as np
import pytest
from myXGBoost.metrics.regression import rmse, mae, r2_score, RMSE, MAE, R2Score
from myXGBoost.metrics.classification import (
    accuracy_score, log_loss, auc_score, 
    Accuracy, LogLoss, AUC
)


class TestRegressionMetrics:
    """Tests for regression metrics."""
    
    def test_rmse_perfect_predictions(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert rmse(y_true, y_pred) == pytest.approx(0.0)
    
    def test_rmse_constant_error(self):
        """Test RMSE with constant error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        
        # Error = [1, 1, 1], MSE = 1, RMSE = 1
        assert rmse(y_true, y_pred) == pytest.approx(1.0)
    
    def test_rmse_with_weights(self):
        """Test RMSE with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        weights = np.array([1.0, 1.0, 2.0])
        
        # Weighted MSE = (1 + 1 + 2*1) / 4 = 1, RMSE = 1
        assert rmse(y_true, y_pred, sample_weight=weights) == pytest.approx(1.0)
    
    def test_mae_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert mae(y_true, y_pred) == pytest.approx(0.0)
    
    def test_mae_constant_error(self):
        """Test MAE with constant error."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        
        # Error = [1, 1, 1], MAE = 1
        assert mae(y_true, y_pred) == pytest.approx(1.0)
    
    def test_mae_with_weights(self):
        """Test MAE with sample weights."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        weights = np.array([1.0, 1.0, 2.0])
        
        # Weighted MAE = (1 + 1 + 2*1) / 4 = 1
        assert mae(y_true, y_pred, sample_weight=weights) == pytest.approx(1.0)
    
    def test_r2_perfect_predictions(self):
        """Test R2 with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert r2_score(y_true, y_pred) == pytest.approx(1.0)
    
    def test_r2_mean_prediction(self):
        """Test R2 when predicting mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # All mean
        
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)
    
    def test_r2_worse_than_mean(self):
        """Test R2 with predictions worse than mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed
        
        # Should be negative (worse than mean)
        assert r2_score(y_true, y_pred) < 0.0
    
    def test_rmse_class(self):
        """Test RMSE metric class."""
        metric = RMSE()
        assert metric.name == "rmse"
        assert metric.is_higher_better() is False
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        
        assert metric.score(y_true, y_pred) == pytest.approx(1.0)
    
    def test_mae_class(self):
        """Test MAE metric class."""
        metric = MAE()
        assert metric.name == "mae"
        assert metric.is_higher_better() is False
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        
        assert metric.score(y_true, y_pred) == pytest.approx(1.0)
    
    def test_r2_class(self):
        """Test R2Score metric class."""
        metric = R2Score()
        assert metric.name == "r2"
        assert metric.is_higher_better() is True
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert metric.score(y_true, y_pred) == pytest.approx(1.0)


class TestClassificationMetrics:
    """Tests for classification metrics."""
    
    def test_accuracy_perfect_predictions(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        
        assert accuracy_score(y_true, y_pred) == pytest.approx(1.0)
    
    def test_accuracy_random_predictions(self):
        """Test accuracy with random predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        
        # 4 correct out of 6
        assert accuracy_score(y_true, y_pred) == pytest.approx(4/6)
    
    def test_accuracy_with_weights(self):
        """Test accuracy with sample weights."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        weights = np.array([1.0, 1.0, 2.0, 2.0])
        
        # Correct predictions: indices 0, 2 with weights 1, 2. Total correct weight = 3
        # Total weight = 6. Score = 3 / 6 = 0.5
        assert accuracy_score(y_true, y_pred, sample_weight=weights) == pytest.approx(0.5)
    
    def test_accuracy_not_normalized(self):
        """Test accuracy without normalization."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        
        assert accuracy_score(y_true, y_pred, normalize=False) == 5
    
    def test_log_loss_perfect_predictions(self):
        """Test log loss with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.0001, 0.9999, 0.0001, 0.9999])
        
        # Very close to 0
        loss = log_loss(y_true, y_pred)
        assert loss < 0.01
    
    def test_log_loss_worst_predictions(self):
        """Test log loss with worst predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.9999, 0.0001, 0.9999, 0.0001])
        
        # Very high loss (clipped to avoid overflow)
        loss = log_loss(y_true, y_pred)
        assert loss > 5  # At least 5, but clipping limits how high
    
    def test_log_loss_random_predictions(self):
        """Test log loss with 0.5 predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        
        # log_loss = -mean(0*log(0.5) + 1*log(0.5) + ...) = log(2)
        expected = np.log(2)
        assert log_loss(y_true, y_pred) == pytest.approx(expected)
    
    def test_auc_perfect_predictions(self):
        """Test AUC with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        
        assert auc_score(y_true, y_pred) == pytest.approx(1.0)
    
    def test_auc_random_predictions(self):
        """Test AUC with similar predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.3, 0.4, 0.6, 0.7])
        
        # Good separation but not perfect - AUC should be high but not 1.0
        auc = auc_score(y_true, y_pred)
        assert 0.7 < auc <= 1.0
    
    def test_auc_worst_predictions(self):
        """Test AUC with reversed predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.9, 0.8, 0.2, 0.1])
        
        assert auc_score(y_true, y_pred) == pytest.approx(0.0)
    
    def test_accuracy_class(self):
        """Test Accuracy metric class."""
        metric = Accuracy()
        assert metric.name == "accuracy"
        assert metric.is_higher_better() is True
        
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        assert metric.score(y_true, y_pred) == pytest.approx(1.0)
    
    def test_logloss_class(self):
        """Test LogLoss metric class."""
        metric = LogLoss()
        assert metric.name == "logloss"
        assert metric.is_higher_better() is False
        
        y_true = np.array([0, 1])
        y_pred = np.array([0.5, 0.5])
        
        assert metric.score(y_true, y_pred) == pytest.approx(np.log(2))
    
    def test_auc_class(self):
        """Test AUC metric class."""
        metric = AUC()
        assert metric.name == "auc"
        assert metric.is_higher_better() is True
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        
        assert metric.score(y_true, y_pred) == pytest.approx(1.0)


class TestMetricEdgeCases:
    """Tests for edge cases in metrics."""
    
    def test_rmse_single_sample(self):
        """Test RMSE with single sample."""
        y_true = np.array([5.0])
        y_pred = np.array([3.0])
        
        assert rmse(y_true, y_pred) == pytest.approx(2.0)
    
    def test_mae_negative_errors(self):
        """Test MAE with negative errors."""
        y_true = np.array([5.0, 5.0])
        y_pred = np.array([3.0, 7.0])
        
        # Errors: [-2, 2], MAE = 2
        assert mae(y_true, y_pred) == pytest.approx(2.0)
    
    def test_r2_constant_target(self):
        """Test R2 with constant target."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        
        # When y_true is constant and perfect prediction
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)
    
    def test_auc_single_class(self):
        """Test AUC with single class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Should return 0.5 when only one class present
        assert auc_score(y_true, y_pred) == 0.5
