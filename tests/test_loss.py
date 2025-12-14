"""Tests for loss functions."""

import pytest
import numpy as np
from myXGBoost.loss.regression import MSELoss
from myXGBoost.loss.classification import LogisticLoss, sigmoid


class TestMSELoss:
    """Tests for MSE loss function."""
    
    def test_grad_formula(self):
        """Test that gradient formula is correct: grad = pred - y."""
        loss_fn = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        grad = loss_fn.grad(y_true, y_pred)
        expected_grad = y_pred - y_true
        
        np.testing.assert_array_almost_equal(grad, expected_grad)
    
    def test_hess_formula(self):
        """Test that hessian is constant: hess = 1."""
        loss_fn = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        hess = loss_fn.hess(y_true, y_pred)
        expected_hess = np.ones_like(y_pred)
        
        np.testing.assert_array_almost_equal(hess, expected_hess)
    
    def test_grad_hess_combined(self):
        """Test grad_hess method returns correct values."""
        loss_fn = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        grad, hess = loss_fn.grad_hess(y_true, y_pred)
        
        # Verify against individual methods
        expected_grad = loss_fn.grad(y_true, y_pred)
        expected_hess = loss_fn.hess(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(grad, expected_grad)
        np.testing.assert_array_almost_equal(hess, expected_hess)
    
    def test_loss_value(self):
        """Test loss computation."""
        loss_fn = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        loss = loss_fn.loss(y_true, y_pred)
        
        # MSE = mean((pred - true)^2) / 2
        expected_loss = np.mean((y_pred - y_true) ** 2) / 2.0
        
        assert abs(loss - expected_loss) < 1e-10
    
    def test_perfect_prediction(self):
        """Test that perfect predictions yield zero gradient and loss."""
        loss_fn = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()
        
        grad = loss_fn.grad(y_true, y_pred)
        loss = loss_fn.loss(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(grad, np.zeros_like(y_pred))
        assert abs(loss) < 1e-10
    
    def test_large_values(self):
        """Test with large values."""
        loss_fn = MSELoss()
        y_true = np.array([1000.0, 2000.0])
        y_pred = np.array([1001.0, 1999.0])
        
        grad = loss_fn.grad(y_true, y_pred)
        hess = loss_fn.hess(y_true, y_pred)
        
        # Grad should be pred - true
        expected_grad = y_pred - y_true
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # Hess should still be ones
        np.testing.assert_array_almost_equal(hess, np.ones_like(y_pred))


class TestLogisticLoss:
    """Tests for logistic loss function."""
    
    def test_sigmoid_function(self):
        """Test sigmoid helper function."""
        x = np.array([0.0, 1.0, -1.0, 10.0, -10.0])
        result = sigmoid(x)
        
        # Sigmoid(0) = 0.5
        assert abs(result[0] - 0.5) < 1e-10
        
        # Sigmoid should be in (0, 1)
        assert (result > 0).all()
        assert (result < 1).all()
        
        # Sigmoid(-x) = 1 - sigmoid(x)
        assert abs(result[2] - (1 - result[1])) < 1e-10
    
    def test_grad_formula(self):
        """Test that gradient formula is correct: grad = sigmoid(pred) - y."""
        loss_fn = LogisticLoss()
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.5, -0.5, 1.0, -1.0])
        
        grad = loss_fn.grad(y_true, y_pred)
        p = sigmoid(y_pred)
        expected_grad = p - y_true
        
        np.testing.assert_array_almost_equal(grad, expected_grad)
    
    def test_hess_formula(self):
        """Test that hessian formula is correct: hess = p * (1 - p)."""
        loss_fn = LogisticLoss()
        y_pred = np.array([0.0, 1.0, -1.0, 2.0])
        
        hess = loss_fn.hess(None, y_pred)  # y_true not needed for hess
        p = sigmoid(y_pred)
        expected_hess = p * (1 - p)
        
        np.testing.assert_array_almost_equal(hess, expected_hess)
    
    def test_grad_hess_combined(self):
        """Test grad_hess method returns correct values."""
        loss_fn = LogisticLoss()
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.5, -0.5, 1.0, -1.0])
        
        grad, hess = loss_fn.grad_hess(y_true, y_pred)
        
        # Verify against individual methods
        expected_grad = loss_fn.grad(y_true, y_pred)
        expected_hess = loss_fn.hess(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(grad, expected_grad)
        np.testing.assert_array_almost_equal(hess, expected_hess)
    
    def test_loss_value(self):
        """Test loss computation."""
        loss_fn = LogisticLoss()
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.5, -0.5, 1.0, -1.0])
        
        loss = loss_fn.loss(y_true, y_pred)
        
        # Loss should be positive
        assert loss > 0
        
        # Loss should be finite
        assert np.isfinite(loss)
    
    def test_perfect_prediction(self):
        """Test that perfect predictions yield small gradients."""
        loss_fn = LogisticLoss()
        y_true = np.array([0.0, 1.0])
        # Large positive prediction for class 1, large negative for class 0
        y_pred = np.array([-10.0, 10.0])
        
        grad = loss_fn.grad(y_true, y_pred)
        
        # Gradients should be very small (close to zero)
        assert np.all(np.abs(grad) < 0.01)
    
    def test_extreme_predictions(self):
        """Test with extreme prediction values."""
        loss_fn = LogisticLoss()
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([-500.0, 500.0])  # Extreme values
        
        # Should not raise overflow errors
        grad = loss_fn.grad(y_true, y_pred)
        hess = loss_fn.hess(y_true, y_pred)
        loss = loss_fn.loss(y_true, y_pred)
        
        # All should be finite
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))
        assert np.isfinite(loss)
        
        # Gradients should be in reasonable range
        assert np.all(np.abs(grad) <= 1.0)
        
        # Hessians should be in [0, 0.25] (max when p=0.5)
        # Note: For extreme values, hessian can be exactly 0.0 due to numerical precision
        # (when p is very close to 0 or 1, p*(1-p) underflows to 0)
        assert np.all(hess >= 0)
        assert np.all(hess <= 0.25)
    
    def test_hessian_properties(self):
        """Test that hessian has correct mathematical properties."""
        loss_fn = LogisticLoss()
        y_pred = np.linspace(-5, 5, 100)
        
        hess = loss_fn.hess(None, y_pred)
        
        # Hessian should always be positive
        assert np.all(hess > 0)
        
        # Hessian should be symmetric (max at pred=0, where p=0.5)
        # Maximum hessian is 0.25 when p = 0.5 (pred = 0)
        max_hess_idx = np.argmax(hess)
        assert abs(y_pred[max_hess_idx]) < 0.1  # Should be near 0
        
        # Hessian should approach 0 as |pred| -> infinity
        assert hess[0] < 0.01  # At pred = -5
        assert hess[-1] < 0.01  # At pred = 5
    
    def test_gradient_properties(self):
        """Test that gradient has correct mathematical properties."""
        loss_fn = LogisticLoss()
        
        # For y_true = 0: grad = sigmoid(pred) - 0 = sigmoid(pred)
        # Should be in (0, 1)
        y_true_0 = np.array([0.0])
        y_pred = np.array([0.0])
        grad_0 = loss_fn.grad(y_true_0, y_pred)
        assert 0 < grad_0[0] < 1
        
        # For y_true = 1: grad = sigmoid(pred) - 1
        # Should be in (-1, 0)
        y_true_1 = np.array([1.0])
        grad_1 = loss_fn.grad(y_true_1, y_pred)
        assert -1 < grad_1[0] < 0


class TestLossFunctionInterface:
    """Tests for loss function interface and extensibility."""
    
    def test_loss_function_is_abstract(self):
        """Test that LossFunction cannot be instantiated."""
        from myXGBoost.loss.base import LossFunction
        
        with pytest.raises(TypeError):
            LossFunction()
    
    def test_mse_inherits_regression_loss(self):
        """Test that MSELoss inherits from RegressionLoss."""
        from myXGBoost.loss.base import RegressionLoss
        
        mse = MSELoss()
        assert isinstance(mse, RegressionLoss)
    
    def test_logistic_inherits_classification_loss(self):
        """Test that LogisticLoss inherits from ClassificationLoss."""
        from myXGBoost.loss.base import ClassificationLoss
        
        logistic = LogisticLoss()
        assert isinstance(logistic, ClassificationLoss)
    
    def test_loss_function_interface(self):
        """Test that all loss functions implement required methods."""
        mse = MSELoss()
        logistic = LogisticLoss()
        
        # Both should have loss, grad, hess, grad_hess methods
        assert hasattr(mse, 'loss')
        assert hasattr(mse, 'grad')
        assert hasattr(mse, 'hess')
        assert hasattr(mse, 'grad_hess')
        
        assert hasattr(logistic, 'loss')
        assert hasattr(logistic, 'grad')
        assert hasattr(logistic, 'hess')
        assert hasattr(logistic, 'grad_hess')
    
    def test_grad_hess_consistency(self):
        """Test that grad_hess is consistent with separate grad/hess calls."""
        mse = MSELoss()
        logistic = LogisticLoss()
        
        y_true_reg = np.array([1.0, 2.0, 3.0])
        y_pred_reg = np.array([1.5, 2.5, 3.5])
        
        y_true_clf = np.array([0.0, 1.0, 0.0])
        y_pred_clf = np.array([0.5, -0.5, 1.0])
        
        # Test MSE
        grad_mse, hess_mse = mse.grad_hess(y_true_reg, y_pred_reg)
        assert np.allclose(grad_mse, mse.grad(y_true_reg, y_pred_reg))
        assert np.allclose(hess_mse, mse.hess(y_true_reg, y_pred_reg))
        
        # Test Logistic
        grad_log, hess_log = logistic.grad_hess(y_true_clf, y_pred_clf)
        assert np.allclose(grad_log, logistic.grad(y_true_clf, y_pred_clf))
        assert np.allclose(hess_log, logistic.hess(y_true_clf, y_pred_clf))
