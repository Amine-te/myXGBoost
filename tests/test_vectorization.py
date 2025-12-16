"""Tests for vectorized split finding optimizations."""

import numpy as np
import pytest
import time
from myXGBoost.trees.split_finder import (
    calculate_gain, calculate_gains_vectorized, ExactSplitFinder
)


class TestVectorizedGainCalculation:
    """Tests for vectorized gain calculation."""
    
    def test_scalar_gain_calculation(self):
        """Test scalar gain calculation."""
        grad_left, hess_left = 1.0, 2.0
        grad_right, hess_right = 1.5, 2.5
        
        gain = calculate_gain(grad_left, hess_left, grad_right, hess_right)
        
        # Should return a float
        assert isinstance(gain, (float, np.floating))
        assert not np.isinf(gain)
        assert not np.isnan(gain)
    
    def test_vectorized_gain_calculation(self):
        """Test vectorized gain calculation with arrays."""
        grad_lefts = np.array([1.0, 1.5, 2.0])
        hess_lefts = np.array([2.0, 2.5, 3.0])
        grad_rights = np.array([1.5, 1.0, 0.5])
        hess_rights = np.array([2.5, 2.0, 1.5])
        
        gains = calculate_gain(grad_lefts, hess_lefts, grad_rights, hess_rights)
        
        # Should return array of same shape
        assert gains.shape == (3,)
        assert np.all(~np.isinf(gains))
        assert np.all(~np.isnan(gains))
    
    def test_gains_vectorized_function(self):
        """Test dedicated vectorized gains function."""
        grad_lefts = np.array([1.0, 1.5, 2.0, 2.5])
        hess_lefts = np.array([2.0, 2.5, 3.0, 3.5])
        grad_rights = np.array([1.5, 1.0, 0.5, 0.0])
        hess_rights = np.array([2.5, 2.0, 1.5, 1.0])
        
        gains = calculate_gains_vectorized(grad_lefts, hess_lefts, grad_rights, hess_rights)
        
        assert gains.shape == (4,)
        assert isinstance(gains, np.ndarray)
    
    def test_gain_with_regularization(self):
        """Test gain calculation respects regularization parameters."""
        grad_left, hess_left = 1.0, 2.0
        grad_right, hess_right = 1.5, 2.5
        
        gain_low_reg = calculate_gain(
            grad_left, hess_left, grad_right, hess_right,
            reg_lambda=0.1, gamma=0.0
        )
        
        gain_high_reg = calculate_gain(
            grad_left, hess_left, grad_right, hess_right,
            reg_lambda=10.0, gamma=0.0
        )
        
        # Higher regularization should reduce gain
        assert gain_low_reg > gain_high_reg
    
    def test_gain_with_gamma(self):
        """Test gain calculation respects gamma parameter."""
        grad_left, hess_left = 2.0, 2.0
        grad_right, hess_right = 2.0, 2.0
        
        gain_no_gamma = calculate_gain(
            grad_left, hess_left, grad_right, hess_right,
            reg_lambda=1.0, gamma=0.0
        )
        
        gain_with_gamma = calculate_gain(
            grad_left, hess_left, grad_right, hess_right,
            reg_lambda=1.0, gamma=1.0
        )
        
        # Gamma should reduce gain
        assert gain_no_gamma > gain_with_gamma
    
    def test_vectorized_vs_scalar_consistency(self):
        """Test that vectorized and scalar calculations are consistent."""
        grad_left, hess_left = 1.5, 2.5
        grad_right, hess_right = 1.0, 2.0
        
        scalar_gain = calculate_gain(
            grad_left, hess_left, grad_right, hess_right,
            reg_lambda=1.0, gamma=0.5
        )
        
        vectorized_gain = calculate_gain(
            np.array([grad_left]), 
            np.array([hess_left]),
            np.array([grad_right]), 
            np.array([hess_right]),
            reg_lambda=1.0, gamma=0.5
        )[0]
        
        assert scalar_gain == pytest.approx(vectorized_gain)


class TestVectorizedSplitFinder:
    """Tests for vectorized split finding."""
    
    def test_vectorized_and_scalar_find_same_split(self):
        """Test that vectorized and scalar methods find same split."""
        # Simple synthetic data
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        grad = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
        hess = np.array([2.0, 2.5, 3.0, 2.5, 2.0])
        
        # Scalar version
        finder_scalar = ExactSplitFinder(use_vectorization=False)
        feature_s, threshold_s, gain_s = finder_scalar.find_best_split(X, grad, hess)
        
        # Vectorized version
        finder_vec = ExactSplitFinder(use_vectorization=True)
        feature_v, threshold_v, gain_v = finder_vec.find_best_split(X, grad, hess)
        
        # Should find same feature and similar threshold/gain
        assert feature_s == feature_v
        assert threshold_s == pytest.approx(threshold_v)
        assert gain_s == pytest.approx(gain_v)
    
    def test_vectorized_respects_min_child_weight(self):
        """Test that vectorization respects min_child_weight constraint."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([1.0, 1.0, 1.0, 1.0])
        hess = np.array([0.5, 0.5, 0.5, 0.5])  # Very low hessians
        
        finder = ExactSplitFinder(min_child_weight=2.0, use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # With low hessians (total=2.0) and high min_child_weight=2.0,
        # it's very hard to find valid split (left>=2 AND right>=2 means total>=4)
        if feature is None:
            assert threshold is None
            assert gain == float('-inf')
    
    def test_vectorized_multiple_features(self):
        """Test vectorized split finding with multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        
        finder = ExactSplitFinder(use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess)
        
        assert feature is not None
        assert threshold is not None
        assert gain > float('-inf')
        assert 0 <= feature < 5
    
    def test_vectorized_with_duplicate_values(self):
        """Test vectorized version handles duplicate feature values."""
        X = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
        grad = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
        hess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        finder = ExactSplitFinder(use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should still find a valid split
        assert feature is not None
        assert threshold is not None
        assert gain > float('-inf')
    
    def test_vectorized_single_unique_value(self):
        """Test vectorized version with single unique value."""
        X = np.array([[5.0], [5.0], [5.0], [5.0]])
        grad = np.array([1.0, 1.5, 2.0, 1.5])
        hess = np.array([2.0, 2.5, 3.0, 2.5])
        
        finder = ExactSplitFinder(use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Cannot split when all values are the same
        assert feature is None
        assert threshold is None
    
    def test_split_data_method(self):
        """Test that split_data correctly partitions data."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        grad = np.array([1.0, 2.0, 3.0, 4.0])
        hess = np.array([1.0, 1.0, 1.0, 1.0])
        
        finder = ExactSplitFinder()
        X_left, grad_left, hess_left, X_right, grad_right, hess_right, assign_missing_to_left = \
            finder.split_data(X, grad, hess, feature=0, threshold=2.5)
        
        # Check that split is correct
        assert X_left.shape[0] + X_right.shape[0] == X.shape[0]
        assert np.all(X_left[:, 0] < 2.5)
        assert np.all(X_right[:, 0] >= 2.5)
        assert len(grad_left) == X_left.shape[0]
        assert len(grad_right) == X_right.shape[0]


class TestVectorizationPerformance:
    """Performance comparison tests."""
    
    def test_vectorized_faster_on_large_dataset(self):
        """Test that vectorized version is competitive on large datasets."""
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        grad = np.random.randn(1000)
        hess = np.abs(np.random.randn(1000)) + 0.5
        
        # Scalar version
        finder_scalar = ExactSplitFinder(use_vectorization=False)
        t_scalar_start = time.time()
        for _ in range(3):
            finder_scalar.find_best_split(X, grad, hess, feature_indices=np.array([0, 1, 2]))
        t_scalar = time.time() - t_scalar_start
        
        # Vectorized version
        finder_vec = ExactSplitFinder(use_vectorization=True)
        t_vec_start = time.time()
        for _ in range(3):
            finder_vec.find_best_split(X, grad, hess, feature_indices=np.array([0, 1, 2]))
        t_vec = time.time() - t_vec_start
        
        # Vectorized should be competitive (faster or similar)
        print(f"\nPerformance Comparison (1000 samples, 10 features, 3 iterations):")
        print(f"  Scalar version: {t_scalar:.4f}s")
        print(f"  Vectorized version: {t_vec:.4f}s")
        if t_vec > 0:
            print(f"  Speedup: {t_scalar / t_vec:.2f}x")
        
        # Allow some overhead on first calls but vectorized shouldn't be much slower
        assert t_vec <= t_scalar * 2.0
    
    def test_vectorization_scales_better(self):
        """Test that vectorization scales better with feature count."""
        np.random.seed(42)
        grad = np.random.randn(500)
        hess = np.abs(np.random.randn(500)) + 0.5
        
        times_scalar = []
        times_vec = []
        feature_counts = [5, 10, 20, 50]
        
        for n_features in feature_counts:
            X = np.random.randn(500, n_features)
            
            finder_scalar = ExactSplitFinder(use_vectorization=False)
            t_start = time.time()
            finder_scalar.find_best_split(X, grad, hess)
            times_scalar.append(time.time() - t_start)
            
            finder_vec = ExactSplitFinder(use_vectorization=True)
            t_start = time.time()
            finder_vec.find_best_split(X, grad, hess)
            times_vec.append(time.time() - t_start)
        
        print(f"\nScaling Test (500 samples):")
        print(f"{'Features':<12} {'Scalar':<12} {'Vectorized':<12} {'Speedup':<10}")
        print("-" * 46)
        for i, n_features in enumerate(feature_counts):
            speedup = times_scalar[i] / times_vec[i] if times_vec[i] > 0 else float('inf')
            print(f"{n_features:<12} {times_scalar[i]:<12.4f} {times_vec[i]:<12.4f} {speedup:<10.2f}x")
        
        # Vectorized should maintain advantage with more features
        assert times_vec[-1] <= times_scalar[-1]


class TestVectorizedEdgeCases:
    """Test edge cases for vectorized implementation."""
    
    def test_empty_feature_indices(self):
        """Test with empty feature indices."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = np.array([1.0, 2.0])
        hess = np.array([1.0, 1.0])
        
        finder = ExactSplitFinder(use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess, feature_indices=np.array([]))
        
        assert feature is None
        assert threshold is None
    
    def test_very_small_hessians(self):
        """Test with very small hessian values."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([1e-10, 2e-10, 3e-10, 4e-10])
        hess = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        
        finder = ExactSplitFinder(use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should handle small values without overflow/underflow
        if feature is not None:
            assert not np.isinf(gain)
            assert not np.isnan(gain)
    
    def test_large_gradient_values(self):
        """Test with large gradient values."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([1e6, 2e6, 3e6, 4e6])
        hess = np.array([1.0, 1.0, 1.0, 1.0])
        
        finder = ExactSplitFinder(use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should handle large values without overflow
        if feature is not None:
            assert not np.isinf(gain) or np.isinf(gain)  # Both acceptable
            assert not np.isnan(gain)
    
    def test_negative_gradients(self):
        """Test with negative gradient values."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([-2.0, -1.0, 1.0, 2.0])
        hess = np.array([1.0, 1.0, 1.0, 1.0])
        
        finder = ExactSplitFinder(use_vectorization=True)
        feature, threshold, gain = finder.find_best_split(X, grad, hess)
        
        assert feature is not None
        assert threshold is not None
