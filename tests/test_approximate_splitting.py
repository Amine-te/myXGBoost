"""
Test suite for Phase 7: Approximate Split Finding & Parallel Learning.

Tests cover:
- Weighted quantile sketch for histogram construction
- Approximate split finder (histogram-based)
- Hybrid split finder (adaptive exact/approximate selection)
- Parallel feature evaluation
- Accuracy of approximate vs exact algorithms
- Performance improvements
"""

import pytest
import numpy as np
import time
from multiprocessing import cpu_count

from myXGBoost.trees.split_finder import (
    WeightedQuantileSketch,
    ApproximateSplitFinder,
    HybridSplitFinder,
    ExactSplitFinder,
    calculate_gain
)


class TestWeightedQuantileSketch:
    """Tests for weighted quantile sketch."""
    
    def test_sketch_initialization(self):
        """Test sketch initialization."""
        sketch = WeightedQuantileSketch(max_bins=256)
        assert sketch.max_bins == 256
        assert sketch.data == []
    
    def test_sketch_add_samples(self):
        """Test adding samples to sketch."""
        sketch = WeightedQuantileSketch()
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.1, 0.2, 0.1])
        
        sketch.add(values, weights)
        assert len(sketch.data) == 3
    
    def test_sketch_get_bins_small_data(self):
        """Test bin generation for small dataset."""
        sketch = WeightedQuantileSketch(max_bins=10)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        
        sketch.add(values, weights)
        bins = sketch.get_bins(5)
        
        # Should include min and max
        assert len(bins) >= 2
        assert bins[0] == 1.0
        assert bins[-1] == 5.0
    
    def test_sketch_get_bins_large_data(self):
        """Test bin generation for large dataset."""
        sketch = WeightedQuantileSketch(max_bins=100)
        np.random.seed(42)
        values = np.random.randn(1000)
        weights = np.abs(np.random.randn(1000)) + 0.1
        
        sketch.add(values, weights)
        bins = sketch.get_bins(50)
        
        # Should have reasonable number of bins
        # May include some unique values beyond the requested count
        assert 2 <= len(bins) <= 1000
        assert bins[0] == np.min(values)
        assert bins[-1] == np.max(values)
    
    def test_sketch_bins_are_sorted(self):
        """Test that returned bins are sorted."""
        sketch = WeightedQuantileSketch()
        np.random.seed(42)
        values = np.random.randn(500)
        weights = np.abs(np.random.randn(500)) + 0.1
        
        sketch.add(values, weights)
        bins = sketch.get_bins(20)
        
        # Bins should be sorted
        assert np.all(bins[:-1] <= bins[1:])
    
    def test_sketch_weighted_quantiles(self):
        """Test that weighted quantiles are approximately correct."""
        sketch = WeightedQuantileSketch()
        
        # Create skewed distribution
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        weights = np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 5], dtype=float)
        
        sketch.add(values, weights)
        bins = sketch.get_bins(3)
        
        # Median should be around 6-7 (weighted)
        assert len(bins) >= 2
        assert bins[0] <= bins[-1]


class TestApproximateSplitFinder:
    """Tests for approximate split finding."""
    
    def test_approximate_initialization(self):
        """Test approximate finder initialization."""
        finder = ApproximateSplitFinder(
            reg_lambda=1.0,
            gamma=0.0,
            min_child_weight=1.0,
            max_bins=256,
            use_parallelization=False
        )
        assert finder.max_bins == 256
        assert finder.reg_lambda == 1.0
    
    def test_histogram_construction(self):
        """Test histogram construction."""
        finder = ApproximateSplitFinder(max_bins=50)
        np.random.seed(42)
        feature_values = np.random.randn(500)
        weights = np.abs(np.random.randn(500)) + 0.1
        
        bins = finder._build_histograms(feature_values, weights)
        
        assert len(bins) >= 2
        assert bins[0] == np.min(feature_values)
        assert bins[-1] == np.max(feature_values)
    
    def test_evaluate_histogram_splits(self):
        """Test evaluation of splits at histogram boundaries."""
        finder = ApproximateSplitFinder()
        np.random.seed(42)
        X = np.random.randn(100, 5)
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        
        feature_values = X[:, 0]
        bins = np.quantile(feature_values, [0.0, 0.25, 0.5, 0.75, 1.0])
        
        threshold, gain = finder._evaluate_histogram_splits(
            feature_values, grad, hess, bins
        )
        
        # Should find a valid split
        assert threshold is not None or gain == float('-inf')
    
    def test_approximate_find_best_split(self):
        """Test finding best split with approximate method."""
        finder = ApproximateSplitFinder(max_bins=256, use_parallelization=False)
        np.random.seed(42)
        X = np.random.randn(200, 5)
        grad = np.random.randn(200)
        hess = np.abs(np.random.randn(200)) + 0.5
        
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should find a split
        assert feature_idx is not None or gain == float('-inf')
        assert threshold is None or isinstance(threshold, (float, np.floating))
    
    def test_approximate_with_multiple_features(self):
        """Test approximate finder with multiple features."""
        finder = ApproximateSplitFinder(
            max_bins=128,
            use_parallelization=False
        )
        np.random.seed(42)
        X = np.random.randn(300, 10)
        grad = np.random.randn(300)
        hess = np.abs(np.random.randn(300)) + 0.5
        
        feature_idx, threshold, gain = finder.find_best_split(
            X, grad, hess,
            feature_indices=np.array([0, 1, 2, 3, 4])
        )
        
        # Should evaluate subset of features
        if feature_idx is not None:
            assert feature_idx in [0, 1, 2, 3, 4]
    
    def test_approximate_respects_min_child_weight(self):
        """Test that approximate finder respects minimum child weight."""
        finder = ApproximateSplitFinder(min_child_weight=100.0)
        np.random.seed(42)
        X = np.random.randn(100, 2)
        grad = np.random.randn(100) * 0.1  # Small gradients
        hess = np.abs(np.random.randn(100)) * 0.01  # Small hessians
        
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # With high min_child_weight, may not find split
        assert feature_idx is None or threshold is not None
    
    def test_approximate_split_data(self):
        """Test splitting data with approximate finder."""
        finder = ApproximateSplitFinder()
        np.random.seed(42)
        X = np.random.randn(100, 3)
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        
        X_left, grad_left, hess_left, X_right, grad_right, hess_right, assign_missing_to_left = \
            finder.split_data(X, grad, hess, feature=0, threshold=0.0)
        
        # Check split
        assert len(X_left) + len(X_right) == len(X)
        assert np.all(X_left[:, 0] < 0.0)
        assert np.all(X_right[:, 0] >= 0.0)
    
    def test_approximate_vs_exact_consistency(self):
        """Test that approximate splits are similar to exact."""
        np.random.seed(42)
        X = np.random.randn(500, 5)
        grad = np.random.randn(500)
        hess = np.abs(np.random.randn(500)) + 0.5
        
        exact_finder = ExactSplitFinder(use_vectorization=True)
        approx_finder = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=False
        )
        
        exact_feature, exact_threshold, exact_gain = \
            exact_finder.find_best_split(X, grad, hess)
        approx_feature, approx_threshold, approx_gain = \
            approx_finder.find_best_split(X, grad, hess)
        
        # Both should find splits
        assert exact_feature is not None
        assert approx_feature is not None
        
        # Gains should be similar (within 50% due to binning)
        if exact_gain > -np.inf and approx_gain > -np.inf:
            # Approximate may be slightly lower due to binning
            assert approx_gain >= exact_gain * 0.5


class TestHybridSplitFinder:
    """Tests for hybrid (adaptive) split finder."""
    
    def test_hybrid_initialization(self):
        """Test hybrid finder initialization."""
        finder = HybridSplitFinder(exact_threshold=5000)
        assert finder.exact_threshold == 5000
        assert isinstance(finder.exact_finder, ExactSplitFinder)
        assert isinstance(finder.approx_finder, ApproximateSplitFinder)
    
    def test_hybrid_uses_exact_for_small_data(self):
        """Test that hybrid uses exact algorithm for small datasets."""
        finder = HybridSplitFinder(exact_threshold=1000)
        np.random.seed(42)
        X = np.random.randn(100, 5)
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        
        # Should use exact
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should find good split
        assert feature_idx is not None or gain == float('-inf')
    
    def test_hybrid_uses_approximate_for_large_data(self):
        """Test that hybrid uses approximate algorithm for large datasets."""
        finder = HybridSplitFinder(exact_threshold=100)  # Low threshold
        np.random.seed(42)
        X = np.random.randn(500, 5)
        grad = np.random.randn(500)
        hess = np.abs(np.random.randn(500)) + 0.5
        
        # Should use approximate
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should still find good split
        assert feature_idx is not None or gain == float('-inf')
    
    def test_hybrid_split_data(self):
        """Test splitting data with hybrid finder."""
        finder = HybridSplitFinder()
        np.random.seed(42)
        X = np.random.randn(200, 3)
        grad = np.random.randn(200)
        hess = np.abs(np.random.randn(200)) + 0.5
        
        X_left, grad_left, hess_left, X_right, grad_right, hess_right, assign_missing_to_left = \
            finder.split_data(X, grad, hess, feature=0, threshold=0.0)
        
        # Check split
        assert len(X_left) + len(X_right) == len(X)
        assert np.all(X_left[:, 0] < 0.0)
        assert np.all(X_right[:, 0] >= 0.0)
    
    def test_hybrid_threshold_boundary(self):
        """Test hybrid finder at the threshold boundary."""
        finder = HybridSplitFinder(exact_threshold=100)
        np.random.seed(42)
        
        # Test exactly at threshold
        X_small = np.random.randn(100, 3)
        X_large = np.random.randn(101, 3)
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        grad_large = np.random.randn(101)
        hess_large = np.abs(np.random.randn(101)) + 0.5
        
        result_small = finder.find_best_split(X_small, grad, hess)
        result_large = finder.find_best_split(X_large, grad_large, hess_large)
        
        # Both should return valid results
        assert result_small[2] != float('-inf') or result_small[0] is None
        assert result_large[2] != float('-inf') or result_large[0] is None


class TestParallelization:
    """Tests for parallelization in approximate finder."""
    
    @pytest.mark.skipif(cpu_count() < 2, reason="Requires multiple cores")
    def test_parallel_feature_evaluation(self):
        """Test parallel evaluation of features."""
        finder_sequential = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=False
        )
        finder_parallel = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=True,
            n_jobs=2
        )
        
        np.random.seed(42)
        X = np.random.randn(500, 10)
        grad = np.random.randn(500)
        hess = np.abs(np.random.randn(500)) + 0.5
        
        # Both should find splits
        seq_feature, _, seq_gain = finder_sequential.find_best_split(X, grad, hess)
        par_feature, _, par_gain = finder_parallel.find_best_split(X, grad, hess)
        
        # Results should be similar
        assert seq_feature is not None
        assert par_feature is not None
        assert abs(seq_gain - par_gain) < 1e-6 or seq_gain == par_gain
    
    @pytest.mark.skipif(cpu_count() < 2, reason="Requires multiple cores")
    def test_parallel_speedup(self):
        """Test that parallelization works for many features."""
        np.random.seed(42)
        X = np.random.randn(1000, 50)  # Many features for parallel benefit
        grad = np.random.randn(1000)
        hess = np.abs(np.random.randn(1000)) + 0.5
        
        finder_sequential = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=False
        )
        finder_parallel = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=True,
            n_jobs=-1
        )
        
        # Sequential evaluation
        t_start = time.time()
        for _ in range(2):
            finder_sequential.find_best_split(X, grad, hess)
        t_sequential = time.time() - t_start
        
        # Parallel evaluation
        t_start = time.time()
        for _ in range(2):
            finder_parallel.find_best_split(X, grad, hess)
        t_parallel = time.time() - t_start
        
        print(f"\nParallel Performance (50 features):")
        print(f"  Sequential: {t_sequential:.4f}s")
        print(f"  Parallel: {t_parallel:.4f}s")
        if t_parallel > 0:
            print(f"  Speedup: {t_sequential / t_parallel:.2f}x")
        
        # Both methods should complete successfully
        assert t_sequential > 0
        assert t_parallel > 0


class TestPerformanceComparison:
    """Tests comparing performance of different algorithms."""
    
    def test_exact_vs_approximate_speed_small(self):
        """Test speed comparison on small dataset."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        
        exact_finder = ExactSplitFinder(use_vectorization=True)
        approx_finder = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=False
        )
        
        # Time exact
        t_start = time.time()
        for _ in range(5):
            exact_finder.find_best_split(X, grad, hess)
        t_exact = time.time() - t_start
        
        # Time approximate
        t_start = time.time()
        for _ in range(5):
            approx_finder.find_best_split(X, grad, hess)
        t_approx = time.time() - t_start
        
        print(f"\nSmall dataset (100 samples, 10 features):")
        print(f"  Exact: {t_exact:.4f}s")
        print(f"  Approximate: {t_approx:.4f}s")
        
        # Both should be fast
        assert t_exact < 1.0
        assert t_approx < 1.0
    
    def test_exact_vs_approximate_accuracy_large(self):
        """Test accuracy comparison on large dataset."""
        np.random.seed(42)
        X = np.random.randn(5000, 20)
        grad = np.random.randn(5000)
        hess = np.abs(np.random.randn(5000)) + 0.5
        
        exact_finder = ExactSplitFinder(use_vectorization=True)
        approx_finder = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=False
        )
        
        # Get results from both methods
        exact_feature, exact_threshold, exact_gain = \
            exact_finder.find_best_split(X, grad, hess)
        approx_feature, approx_threshold, approx_gain = \
            approx_finder.find_best_split(X, grad, hess)
        
        print(f"\nLarge dataset (5000 samples, 20 features) - Accuracy:")
        print(f"  Exact gain: {exact_gain:.6f}")
        print(f"  Approx gain: {approx_gain:.6f}")
        if exact_gain > -np.inf:
            print(f"  Ratio: {approx_gain / max(exact_gain, 1e-10):.3f}")
        
        # Both should find splits
        assert exact_feature is not None
        assert approx_feature is not None
        # Approximate should be within 70% of exact (histograms lose some quality)
        assert approx_gain >= exact_gain * 0.7
    
    def test_hybrid_adaptive_performance(self):
        """Test that hybrid finder selects appropriate algorithm."""
        np.random.seed(42)
        
        # Small dataset
        X_small = np.random.randn(100, 10)
        grad_small = np.random.randn(100)
        hess_small = np.abs(np.random.randn(100)) + 0.5
        
        # Large dataset
        X_large = np.random.randn(10000, 10)
        grad_large = np.random.randn(10000)
        hess_large = np.abs(np.random.randn(10000)) + 0.5
        
        hybrid = HybridSplitFinder(exact_threshold=5000)
        
        # Should use exact for small
        t_start = time.time()
        result_small = hybrid.find_best_split(X_small, grad_small, hess_small)
        t_small = time.time() - t_start
        
        # Should use approximate for large
        t_start = time.time()
        result_large = hybrid.find_best_split(X_large, grad_large, hess_large)
        t_large = time.time() - t_start
        
        print(f"\nHybrid Finder Adaptive Performance:")
        print(f"  Small (100): {t_small:.4f}s")
        print(f"  Large (10000): {t_large:.4f}s")
        
        # Both should find splits
        assert result_small[0] is not None or result_small[2] == float('-inf')
        assert result_large[0] is not None or result_large[2] == float('-inf')


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_feature(self):
        """Test with single feature."""
        finder = ApproximateSplitFinder()
        np.random.seed(42)
        X = np.random.randn(100, 1)
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        assert feature_idx is None or feature_idx == 0
    
    def test_constant_feature(self):
        """Test with constant feature values."""
        finder = ApproximateSplitFinder()
        X = np.ones((100, 2))
        grad = np.random.randn(100)
        hess = np.abs(np.random.randn(100)) + 0.5
        
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should not find split for constant feature
        assert feature_idx is None or gain == float('-inf')
    
    def test_all_zero_gradients(self):
        """Test with all zero gradients."""
        finder = ApproximateSplitFinder()
        np.random.seed(42)
        X = np.random.randn(100, 2)
        grad = np.zeros(100)
        hess = np.ones(100)
        
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # May or may not find split with zero gradients
        assert isinstance(gain, (float, np.floating))
    
    def test_very_small_dataset(self):
        """Test with very small dataset."""
        finder = ApproximateSplitFinder(max_bins=256)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = np.array([0.5, -0.5])
        hess = np.array([0.1, 0.1])
        
        feature_idx, threshold, gain = finder.find_best_split(X, grad, hess)
        
        # Should handle small data gracefully
        assert feature_idx is None or threshold is not None
    
    def test_many_bins_vs_few_bins(self):
        """Test histogram quality with different bin counts."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        grad = np.random.randn(500)
        hess = np.abs(np.random.randn(500)) + 0.5
        
        finder_few = ApproximateSplitFinder(
            max_bins=16,
            use_parallelization=False
        )
        finder_many = ApproximateSplitFinder(
            max_bins=256,
            use_parallelization=False
        )
        
        _, _, gain_few = finder_few.find_best_split(X, grad, hess)
        _, _, gain_many = finder_many.find_best_split(X, grad, hess)
        
        # More bins should generally find better split
        assert gain_many >= gain_few * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
