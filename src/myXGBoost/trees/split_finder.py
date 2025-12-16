"""Split finding algorithms (exact, approximate, histogram-based)."""

import numpy as np
from typing import Tuple, Optional, Union
from multiprocessing import Pool, cpu_count
import warnings
from joblib import Parallel, delayed
from myXGBoost.utils.parallel import build_histogram_parallel


def calculate_gain(
    grad_left: Union[float, np.ndarray],
    hess_left: Union[float, np.ndarray],
    grad_right: Union[float, np.ndarray],
    hess_right: Union[float, np.ndarray],
    reg_lambda: float = 1.0,
    gamma: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate split gain using XGBoost formula.
    
    Vectorized version that works with scalars or arrays.
    
    Formula:
    Gain = 0.5 * (G_L^2 / (H_L + λ) + G_R^2 / (H_R + λ) - (G_L+G_R)^2 / (H_L+H_R+λ)) - γ
    
    where:
    - G_L, H_L = sum of gradients and hessians in left child
    - G_R, H_R = sum of gradients and hessians in right child
    - λ = L2 regularization parameter
    - γ = minimum loss reduction (regularization)
    
    Parameters
    ----------
    grad_left : float or ndarray
        Sum of gradients in left child.
    hess_left : float or ndarray
        Sum of hessians in left child.
    grad_right : float or ndarray
        Sum of gradients in right child.
    hess_right : float or ndarray
        Sum of hessians in right child.
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction (gamma regularization).
        
    Returns
    -------
    gain : float or ndarray
        Split gain value(s). Higher is better.
        
    Notes
    -----
    This vectorized version is significantly faster than scalar version
    when operating on arrays. For example, when computing gains for all
    split points in a feature at once.
    """
    # Calculate parent statistics
    grad_parent = grad_left + grad_right
    hess_parent = hess_left + hess_right
    
    # Avoid division by zero with small epsilon
    epsilon = 1e-10
    
    # Calculate scores for left, right, and parent
    # Score = G^2 / (H + λ)
    score_left = (grad_left ** 2) / np.maximum(hess_left + reg_lambda, epsilon)
    score_right = (grad_right ** 2) / np.maximum(hess_right + reg_lambda, epsilon)
    score_parent = (grad_parent ** 2) / np.maximum(hess_parent + reg_lambda, epsilon)
    
    # Calculate gain
    gain = 0.5 * (score_left + score_right - score_parent) - gamma
    
    return gain


def calculate_gains_vectorized(
    grad_lefts: np.ndarray,
    hess_lefts: np.ndarray,
    grad_rights: np.ndarray,
    hess_rights: np.ndarray,
    reg_lambda: float = 1.0,
    gamma: float = 0.0
) -> np.ndarray:
    """
    Vectorized calculation of gains for multiple splits.
    
    This is an optimized version that uses pure NumPy operations
    to calculate gains for all split points at once, avoiding Python loops.
    
    Parameters
    ----------
    grad_lefts : ndarray of shape (n_splits,)
        Sum of gradients in left child for each split.
    hess_lefts : ndarray of shape (n_splits,)
        Sum of hessians in left child for each split.
    grad_rights : ndarray of shape (n_splits,)
        Sum of gradients in right child for each split.
    hess_rights : ndarray of shape (n_splits,)
        Sum of hessians in right child for each split.
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction.
        
    Returns
    -------
    gains : ndarray of shape (n_splits,)
        Gain for each split point.
    """
    return calculate_gain(grad_lefts, hess_lefts, grad_rights, hess_rights, reg_lambda, gamma)


class ExactSplitFinder:
    """
    Exact greedy algorithm for finding optimal splits.
    
    This algorithm evaluates all possible split points for each feature
    and selects the one with maximum gain.
    
    Uses vectorized NumPy operations for fast computation.
    
    Parameters
    ----------
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction (gamma regularization).
    min_child_weight : float, default=1.0
        Minimum sum of hessians required in a child node.
    use_vectorization : bool, default=True
        Whether to use vectorized operations (much faster for large datasets).
    use_parallelization : bool, default=False
        Whether to evaluate features in parallel.
    n_jobs : int, default=-1
        Number of parallel jobs for feature evaluation. -1 means use all cores.
    """
    
    def __init__(
        self,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        use_vectorization: bool = True,
        use_parallelization: bool = False,
        n_jobs: int = -1,
    ):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.use_vectorization = use_vectorization
        self.use_parallelization = use_parallelization

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
    
    def find_best_split(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature_indices: Optional[np.ndarray] = None
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split across all features.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
        feature_indices : ndarray, optional
            Indices of features to consider. If None, considers all features.
            
        Returns
        -------
        best_feature : int or None
            Index of best feature for splitting. None if no valid split found.
        best_threshold : float or None
            Best threshold value. None if no valid split found.
        best_gain : float
            Gain of the best split. -inf if no valid split found.
        """
        n_samples, n_features = X.shape
        
        if feature_indices is None:
            feature_indices = np.arange(n_features)
        
        # Sequential path (default / few features)
        if (not self.use_parallelization) or len(feature_indices) <= 1:
            best_feature = None
            best_threshold = None
            best_gain = float('-inf')
            
            # Try each feature
            for feature_idx in feature_indices:
                feature_values = X[:, feature_idx]
                
                # Find best split for this feature
                if self.use_vectorization:
                    threshold, gain = self._find_best_split_for_feature_vectorized(
                        feature_values, grad, hess
                    )
                else:
                    threshold, gain = self._find_best_split_for_feature(
                        feature_values, grad, hess
                    )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        else:
            # Parallel evaluation across features using threads (avoids pickling issues)
            def _evaluate(feature_idx: int):
                feature_values = X[:, feature_idx]
                if self.use_vectorization:
                    return self._find_best_split_for_feature_vectorized(
                        feature_values, grad, hess
                    )
                return self._find_best_split_for_feature(
                    feature_values, grad, hess
                )

            results = Parallel(
                n_jobs=self.n_jobs,
                backend="threading",
            )(delayed(_evaluate)(f_idx) for f_idx in feature_indices)

            best_feature = None
            best_threshold = None
            best_gain = float('-inf')

            for feature_idx, (threshold, gain) in zip(feature_indices, results):
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_gain == float('-inf'):
            return None, None, best_gain
        
        return best_feature, best_threshold, best_gain
    
    def _find_best_split_for_feature(
        self,
        feature_values: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Find best split for a single feature (scalar loop version).
        
        Parameters
        ----------
        feature_values : ndarray of shape (n_samples,)
            Values of the feature for all samples.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
            
        Returns
        -------
        best_threshold : float or None
            Best threshold value. None if no valid split found.
        best_gain : float
            Gain of the best split. -inf if no valid split found.
        """
        # Ignore missing values when searching for splits
        mask = ~np.isnan(feature_values)
        if not np.any(mask):
            return None, float('-inf')

        # Work only on non-missing subset
        feat = feature_values[mask]
        grad_masked = grad[mask]
        hess_masked = hess[mask]

        # If all values are identical, no valid split
        unique_values = np.unique(feat)
        if len(unique_values) < 2:
            return None, float('-inf')
        
        # Sort once and avoid creating multiple large intermediate arrays
        sorted_indices = np.argsort(feat)
        sorted_values = feat[sorted_indices]

        # Cumulative sums using indexed gradients/hessians (no extra sorted_* arrays)
        grad_sorted = grad_masked[sorted_indices]
        hess_sorted = hess_masked[sorted_indices]

        grad_cumsum = np.cumsum(grad_sorted)
        hess_cumsum = np.cumsum(hess_sorted)

        grad_total = grad_cumsum[-1]
        hess_total = hess_cumsum[-1]

        best_threshold = None
        best_gain = float('-inf')

        # Try each possible split point between distinct consecutive values
        for i in range(len(sorted_values) - 1):
            if sorted_values[i] == sorted_values[i + 1]:
                continue

            grad_left = grad_cumsum[i]
            hess_left = hess_cumsum[i]
            grad_right = grad_total - grad_left
            hess_right = hess_total - hess_left

            # Check minimum child weight constraint
            if hess_left < self.min_child_weight or hess_right < self.min_child_weight:
                continue

            # Threshold is midpoint between consecutive values
            threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0

            gain = calculate_gain(
                grad_left,
                hess_left,
                grad_right,
                hess_right,
                self.reg_lambda,
                self.gamma,
            )

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_threshold, best_gain
    
    def _find_best_split_for_feature_vectorized(
        self,
        feature_values: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Find best split for a single feature (VECTORIZED version).
        
        This is a significant optimization using NumPy operations instead
        of Python loops. It computes statistics for all valid split points
        simultaneously using cumulative sums.
        
        Parameters
        ----------
        feature_values : ndarray of shape (n_samples,)
            Values of the feature for all samples.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
            
        Returns
        -------
        best_threshold : float or None
            Best threshold value. None if no valid split found.
        best_gain : float
            Gain of the best split. -inf if no valid split found.
        """
        # Get unique sorted values
        # Ignore missing values when searching for splits
        mask = ~np.isnan(feature_values)
        if not np.any(mask):
            return None, float('-inf')

        sorted_indices = np.argsort(feature_values[mask])
        sorted_values = feature_values[mask][sorted_indices]
        sorted_grad = grad[mask][sorted_indices]
        sorted_hess = hess[mask][sorted_indices]
        
        # Use cumulative sums for fast computation
        # cumsum[i] = sum of first i elements
        cum_grad = np.cumsum(sorted_grad)
        cum_hess = np.cumsum(sorted_hess)
        
        total_grad = cum_grad[-1]
        total_hess = cum_hess[-1]
        
        # Find indices where split should be considered (values change)
        split_points = np.where(sorted_values[:-1] != sorted_values[1:])[0]
        
        if len(split_points) == 0:
            return None, float('-inf')
        
        # Compute statistics for all valid split points
        # At split point i, left contains elements 0..i, right contains elements i+1..n-1
        grad_lefts = cum_grad[split_points]
        hess_lefts = cum_hess[split_points]
        grad_rights = total_grad - grad_lefts
        hess_rights = total_hess - hess_lefts
        
        # Check minimum child weight constraint (vectorized)
        valid_mask = (hess_lefts >= self.min_child_weight) & \
                     (hess_rights >= self.min_child_weight)
        
        if not np.any(valid_mask):
            return None, float('-inf')
        
        # Calculate gains for all valid splits (vectorized)
        gains = calculate_gains_vectorized(
            grad_lefts[valid_mask],
            hess_lefts[valid_mask],
            grad_rights[valid_mask],
            hess_rights[valid_mask],
            self.reg_lambda,
            self.gamma
        )
        
        # Find best split
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == float('-inf'):
            return None, float('-inf')
        
        # Get the best split point indices
        valid_split_points = split_points[valid_mask]
        best_split_point = valid_split_points[best_idx]
        
        # Calculate threshold (midpoint between consecutive values)
        best_threshold = (sorted_values[best_split_point] + sorted_values[best_split_point + 1]) / 2.0
        
        return best_threshold, best_gain
    
    def split_data(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature: int,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Split data based on feature and threshold.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
        feature : int
            Feature index for splitting.
        threshold : float
            Threshold value.
            
        Returns
        -------
        X_left, grad_left, hess_left : ndarray
            Data for left child (feature < threshold, plus any assigned missing).
        X_right, grad_right, hess_right : ndarray
            Data for right child (feature >= threshold, plus remaining missing).
        assign_missing_to_left : bool
            True if NaNs are routed to the left child for this split, False if they
            are routed to the right child.
        """
        # Handle missing values (NaN) in the feature by assigning them to
        # the side (left/right) that yields the higher gain.
        feature_col = X[:, feature]
        missing_mask = np.isnan(feature_col)

        # Non-missing split
        non_missing_mask = ~missing_mask
        mask = feature_col[non_missing_mask] < threshold

        # Stats for non-missing
        grad_left_non = np.sum(grad[non_missing_mask][mask])
        hess_left_non = np.sum(hess[non_missing_mask][mask])
        grad_right_non = np.sum(grad[non_missing_mask][~mask])
        hess_right_non = np.sum(hess[non_missing_mask][~mask])

        # Stats for missing
        grad_missing = np.sum(grad[missing_mask]) if np.any(missing_mask) else 0.0
        hess_missing = np.sum(hess[missing_mask]) if np.any(missing_mask) else 0.0

        # Option 1: assign missing to left
        g_left1 = grad_left_non + grad_missing
        h_left1 = hess_left_non + hess_missing
        g_right1 = grad_right_non
        h_right1 = hess_right_non
        gain_left = calculate_gain(g_left1, h_left1, g_right1, h_right1, reg_lambda=self.reg_lambda, gamma=self.gamma)

        # Option 2: assign missing to right
        g_left2 = grad_left_non
        h_left2 = hess_left_non
        g_right2 = grad_right_non + grad_missing
        h_right2 = hess_right_non + hess_missing
        gain_right = calculate_gain(g_left2, h_left2, g_right2, h_right2, reg_lambda=self.reg_lambda, gamma=self.gamma)

        assign_missing_to_left = gain_left >= gain_right

        # Build final masks including missing values
        final_left_mask = np.zeros_like(feature_col, dtype=bool)
        final_right_mask = np.zeros_like(feature_col, dtype=bool)

        # Fill non-missing
        final_left_mask[non_missing_mask] = mask
        final_right_mask[non_missing_mask] = ~mask

        # Assign missing according to chosen side
        if np.any(missing_mask):
            if assign_missing_to_left:
                final_left_mask[missing_mask] = True
            else:
                final_right_mask[missing_mask] = True

        X_left = X[final_left_mask]
        grad_left = grad[final_left_mask]
        hess_left = hess[final_left_mask]

        X_right = X[final_right_mask]
        grad_right = grad[final_right_mask]
        hess_right = hess[final_right_mask]
        
        return (
            X_left,
            grad_left,
            hess_left,
            X_right,
            grad_right,
            hess_right,
            bool(assign_missing_to_left),
        )


class WeightedQuantileSketch:
    """
    Weighted quantile sketch for approximate histogram construction.
    
    Uses a simplified merge-based algorithm to maintain quantile estimates
    with weighted samples. This allows efficient histogram construction
    with memory O(log n) instead of O(n).
    
    Parameters
    ----------
    max_bins : int, default=256
        Maximum number of bins to keep.
    """
    
    def __init__(self, max_bins: int = 256):
        self.max_bins = max_bins
        self.data = []  # List of (value, weight) tuples
        
    def add(self, values: np.ndarray, weights: np.ndarray):
        """
        Add weighted samples to the sketch.
        
        Parameters
        ----------
        values : ndarray of shape (n,)
            Feature values.
        weights : ndarray of shape (n,)
            Sample weights (typically absolute gradient values).
        """
        # Ignore missing values and append
        mask = ~np.isnan(values)
        for val, weight in zip(values[mask], weights[mask]):
            self.data.append((float(val), float(weight)))
    
    def get_bins(self, n_bins: int) -> np.ndarray:
        """
        Get quantile-based bin boundaries.
        
        Parameters
        ----------
        n_bins : int
            Number of bins to return.
            
        Returns
        -------
        bins : ndarray
            Bin boundaries (cut points).
        """
        if not self.data:
            return np.array([])
        
        # Sort by value
        sorted_data = sorted(self.data, key=lambda x: x[0])
        values = np.array([x[0] for x in sorted_data])
        weights = np.array([x[1] for x in sorted_data])
        
        if len(values) <= n_bins:
            return np.unique(values)
        
        # Cumulative sum of weights for quantile calculation
        cum_weights = np.cumsum(weights)
        total_weight = cum_weights[-1]
        
        # Target weights for quantiles
        target_weights = np.linspace(0, total_weight, n_bins + 1)[1:-1]
        
        # Find values closest to target weights
        bins = []
        for target in target_weights:
            idx = np.searchsorted(cum_weights, target)
            idx = min(idx, len(values) - 1)
            bins.append(values[idx])
        
        # Add min and max values
        bins = np.unique([values[0]] + bins + [values[-1]])
        
        return bins


class ApproximateSplitFinder:
    """
    Approximate greedy algorithm using histogram-based split finding.
    
    This algorithm uses weighted quantile sketches to create histograms
    and evaluates splits on histogram boundaries instead of all unique values.
    Reduces complexity from O(n) to O(bins) per feature.
    
    Much faster for large datasets while maintaining reasonable accuracy.
    
    Parameters
    ----------
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction.
    min_child_weight : float, default=1.0
        Minimum sum of hessians in a child.
    max_bins : int, default=256
        Maximum number of bins for histogram construction.
    use_parallelization : bool, default=True
        Whether to parallelize feature evaluation across cores.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all cores.
    use_parallel_histograms : bool, default=False
        Whether to build histograms in parallel across data chunks.
        Default False for backward compatibility.
    n_jobs_histograms : int, default=-1
        Number of parallel jobs for histogram building. -1 means use all cores.
        Only used if use_parallel_histograms=True.
    """
    
    def __init__(
        self,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        max_bins: int = 256,
        use_parallelization: bool = True,
        n_jobs: int = -1,
        use_parallel_histograms: bool = False,
        n_jobs_histograms: int = -1
    ):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_bins = max_bins
        self.use_parallelization = use_parallelization
        self.use_parallel_histograms = use_parallel_histograms
        self.n_jobs_histograms = n_jobs_histograms
        
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = max(1, min(n_jobs, cpu_count()))
    
    def _build_histograms(
        self,
        feature_values: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Build histogram bins using weighted quantile sketch.
        
        Parameters
        ----------
        feature_values : ndarray
            Feature values.
        weights : ndarray
            Sample weights (absolute gradient values).
            
        Returns
        -------
        bins : ndarray
            Histogram bin boundaries.
        """
        sketch = WeightedQuantileSketch(max_bins=self.max_bins)
        sketch.add(feature_values, weights)
        bins = sketch.get_bins(min(self.max_bins, len(np.unique(feature_values))))
        return bins
    
    def _build_histogram(
        self,
        feature_values: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build gradient/hessian histogram statistics for a single feature.

        Returns
        -------
        tuple (bins, g_hist, h_hist) or None if no valid split is possible.
        """
        # Filter missing values
        mask_non_missing = ~np.isnan(feature_values)
        if not np.any(mask_non_missing):
            return None
        
        f = feature_values[mask_non_missing]
        g = grad[mask_non_missing]
        h = hess[mask_non_missing]
        
        unique_values = np.unique(f)
        if len(unique_values) < 2:
            # Constant feature: no valid split
            return None
        
        # Determine candidate bins
        if len(unique_values) <= self.max_bins:
            bins = unique_values
        else:
            # Build bins using weighted quantile sketch
            weights = np.abs(g) + 1e-10  # Use gradient magnitude as weight
            bins = self._build_histograms(f, weights)
        
        if len(bins) < 2:
            return None
        
        # Build histogram statistics for these bins
        if self.use_parallel_histograms:
            g_hist, h_hist = build_histogram_parallel(
                f, g, h, bins, n_jobs=self.n_jobs_histograms
            )
        else:
            # Map values to bin indices: O(N)
            indices = np.digitize(f, bins)
            minlength = len(bins) + 1
            g_hist = np.bincount(indices, weights=g, minlength=minlength)
            h_hist = np.bincount(indices, weights=h, minlength=minlength)
        
        return bins, g_hist, h_hist
    
    def _find_best_split_from_histogram(
        self,
        bins: np.ndarray,
        g_hist: np.ndarray,
        h_hist: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Find best split threshold given precomputed histogram statistics.

        Parameters
        ----------
        bins : ndarray
            Histogram bin boundaries.
        g_hist : ndarray
            Gradient histogram.
        h_hist : ndarray
            Hessian histogram.
        """
        if len(bins) < 2:
            return None, float('-inf')
        
        # Calculate cumulative stats for fast split evaluation: O(K)
        # g_cum[i] = sum of gradients for all bins <= i
        g_cum = np.cumsum(g_hist)
        h_cum = np.cumsum(h_hist)
        
        g_total = g_cum[-1]
        h_total = h_cum[-1]
        
        best_threshold = None
        best_gain = float('-inf')
        
        # Evaluate splits at each bin boundary: O(K)
        # Threshold bins[i] implies:
        # Left node: x < bins[i]  (Indices 0..i)
        # Right node: x >= bins[i] (Indices i+1..end)
        for i in range(len(bins)):
            threshold = bins[i]
            
            # Left stats
            g_left = g_cum[i] if i < len(g_cum) else g_total
            h_left = h_cum[i] if i < len(h_cum) else h_total
            
            # Right stats
            g_right = g_total - g_left
            h_right = h_total - h_left
            
            # Check constraints
            if h_left < self.min_child_weight or h_right < self.min_child_weight:
                continue
            
            # Calculate gain
            gain = calculate_gain(
                g_left, h_left,
                g_right, h_right,
                self.reg_lambda, self.gamma
            )
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain
    
    def _evaluate_histogram_splits(
        self,
        feature_values: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        bins: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Evaluate splits at histogram bin boundaries.
        
        Parameters
        ----------
        feature_values : ndarray
            Feature values.
        grad : ndarray
            Gradient values.
        hess : ndarray
            Hessian values.
        bins : ndarray
            Histogram bin boundaries.
            
        Returns
        -------
        best_threshold : float or None
            Best threshold from histogram bins.
        best_gain : float
            Gain of best split.
        """
        if len(bins) < 2:
            return None, float('-inf')
        
        # Reuse shared histogram-building logic
        hist = self._build_histogram(feature_values, grad, hess)
        if hist is None:
            return None, float('-inf')
        
        bins_hist, g_hist, h_hist = hist
        # Use bins from histogram (should be identical to input bins for valid cases)
        return self._find_best_split_from_histogram(bins_hist, g_hist, h_hist)
    
    def _find_best_split_for_feature(
        self,
        feature_values: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Find best split for a single feature (approximate method).
        """
        hist = self._build_histogram(feature_values, grad, hess)
        if hist is None:
            return None, float('-inf')
        
        bins, g_hist, h_hist = hist
        return self._find_best_split_from_histogram(bins, g_hist, h_hist)
    
    def find_best_split(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature_indices: Optional[np.ndarray] = None
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find best split across features using histogram method.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
        feature_indices : ndarray, optional
            Indices of features to consider.
            
        Returns
        -------
        best_feature : int or None
            Index of best feature.
        best_threshold : float or None
            Best threshold.
        best_gain : float
            Gain of best split.
        """
        n_samples, n_features = X.shape
        
        if feature_indices is None:
            feature_indices = np.arange(n_features)
        
        if not self.use_parallelization or len(feature_indices) <= 1:
            # Sequential evaluation (build and use histograms per feature)
            best_feature = None
            best_threshold = None
            best_gain = float('-inf')
            
            for feature_idx in feature_indices:
                hist = self._build_histogram(X[:, feature_idx], grad, hess)
                if hist is None:
                    continue
                
                bins, g_hist, h_hist = hist
                threshold, gain = self._find_best_split_from_histogram(bins, g_hist, h_hist)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        else:
            # Parallel precomputation of histograms across features
            from joblib import Parallel, delayed
            
            hist_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._build_histogram)(X[:, feat_idx], grad, hess)
                for feat_idx in feature_indices
            )
            
            best_feature = None
            best_threshold = None
            best_gain = float('-inf')
            
            for feature_idx, hist in zip(feature_indices, hist_results):
                if hist is None:
                    continue
                bins, g_hist, h_hist = hist
                threshold, gain = self._find_best_split_from_histogram(bins, g_hist, h_hist)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_gain == float('-inf'):
            return None, None, best_gain
        
        return best_feature, best_threshold, best_gain
    
    def split_data(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature: int,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Split data based on feature and threshold.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
        feature : int
            Feature index for splitting.
        threshold : float
            Threshold value.
            
        Returns
        -------
        X_left, grad_left, hess_left : ndarray
            Data for left child.
        X_right, grad_right, hess_right : ndarray
            Data for right child.
        assign_missing_to_left : bool
            True if NaNs are routed to the left child for this split, False if they
            are routed to the right child.
        """
        # Handle missing values (NaN) in the feature by assigning them to
        # the side (left/right) that yields the higher gain.
        feature_col = X[:, feature]
        missing_mask = np.isnan(feature_col)

        # Non-missing split
        non_missing_mask = ~missing_mask
        mask = feature_col[non_missing_mask] < threshold

        # Stats for non-missing
        grad_left_non = np.sum(grad[non_missing_mask][mask])
        hess_left_non = np.sum(hess[non_missing_mask][mask])
        grad_right_non = np.sum(grad[non_missing_mask][~mask])
        hess_right_non = np.sum(hess[non_missing_mask][~mask])

        # Stats for missing
        grad_missing = np.sum(grad[missing_mask]) if np.any(missing_mask) else 0.0
        hess_missing = np.sum(hess[missing_mask]) if np.any(missing_mask) else 0.0

        # Option 1: assign missing to left
        g_left1 = grad_left_non + grad_missing
        h_left1 = hess_left_non + hess_missing
        g_right1 = grad_right_non
        h_right1 = hess_right_non
        gain_left = calculate_gain(g_left1, h_left1, g_right1, h_right1, reg_lambda=self.reg_lambda, gamma=self.gamma)

        # Option 2: assign missing to right
        g_left2 = grad_left_non
        h_left2 = hess_left_non
        g_right2 = grad_right_non + grad_missing
        h_right2 = hess_right_non + hess_missing
        gain_right = calculate_gain(g_left2, h_left2, g_right2, h_right2, reg_lambda=self.reg_lambda, gamma=self.gamma)

        assign_missing_to_left = gain_left >= gain_right

        # Build final masks including missing values
        final_left_mask = np.zeros_like(feature_col, dtype=bool)
        final_right_mask = np.zeros_like(feature_col, dtype=bool)

        # Fill non-missing
        final_left_mask[non_missing_mask] = mask
        final_right_mask[non_missing_mask] = ~mask

        # Assign missing according to chosen side
        if np.any(missing_mask):
            if assign_missing_to_left:
                final_left_mask[missing_mask] = True
            else:
                final_right_mask[missing_mask] = True

        X_left = X[final_left_mask]
        grad_left = grad[final_left_mask]
        hess_left = hess[final_left_mask]

        X_right = X[final_right_mask]
        grad_right = grad[final_right_mask]
        hess_right = hess[final_right_mask]
        
        return (
            X_left,
            grad_left,
            hess_left,
            X_right,
            grad_right,
            hess_right,
            bool(assign_missing_to_left),
        )


class HybridSplitFinder:
    """
    Adaptive split finder that chooses between exact and approximate methods.
    
    Uses exact greedy algorithm for small datasets where speed is not critical,
    and switches to approximate histogram-based method for larger datasets
    where efficiency is more important.
    
    Parameters
    ----------
    exact_threshold : int, default=10000
        Number of samples above which to switch to approximate method.
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction.
    min_child_weight : float, default=1.0
        Minimum sum of hessians in a child.
    max_bins : int, default=256
        Maximum bins for approximate method.
    use_parallelization : bool, default=True
        Whether to use parallelization in approximate method.
    n_jobs : int, default=-1
        Number of parallel jobs.
    use_parallel_histograms : bool, default=False
        Whether to build histograms in parallel across data chunks.
        Default False for backward compatibility.
    n_jobs_histograms : int, default=-1
        Number of parallel jobs for histogram building. -1 means use all cores.
        Only used if use_parallel_histograms=True.
    """
    
    def __init__(
        self,
        exact_threshold: int = 10000,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        max_bins: int = 256,
        use_parallelization: bool = True,
        n_jobs: int = -1,
        use_parallel_histograms: bool = False,
        n_jobs_histograms: int = -1
    ):
        self.exact_threshold = exact_threshold
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_bins = max_bins
        self.use_parallelization = use_parallelization
        self.n_jobs = n_jobs
        self.use_parallel_histograms = use_parallel_histograms
        self.n_jobs_histograms = n_jobs_histograms
        
        # Initialize both finders
        self.exact_finder = ExactSplitFinder(
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            use_vectorization=True
        )
        self.approx_finder = ApproximateSplitFinder(
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_bins=max_bins,
            use_parallelization=use_parallelization,
            n_jobs=n_jobs,
            use_parallel_histograms=use_parallel_histograms,
            n_jobs_histograms=n_jobs_histograms
        )
    
    def find_best_split(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature_indices: Optional[np.ndarray] = None
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find best split using adaptive algorithm selection.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
        feature_indices : ndarray, optional
            Indices of features to consider.
            
        Returns
        -------
        best_feature : int or None
            Index of best feature.
        best_threshold : float or None
            Best threshold.
        best_gain : float
            Gain of best split.
        """
        n_samples = X.shape[0]
        
        # Choose algorithm based on dataset size
        if n_samples <= self.exact_threshold:
            use_exact = True
        else:
            use_exact = False
        
        if use_exact:
            return self.exact_finder.find_best_split(X, grad, hess, feature_indices)
        else:
            return self.approx_finder.find_best_split(X, grad, hess, feature_indices)
    
    def split_data(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature: int,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Split data based on feature and threshold.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
        feature : int
            Feature index for splitting.
        threshold : float
            Threshold value.
            
        Returns
        -------
        X_left, grad_left, hess_left : ndarray
            Data for left child.
        X_right, grad_right, hess_right : ndarray
            Data for right child.
        assign_missing_to_left : bool
            True if NaNs are routed to the left child for this split, False if they
            are routed to the right child.
        """
        # Make split_data sparsity-aware (handle NaNs)
        feature_col = X[:, feature]
        missing_mask = np.isnan(feature_col)

        non_missing_mask = ~missing_mask
        mask = feature_col[non_missing_mask] < threshold

        # Stats for non-missing
        grad_left_non = np.sum(grad[non_missing_mask][mask])
        hess_left_non = np.sum(hess[non_missing_mask][mask])
        grad_right_non = np.sum(grad[non_missing_mask][~mask])
        hess_right_non = np.sum(hess[non_missing_mask][~mask])

        # Stats for missing
        grad_missing = np.sum(grad[missing_mask]) if np.any(missing_mask) else 0.0
        hess_missing = np.sum(hess[missing_mask]) if np.any(missing_mask) else 0.0

        # Option 1: assign missing to left
        g_left1 = grad_left_non + grad_missing
        h_left1 = hess_left_non + hess_missing
        g_right1 = grad_right_non
        h_right1 = hess_right_non
        gain_left = calculate_gain(g_left1, h_left1, g_right1, h_right1, reg_lambda=self.reg_lambda, gamma=self.gamma)

        # Option 2: assign missing to right
        g_left2 = grad_left_non
        h_left2 = hess_left_non
        g_right2 = grad_right_non + grad_missing
        h_right2 = hess_right_non + hess_missing
        gain_right = calculate_gain(g_left2, h_left2, g_right2, h_right2, reg_lambda=self.reg_lambda, gamma=self.gamma)

        assign_missing_to_left = gain_left >= gain_right

        # Build final masks
        final_left_mask = np.zeros_like(feature_col, dtype=bool)
        final_right_mask = np.zeros_like(feature_col, dtype=bool)
        final_left_mask[non_missing_mask] = mask
        final_right_mask[non_missing_mask] = ~mask
        if np.any(missing_mask):
            if assign_missing_to_left:
                final_left_mask[missing_mask] = True
            else:
                final_right_mask[missing_mask] = True

        X_left = X[final_left_mask]
        grad_left = grad[final_left_mask]
        hess_left = hess[final_left_mask]

        X_right = X[final_right_mask]
        grad_right = grad[final_right_mask]
        hess_right = hess[final_right_mask]
        
        return (
            X_left,
            grad_left,
            hess_left,
            X_right,
            grad_right,
            hess_right,
            bool(assign_missing_to_left),
        )
        