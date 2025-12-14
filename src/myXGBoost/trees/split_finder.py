"""Split finding algorithms (exact, approximate, histogram-based)."""

import numpy as np
from typing import Tuple, Optional, Union


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
    """
    
    def __init__(
        self,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        use_vectorization: bool = True
    ):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.use_vectorization = use_vectorization
    
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
        # Get unique sorted values
        unique_values = np.unique(feature_values)
        
        if len(unique_values) < 2:
            return None, float('-inf')
        
        # Try splits between consecutive unique values
        best_threshold = None
        best_gain = float('-inf')
        
        # Sort indices by feature values
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_grad = grad[sorted_indices]
        sorted_hess = hess[sorted_indices]
        
        # Initialize left and right statistics
        grad_left = 0.0
        hess_left = 0.0
        grad_right = np.sum(sorted_grad)
        hess_right = np.sum(sorted_hess)
        
        # Try each possible split point
        for i in range(len(sorted_values) - 1):
            # Skip if values are the same
            if sorted_values[i] == sorted_values[i + 1]:
                continue
            
            # Move sample from right to left
            grad_left += sorted_grad[i]
            hess_left += sorted_hess[i]
            grad_right -= sorted_grad[i]
            hess_right -= sorted_hess[i]
            
            # Check minimum child weight constraint
            if hess_left < self.min_child_weight or hess_right < self.min_child_weight:
                continue
            
            # Calculate threshold (midpoint between consecutive values)
            threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0
            
            # Calculate gain
            gain = calculate_gain(
                grad_left, hess_left,
                grad_right, hess_right,
                self.reg_lambda, self.gamma
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
        unique_values = np.unique(feature_values)
        
        if len(unique_values) < 2:
            return None, float('-inf')
        
        # Sort indices by feature values
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_grad = grad[sorted_indices]
        sorted_hess = hess[sorted_indices]
        
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            Data for left child (feature < threshold).
        X_right, grad_right, hess_right : ndarray
            Data for right child (feature >= threshold).
        """
        mask = X[:, feature] < threshold
        
        X_left = X[mask]
        grad_left = grad[mask]
        hess_left = hess[mask]
        
        X_right = X[~mask]
        grad_right = grad[~mask]
        hess_right = hess[~mask]
        
        return X_left, grad_left, hess_left, X_right, grad_right, hess_right
