"""Split finding algorithms (exact, approximate, histogram-based)."""

import numpy as np
from typing import Tuple, Optional


def calculate_gain(
    grad_left: float,
    hess_left: float,
    grad_right: float,
    hess_right: float,
    reg_lambda: float = 1.0,
    gamma: float = 0.0
) -> float:
    """
    Calculate split gain using XGBoost formula.
    
    Formula:
    Gain = 0.5 * (G_L^2 / (H_L + λ) + G_R^2 / (H_R + λ) - (G_L+G_R)^2 / (H_L+H_R+λ)) - γ
    
    where:
    - G_L, H_L = sum of gradients and hessians in left child
    - G_R, H_R = sum of gradients and hessians in right child
    - λ = L2 regularization parameter
    - γ = minimum loss reduction (regularization)
    
    Parameters
    ----------
    grad_left : float
        Sum of gradients in left child.
    hess_left : float
        Sum of hessians in left child.
    grad_right : float
        Sum of gradients in right child.
    hess_right : float
        Sum of hessians in right child.
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction (gamma regularization).
        
    Returns
    -------
    gain : float
        Split gain value.
        
    Notes
    -----
    Higher gain means better split. The gain represents the reduction
    in loss achieved by splitting at this point.
    """
    # Calculate parent statistics
    grad_parent = grad_left + grad_right
    hess_parent = hess_left + hess_right
    
    # Calculate scores for left, right, and parent
    # Score = G^2 / (H + λ)
    score_left = (grad_left ** 2) / (hess_left + reg_lambda) if (hess_left + reg_lambda) > 1e-10 else 0.0
    score_right = (grad_right ** 2) / (hess_right + reg_lambda) if (hess_right + reg_lambda) > 1e-10 else 0.0
    score_parent = (grad_parent ** 2) / (hess_parent + reg_lambda) if (hess_parent + reg_lambda) > 1e-10 else 0.0
    
    # Calculate gain
    gain = 0.5 * (score_left + score_right - score_parent) - gamma
    
    return gain


class ExactSplitFinder:
    """
    Exact greedy algorithm for finding optimal splits.
    
    This algorithm evaluates all possible split points for each feature
    and selects the one with maximum gain.
    
    Parameters
    ----------
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction (gamma regularization).
    min_child_weight : float, default=1.0
        Minimum sum of hessians required in a child node.
    """
    
    def __init__(
        self,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0
    ):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
    
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
        Find best split for a single feature.
        
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
