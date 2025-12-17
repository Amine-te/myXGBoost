"""Leaf node implementation and leaf value calculation."""

import numpy as np


def calculate_leaf_weight(grad_sum: float, hess_sum: float, reg_lambda: float = 1.0) -> float:
    """
    Calculate optimal leaf weight using XGBoost formula.
    
    Formula: w = -G / (H + λ)
    
    where:
    - G = sum of gradients
    - H = sum of hessians
    - λ = L2 regularization parameter
    
    Parameters
    ----------
    grad_sum : float
        Sum of gradients in the leaf.
    hess_sum : float
        Sum of hessians in the leaf.
    reg_lambda : float, default=1.0
        L2 regularization parameter (lambda).
        
    Returns
    -------
    weight : float
        Optimal leaf weight.
        
    Notes
    -----
    This formula minimizes the loss function for the leaf node.
    The negative sign comes from the fact that we're minimizing
    the loss (gradient points in the direction of increasing loss).
    """
    # Avoid division by zero
    denominator = hess_sum + reg_lambda
    if abs(denominator) < 1e-10:
        return 0.0
    
    return -grad_sum / denominator


def calculate_leaf_weights(
    grad_sums: np.ndarray,
    hess_sums: np.ndarray,
    reg_lambda: float = 1.0
) -> np.ndarray:
    """
    Calculate optimal leaf weights for multiple leaves.
    
    Parameters
    ----------
    grad_sums : ndarray of shape (n_leaves,)
        Sum of gradients for each leaf.
    hess_sums : ndarray of shape (n_leaves,)
        Sum of hessians for each leaf.
    reg_lambda : float, default=1.0
        L2 regularization parameter.
        
    Returns
    -------
    weights : ndarray of shape (n_leaves,)
        Optimal leaf weights.
    """
    grad_sums = np.asarray(grad_sums, dtype=np.float64)
    hess_sums = np.asarray(hess_sums, dtype=np.float64)
    
    denominator = hess_sums + reg_lambda
    # Avoid division by zero
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    
    return -grad_sums / denominator

