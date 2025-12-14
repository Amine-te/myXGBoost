"""Decision tree implementation."""

import numpy as np
from typing import Optional
from myXGBoost.base.tree import TreeNode
from myXGBoost.trees.split_finder import ExactSplitFinder
from myXGBoost.trees.leaf import calculate_leaf_weight


class DecisionTree:
    """
    Decision tree for gradient boosting.
    
    This tree stores gradient and hessian statistics at each node
    and uses them to find optimal splits and calculate leaf values.
    
    Parameters
    ----------
    max_depth : int, default=6
        Maximum depth of the tree.
    min_child_weight : float, default=1.0
        Minimum sum of hessians required in a child node.
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    gamma : float, default=0.0
        Minimum loss reduction (gamma regularization).
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0
    ):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        
        self.root = None
        self.split_finder = ExactSplitFinder(
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight
        )
    
    def fit(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature_indices: Optional[np.ndarray] = None
    ):
        """
        Build the decision tree.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        grad : ndarray of shape (n_samples,)
            Gradient values.
        hess : ndarray of shape (n_samples,)
            Hessian values.
        feature_indices : ndarray, optional
            Indices of features to consider for splitting.
            If None, considers all features.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        grad = np.asarray(grad, dtype=np.float64)
        hess = np.asarray(hess, dtype=np.float64)
        
        # Validate inputs
        if X.shape[0] != len(grad) or X.shape[0] != len(hess):
            raise ValueError(
                "X, grad, and hess must have the same number of samples."
            )
        
        # Build tree recursively
        self.root = self._build_node(
            X, grad, hess, feature_indices, depth=0
        )
        
        return self
    
    def _build_node(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        hess: np.ndarray,
        feature_indices: Optional[np.ndarray],
        depth: int
    ) -> TreeNode:
        """
        Recursively build a tree node.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for current node.
        grad : ndarray of shape (n_samples,)
            Gradient values for current node.
        hess : ndarray of shape (n_samples,)
            Hessian values for current node.
        feature_indices : ndarray, optional
            Feature indices to consider.
        depth : int
            Current depth in the tree.
            
        Returns
        -------
        node : TreeNode
            Built tree node (leaf or internal).
        """
        # Create node and update statistics
        node = TreeNode()
        node.grad_sum = np.sum(grad)
        node.hess_sum = np.sum(hess)
        
        # Check stopping criteria
        if depth >= self.max_depth:
            # Create leaf node
            weight = calculate_leaf_weight(
                node.grad_sum, node.hess_sum, self.reg_lambda
            )
            node.set_leaf_value(weight)
            return node
        
        if len(X) == 0:
            # Empty node - create leaf
            node.set_leaf_value(0.0)
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self.split_finder.find_best_split(
            X, grad, hess, feature_indices
        )
        
        # Check if split is valid and beneficial
        if best_feature is None or best_gain <= 0:
            # Create leaf node
            weight = calculate_leaf_weight(
                node.grad_sum, node.hess_sum, self.reg_lambda
            )
            node.set_leaf_value(weight)
            return node
        
        # Split the data
        X_left, grad_left, hess_left, X_right, grad_right, hess_right = \
            self.split_finder.split_data(X, grad, hess, best_feature, best_threshold)
        
        # Check minimum child weight
        if (np.sum(hess_left) < self.min_child_weight or
            np.sum(hess_right) < self.min_child_weight):
            # Create leaf node
            weight = calculate_leaf_weight(
                node.grad_sum, node.hess_sum, self.reg_lambda
            )
            node.set_leaf_value(weight)
            return node
        
        # Set split information
        node.set_split(best_feature, best_threshold)
        
        # Recursively build children
        left_child = self._build_node(
            X_left, grad_left, hess_left, feature_indices, depth + 1
        )
        right_child = self._build_node(
            X_right, grad_right, hess_right, feature_indices, depth + 1
        )
        
        node.set_children(left_child, right_child)
        
        return node
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for input samples.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")
        
        X = np.asarray(X)
        predictions = np.array([
            self.root.predict(x) for x in X
        ])
        
        return predictions
    
    def get_depth(self) -> int:
        """
        Get the depth of the tree.
        
        Returns
        -------
        depth : int
            Maximum depth of the tree.
        """
        if self.root is None:
            return 0
        return self.root.get_depth()
    
    def get_num_leaves(self) -> int:
        """
        Get the number of leaf nodes in the tree.
        
        Returns
        -------
        num_leaves : int
            Number of leaf nodes.
        """
        if self.root is None:
            return 0
        return self.root.get_num_leaves()
