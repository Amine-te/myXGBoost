"""Decision tree implementation."""

import numpy as np
from typing import Optional
from myXGBoost.base.tree import TreeNode
from myXGBoost.trees.split_finder import ExactSplitFinder, HybridSplitFinder
from myXGBoost.trees.leaf import calculate_leaf_weight


class DecisionTree:
    """
    Decision tree for gradient boosting.
    
    This tree stores gradient and hessian statistics at each node
    and uses them to find optimal splits and calculate leaf values.
    
    Supports both exact and approximate (histogram-based) split finding
    with automatic algorithm selection based on dataset size.
    
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
    use_hybrid_split_finder : bool, default=True
        Whether to use hybrid (adaptive) split finder that chooses
        between exact and approximate methods based on dataset size.
    exact_threshold : int, default=10000
        Threshold for switching from exact to approximate algorithm.
        Used only if use_hybrid_split_finder=True.
    max_bins : int, default=256
        Maximum number of bins for histogram construction
        in approximate method. Used only if use_hybrid_split_finder=True.
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        use_hybrid_split_finder: bool = True,
        exact_threshold: int = 10000,
        max_bins: int = 256
    ):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.use_hybrid_split_finder = use_hybrid_split_finder
        self.exact_threshold = exact_threshold
        self.max_bins = max_bins
        
        self.root = None
        
        # Initialize split finder based on configuration
        if use_hybrid_split_finder:
            self.split_finder = HybridSplitFinder(
                exact_threshold=exact_threshold,
                reg_lambda=reg_lambda,
                gamma=gamma,
                min_child_weight=min_child_weight,
                max_bins=max_bins,
                use_parallelization=True,  # Enable parallelization
                n_jobs=-1  # Use all cores
            )
        else:
            self.split_finder = ExactSplitFinder(
                reg_lambda=reg_lambda,
                gamma=gamma,
                min_child_weight=min_child_weight,
                use_vectorization=True
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
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.float64)
        
        # Start recursion with all indices
        self._predict_vectorized(self.root, X, np.arange(n_samples), predictions)
        
        return predictions
    
    def _predict_vectorized(
        self,
        node: TreeNode,
        X: np.ndarray,
        indices: np.ndarray,
        predictions: np.ndarray
    ):
        """
        Helper for vectorized prediction.
        
        Parameters
        ----------
        node : TreeNode
            Current node.
        X : ndarray
            Full feature matrix.
        indices : ndarray
            Indices of samples currently in this node.
        predictions : ndarray
            Array to store predictions (modified in-place).
        """
        if len(indices) == 0:
            return

        if node.is_leaf:
            val = node.leaf_value if node.leaf_value is not None else 0.0
            predictions[indices] = val
            return
        
        # Get feature values for relevant samples
        # Use slicing for speed if convenient, but array indexing is needed here
        values = X[indices, node.split_feature]
        
        # Determine split
        left_mask = values < node.split_threshold
        
        # Recurse left
        # node indices corresponding to left mask
        left_indices = indices[left_mask]
        self._predict_vectorized(node.left_child, X, left_indices, predictions)
        
        # Recurse right
        right_indices = indices[~left_mask]
        self._predict_vectorized(node.right_child, X, right_indices, predictions)
    
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
