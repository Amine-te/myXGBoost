"""Base tree structure and node classes."""

import numpy as np
from typing import Optional, Tuple


class TreeNode:
    """
    Base tree node structure.
    
    Each node stores:
    - Sum of gradients (G)
    - Sum of hessians (H)
    - Split information (if internal node)
    - Left and right children (if internal node)
    
    Attributes
    ----------
    grad_sum : float
        Sum of gradients in this node.
    hess_sum : float
        Sum of hessians in this node.
    is_leaf : bool
        Whether this node is a leaf.
    split_feature : int, optional
        Feature index used for splitting (None for leaf nodes).
    split_threshold : float, optional
        Threshold value for splitting (None for leaf nodes).
    left_child : TreeNode, optional
        Left child node (None for leaf nodes).
    right_child : TreeNode, optional
        Right child node (None for leaf nodes).
    leaf_value : float, optional
        Leaf value/weight (only for leaf nodes).
    """
    
    def __init__(self):
        """Initialize an empty tree node."""
        self.grad_sum = 0.0
        self.hess_sum = 0.0
        self.is_leaf = True
        
        # Split information (for internal nodes)
        self.split_feature = None
        self.split_threshold = None
        self.gain = 0.0  # Gain achieved by the split
        
        # Children (for internal nodes)
        self.left_child = None
        self.right_child = None
        
        # Leaf value (for leaf nodes)
        self.leaf_value = None
    
    def update_stats(self, grad: float, hess: float):
        """
        Update gradient and hessian sums.
        
        Parameters
        ----------
        grad : float
            Gradient value to add.
        hess : float
            Hessian value to add.
        """
        self.grad_sum += grad
        self.hess_sum += hess
    
    def set_split(self, feature: int, threshold: float, gain: float = 0.0):
        """
        Set split information and mark as internal node.
        
        Parameters
        ----------
        feature : int
            Feature index for splitting.
        threshold : float
            Threshold value for splitting.
        gain : float, default=0.0
            Gain achieved by the split.
        """
        self.split_feature = feature
        self.split_threshold = threshold
        self.gain = gain
        self.is_leaf = False
    
    def set_children(self, left: 'TreeNode', right: 'TreeNode'):
        """
        Set left and right child nodes.
        
        Parameters
        ----------
        left : TreeNode
            Left child node.
        right : TreeNode
            Right child node.
        """
        self.left_child = left
        self.right_child = right
        self.is_leaf = False
    
    def set_leaf_value(self, value: float):
        """
        Set leaf value and mark as leaf node.
        
        Parameters
        ----------
        value : float
            Leaf value/weight.
        """
        self.leaf_value = value
        self.is_leaf = True
    
    def predict(self, x: np.ndarray) -> float:
        """
        Predict value for a single sample.
        
        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input sample.
            
        Returns
        -------
        value : float
            Predicted value.
        """
        if self.is_leaf:
            return self.leaf_value if self.leaf_value is not None else 0.0
        
        # Navigate to appropriate child
        if x[self.split_feature] < self.split_threshold:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)
    
    def get_depth(self) -> int:
        """
        Get the depth of the subtree rooted at this node.
        
        Returns
        -------
        depth : int
            Maximum depth of the subtree.
        """
        if self.is_leaf:
            return 0
        
        left_depth = self.left_child.get_depth() if self.left_child else 0
        right_depth = self.right_child.get_depth() if self.right_child else 0
        
        return 1 + max(left_depth, right_depth)
    
    def get_num_leaves(self) -> int:
        """
        Get the number of leaf nodes in the subtree.
        
        Returns
        -------
        num_leaves : int
            Number of leaf nodes.
        """
        if self.is_leaf:
            return 1
        
        left_leaves = self.left_child.get_num_leaves() if self.left_child else 0
        right_leaves = self.right_child.get_num_leaves() if self.right_child else 0
        
        return left_leaves + right_leaves
