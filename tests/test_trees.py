"""Tests for tree building and structure."""

import pytest
import numpy as np
from myXGBoost.base.tree import TreeNode
from myXGBoost.trees.decision_tree import DecisionTree
from myXGBoost.trees.split_finder import ExactSplitFinder, calculate_gain
from myXGBoost.trees.leaf import calculate_leaf_weight, calculate_leaf_weights


class TestTreeNode:
    """Tests for TreeNode class."""
    
    def test_node_initialization(self):
        """Test node initialization."""
        node = TreeNode()
        
        assert node.grad_sum == 0.0
        assert node.hess_sum == 0.0
        assert node.is_leaf is True
        assert node.split_feature is None
        assert node.split_threshold is None
        assert node.left_child is None
        assert node.right_child is None
        assert node.leaf_value is None
    
    def test_update_stats(self):
        """Test updating gradient and hessian sums."""
        node = TreeNode()
        
        node.update_stats(1.0, 2.0)
        assert node.grad_sum == 1.0
        assert node.hess_sum == 2.0
        
        node.update_stats(0.5, 1.5)
        assert node.grad_sum == 1.5
        assert node.hess_sum == 3.5
    
    def test_set_split(self):
        """Test setting split information."""
        node = TreeNode()
        node.set_split(feature=0, threshold=5.0)
        
        assert node.is_leaf is False
        assert node.split_feature == 0
        assert node.split_threshold == 5.0
    
    def test_set_children(self):
        """Test setting child nodes."""
        node = TreeNode()
        left = TreeNode()
        right = TreeNode()
        
        node.set_children(left, right)
        
        assert node.is_leaf is False
        assert node.left_child is left
        assert node.right_child is right
    
    def test_set_leaf_value(self):
        """Test setting leaf value."""
        node = TreeNode()
        node.set_leaf_value(0.5)
        
        assert node.is_leaf is True
        assert node.leaf_value == 0.5
    
    def test_predict_leaf(self):
        """Test prediction from leaf node."""
        node = TreeNode()
        node.set_leaf_value(0.5)
        
        x = np.array([1.0, 2.0, 3.0])
        prediction = node.predict(x)
        
        assert prediction == 0.5
    
    def test_predict_internal_node(self):
        """Test prediction from internal node."""
        node = TreeNode()
        node.set_split(feature=0, threshold=5.0)
        
        left = TreeNode()
        left.set_leaf_value(1.0)
        right = TreeNode()
        right.set_leaf_value(2.0)
        
        node.set_children(left, right)
        
        # Test left path
        x_left = np.array([3.0, 2.0, 1.0])
        assert node.predict(x_left) == 1.0
        
        # Test right path
        x_right = np.array([7.0, 2.0, 1.0])
        assert node.predict(x_right) == 2.0
    
    def test_get_depth(self):
        """Test getting tree depth."""
        # Leaf node
        leaf = TreeNode()
        leaf.set_leaf_value(1.0)
        assert leaf.get_depth() == 0
        
        # Simple tree
        root = TreeNode()
        root.set_split(feature=0, threshold=5.0)
        left = TreeNode()
        left.set_leaf_value(1.0)
        right = TreeNode()
        right.set_leaf_value(2.0)
        root.set_children(left, right)
        
        assert root.get_depth() == 1
    
    def test_get_num_leaves(self):
        """Test getting number of leaves."""
        # Single leaf
        leaf = TreeNode()
        leaf.set_leaf_value(1.0)
        assert leaf.get_num_leaves() == 1
        
        # Tree with 2 leaves
        root = TreeNode()
        root.set_split(feature=0, threshold=5.0)
        left = TreeNode()
        left.set_leaf_value(1.0)
        right = TreeNode()
        right.set_leaf_value(2.0)
        root.set_children(left, right)
        
        assert root.get_num_leaves() == 2


class TestLeafWeight:
    """Tests for leaf weight calculation."""
    
    def test_calculate_leaf_weight(self):
        """Test leaf weight calculation formula."""
        # w = -G / (H + λ)
        weight = calculate_leaf_weight(grad_sum=-2.0, hess_sum=4.0, reg_lambda=1.0)
        
        expected = -(-2.0) / (4.0 + 1.0)  # = 2.0 / 5.0 = 0.4
        assert abs(weight - expected) < 1e-10
    
    def test_calculate_leaf_weight_zero_hess(self):
        """Test leaf weight with zero hessian."""
        # When hess_sum=0.0 and reg_lambda=1.0, denominator is 1.0, so weight = -1.0/1.0 = -1.0
        weight = calculate_leaf_weight(grad_sum=1.0, hess_sum=0.0, reg_lambda=1.0)
        assert weight == -1.0
        
        # Test actual division by zero case (both hess_sum and reg_lambda are zero or very small)
        weight_zero = calculate_leaf_weight(grad_sum=1.0, hess_sum=0.0, reg_lambda=0.0)
        # Should handle division by zero gracefully
        assert weight_zero == 0.0
    
    def test_calculate_leaf_weights_array(self):
        """Test calculating weights for multiple leaves."""
        grad_sums = np.array([-2.0, -4.0, -1.0])
        hess_sums = np.array([4.0, 8.0, 2.0])
        
        weights = calculate_leaf_weights(grad_sums, hess_sums, reg_lambda=1.0)
        
        assert len(weights) == 3
        assert weights[0] == 2.0 / 5.0
        assert weights[1] == 4.0 / 9.0
        assert weights[2] == 1.0 / 3.0


class TestGainCalculation:
    """Tests for gain calculation."""
    
    def test_calculate_gain(self):
        """Test gain calculation formula."""
        # Gain = 0.5 * (G_L^2 / (H_L + λ) + G_R^2 / (H_R + λ) - (G_L+G_R)^2 / (H_L+H_R+λ)) - γ
        
        grad_left, hess_left = 2.0, 4.0
        grad_right, hess_right = 3.0, 6.0
        reg_lambda = 1.0
        gamma = 0.0
        
        gain = calculate_gain(
            grad_left, hess_left,
            grad_right, hess_right,
            reg_lambda, gamma
        )
        
        # Manual calculation
        score_left = (2.0 ** 2) / (4.0 + 1.0)  # 4/5 = 0.8
        score_right = (3.0 ** 2) / (6.0 + 1.0)  # 9/7 ≈ 1.286
        score_parent = (5.0 ** 2) / (10.0 + 1.0)  # 25/11 ≈ 2.273
        expected_gain = 0.5 * (score_left + score_right - score_parent) - 0.0
        
        assert abs(gain - expected_gain) < 1e-10
    
    def test_calculate_gain_with_gamma(self):
        """Test gain calculation with gamma regularization."""
        gain_no_gamma = calculate_gain(2.0, 4.0, 3.0, 6.0, 1.0, 0.0)
        gain_with_gamma = calculate_gain(2.0, 4.0, 3.0, 6.0, 1.0, 0.5)
        
        assert gain_with_gamma == gain_no_gamma - 0.5
    
    def test_calculate_gain_negative(self):
        """Test that gain can be negative."""
        # Small gradients, large hessians -> negative gain
        gain = calculate_gain(0.1, 10.0, 0.1, 10.0, 1.0, 0.0)
        
        # Should be negative or very small
        assert gain < 0.1


class TestExactSplitFinder:
    """Tests for exact split finder."""
    
    def test_find_best_split_simple(self):
        """Test finding best split on simple data."""
        split_finder = ExactSplitFinder(reg_lambda=1.0, gamma=0.0, min_child_weight=0.0)
        
        # Simple 2D data where feature 0 perfectly separates classes
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([1.0, 1.0, -1.0, -1.0])  # Positive for first 2, negative for last 2
        hess = np.ones(4)
        
        best_feature, best_threshold, best_gain = split_finder.find_best_split(X, grad, hess)
        
        assert best_feature == 0
        assert 2.0 < best_threshold < 3.0  # Should split between 2 and 3
        assert best_gain > 0
    
    def test_find_best_split_no_valid_split(self):
        """Test when no valid split exists."""
        split_finder = ExactSplitFinder(reg_lambda=1.0, gamma=1000.0, min_child_weight=0.0)
        
        X = np.array([[1.0], [2.0], [3.0]])
        grad = np.array([1.0, 1.0, 1.0])
        hess = np.ones(3)
        
        best_feature, best_threshold, best_gain = split_finder.find_best_split(X, grad, hess)
        
        # With very high gamma, no split should be beneficial
        assert best_feature is None or best_gain <= 0
    
    def test_split_data(self):
        """Test splitting data based on feature and threshold."""
        split_finder = ExactSplitFinder()
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([1.0, 2.0, 3.0, 4.0])
        hess = np.array([1.0, 1.0, 1.0, 1.0])
        
        X_left, grad_left, hess_left, X_right, grad_right, hess_right, assign_missing_to_left = \
            split_finder.split_data(X, grad, hess, feature=0, threshold=2.5)
        
        # Left: values < 2.5
        assert len(X_left) == 2
        assert X_left[0, 0] == 1.0
        assert X_left[1, 0] == 2.0
        
        # Right: values >= 2.5
        assert len(X_right) == 2
        assert X_right[0, 0] == 3.0
        assert X_right[1, 0] == 4.0
        
        # Check gradients and hessians
        assert np.sum(grad_left) == 3.0
        assert np.sum(grad_right) == 7.0


class TestDecisionTree:
    """Tests for DecisionTree class."""
    
    def test_tree_initialization(self):
        """Test tree initialization."""
        tree = DecisionTree(max_depth=5, min_child_weight=1.0, reg_lambda=1.0, gamma=0.0)
        
        assert tree.max_depth == 5
        assert tree.min_child_weight == 1.0
        assert tree.reg_lambda == 1.0
        assert tree.gamma == 0.0
        assert tree.root is None
    
    def test_fit_simple_tree(self):
        """Test fitting a simple tree."""
        tree = DecisionTree(max_depth=3, min_child_weight=0.0, reg_lambda=1.0, gamma=0.0)
        
        # Simple separable data
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([1.0, 1.0, -1.0, -1.0])
        hess = np.ones(4)
        
        tree.fit(X, grad, hess)
        
        assert tree.root is not None
        assert tree.get_depth() > 0
    
    def test_predict(self):
        """Test prediction."""
        tree = DecisionTree(max_depth=3, min_child_weight=0.0, reg_lambda=1.0, gamma=0.0)
        
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        grad = np.array([1.0, 1.0, -1.0, -1.0])
        hess = np.ones(4)
        
        tree.fit(X, grad, hess)
        
        predictions = tree.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        tree = DecisionTree()
        
        X = np.array([[1.0], [2.0]])
        
        with pytest.raises(ValueError, match="not been fitted"):
            tree.predict(X)
    
    def test_max_depth_constraint(self):
        """Test that max_depth constraint is respected."""
        tree = DecisionTree(max_depth=1, min_child_weight=0.0, reg_lambda=1.0, gamma=0.0)
        
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        grad = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        hess = np.ones(6)
        
        tree.fit(X, grad, hess)
        
        assert tree.get_depth() <= 1
    
    def test_min_child_weight_constraint(self):
        """Test that min_child_weight constraint is respected."""
        tree = DecisionTree(max_depth=5, min_child_weight=10.0, reg_lambda=1.0, gamma=0.0)
        
        # Small hessian values
        X = np.array([[1.0], [2.0], [3.0]])
        grad = np.array([1.0, 1.0, 1.0])
        hess = np.ones(3)  # Total hess = 3, but each child needs >= 10
        
        tree.fit(X, grad, hess)
        
        # Should create a single leaf due to constraint
        assert tree.get_num_leaves() == 1
    
    def test_get_num_leaves(self):
        """Test getting number of leaves."""
        tree = DecisionTree(max_depth=2, min_child_weight=0.0, reg_lambda=1.0, gamma=0.0)
        
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        grad = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        hess = np.ones(6)
        
        tree.fit(X, grad, hess)
        
        num_leaves = tree.get_num_leaves()
        assert num_leaves >= 1
        assert num_leaves <= 2 ** 2  # Max leaves for depth 2
