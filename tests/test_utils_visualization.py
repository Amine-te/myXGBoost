import numpy as np

from myXGBoost.utils.visualization import compute_feature_importance, dump_tree_text, export_tree_dot
from myXGBoost.trees.decision_tree import DecisionTree
from myXGBoost.base.tree import TreeNode


def build_simple_tree():
    # Build a simple hand-crafted tree
    root = TreeNode()
    root.set_split(0, 0.5)
    root.grad_sum = 2.0
    root.hess_sum = 2.0
    left = TreeNode()
    left.set_leaf_value(0.1)
    left.grad_sum = 1.0
    left.hess_sum = 1.0
    right = TreeNode()
    right.set_leaf_value(-0.2)
    right.grad_sum = 1.0
    right.hess_sum = 1.0
    root.set_children(left, right)
    tree = DecisionTree()
    tree.root = root
    return tree


def test_compute_feature_importance():
    tree = build_simple_tree()
    class B: pass
    booster = B()
    booster.trees = [tree]
    booster.n_features_ = 1

    imp = compute_feature_importance(booster, importance_type='weight')
    assert isinstance(imp, dict)
    assert imp.get(0, 0) == 1.0

    imp_gain = compute_feature_importance(booster, importance_type='gain')
    assert imp_gain[0] >= 0.0


def test_dump_and_dot():
    tree = build_simple_tree()
    txt = dump_tree_text(tree)
    assert 'node' in txt
    dot = export_tree_dot(tree)
    assert 'digraph' in dot

    # ensure dot compiles minimally by checking for node ids
    assert 'n0' in dot
