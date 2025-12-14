"""Visualization and utility helpers: feature importance, training curves, tree dump."""

from typing import Dict, List, Optional, Tuple
import numpy as np

def _traverse(node, func):
    if node is None:
        return
    func(node)
    if not node.is_leaf:
        _traverse(node.left_child, func)
        _traverse(node.right_child, func)


def compute_feature_importance(booster, importance_type: str = "gain") -> Dict[int, float]:
    """Compute feature importance from a fitted booster.

    importance_type: 'gain' | 'weight' | 'cover'
    Returns mapping {feature_index: importance_value}
    """
    if not hasattr(booster, 'trees'):
        raise ValueError("Booster has no attribute 'trees'")

    n_features = getattr(booster, 'n_features_', None)
    if n_features is None:
        # try to infer
        n_features = 0
        for t in booster.trees:
            if t.root is not None:
                # assume number of features equals max split feature + 1
                def _check(n):
                    nonlocal n_features
                    if not n.is_leaf and n.split_feature is not None:
                        n_features = max(n_features, n.split_feature + 1)
                _traverse(t.root, _check)

    imp = {i: 0.0 for i in range(n_features)}

    for tree in booster.trees:
        root = getattr(tree, 'root', None)
        if root is None:
            continue

        def _acc(node):
            if node.is_leaf:
                return
            f = node.split_feature
            if f is None:
                return
            if importance_type == 'weight':
                imp[f] = imp.get(f, 0.0) + 1.0
            elif importance_type == 'cover':
                # cover = sum of hessians in children
                left = node.left_child
                right = node.right_child
                cover = 0.0
                if left is not None:
                    cover += getattr(left, 'hess_sum', 0.0)
                if right is not None:
                    cover += getattr(right, 'hess_sum', 0.0)
                imp[f] = imp.get(f, 0.0) + cover
            else:
                # gain: compute from children stats
                left = node.left_child
                right = node.right_child
                if left is None or right is None:
                    return
                g_l = getattr(left, 'grad_sum', 0.0)
                h_l = getattr(left, 'hess_sum', 0.0)
                g_r = getattr(right, 'grad_sum', 0.0)
                h_r = getattr(right, 'hess_sum', 0.0)
                reg = getattr(tree, 'reg_lambda', 1.0)
                gamma = getattr(tree, 'gamma', 0.0)
                # simple gain formula
                parent_g = g_l + g_r
                parent_h = h_l + h_r
                eps = 1e-10
                score_l = (g_l ** 2) / max(h_l + reg, eps)
                score_r = (g_r ** 2) / max(h_r + reg, eps)
                score_p = (parent_g ** 2) / max(parent_h + reg, eps)
                gain = 0.5 * (score_l + score_r - score_p) - gamma
                imp[f] = imp.get(f, 0.0) + max(0.0, gain)

        _traverse(root, _acc)

    return imp


def dump_tree_text(tree, max_depth: Optional[int] = None) -> str:
    """Return a textual dump of a single DecisionTree (preorder)."""
    lines: List[str] = []

    def _dump(node, depth=0):
        if max_depth is not None and depth > max_depth:
            return
        prefix = '  ' * depth
        if node.is_leaf:
            lines.append(f"{prefix}leaf value={node.leaf_value} grad_sum={node.grad_sum:.4f} hess_sum={node.hess_sum:.4f}")
            return
        lines.append(f"{prefix}node feature={node.split_feature} threshold={node.split_threshold} grad_sum={node.grad_sum:.4f} hess_sum={node.hess_sum:.4f}")
        _dump(node.left_child, depth + 1)
        _dump(node.right_child, depth + 1)

    _dump(getattr(tree, 'root', None), 0)
    return '\n'.join(lines)


def export_tree_dot(tree, node_name: str = 'root') -> str:
    """Export tree to DOT format (Graphviz) as a string."""
    lines: List[str] = []
    lines.append('digraph Tree {')

    counter = {'id': 0}

    def _node_id():
        i = counter['id']
        counter['id'] += 1
        return f'n{i}'

    def _emit(node):
        nid = _node_id()
        if node.is_leaf:
            label = f'leaf\nvalue={node.leaf_value:.4f}'
            lines.append(f'  {nid} [label="{label}", shape=box];')
            return nid
        label = f'f={node.split_feature}\nth={node.split_threshold:.4f}\ng={node.grad_sum:.4f}'
        lines.append(f'  {nid} [label="{label}"];')
        left_id = _emit(node.left_child)
        right_id = _emit(node.right_child)
        lines.append(f'  {nid} -> {left_id} [label="L"];')
        lines.append(f'  {nid} -> {right_id} [label="R"];')
        return nid

    root = getattr(tree, 'root', None)
    if root is not None:
        _emit(root)

    lines.append('}')
    return '\n'.join(lines)
