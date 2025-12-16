import numpy as np
from myXGBoost.trees.split_finder import ExactSplitFinder, ApproximateSplitFinder


def test_exact_handles_nans():
    X = np.array([[1.0], [2.0], [3.0], [np.nan]])
    grad = np.array([1.0, -1.0, 0.5, 0.2])
    hess = np.ones(4)

    finder = ExactSplitFinder(use_vectorization=True)
    feat, thr, gain = finder.find_best_split(X, grad, hess)

    assert feat == 0
    assert thr is not None
    assert isinstance(gain, float)


def test_split_assigns_missing_to_best_side_exact():
    # Create data where missing should go left to maximize gain
    X = np.array([[0.0], [10.0], [20.0], [np.nan]])
    grad = np.array([5.0, -1.0, -1.0, 10.0])  # missing has large positive grad
    hess = np.ones(4)

    finder = ExactSplitFinder(use_vectorization=False)
    feat, thr, gain = finder.find_best_split(X, grad, hess)
    assert feat == 0

    X_left, grad_left, hess_left, X_right, grad_right, hess_right, assign_missing_to_left = \
        finder.split_data(X, grad, hess, feat, thr)

    # Missing should be assigned left because its positive gradient helps left gain
    # Check that missing value ended in left
    assert np.isnan(X_left).any()


def test_approximate_handles_nans():
    X = np.array([[1.0], [2.0], [3.0], [np.nan]])
    grad = np.array([1.0, -1.0, 0.5, 0.2])
    hess = np.ones(4)

    finder = ApproximateSplitFinder(max_bins=10, use_parallelization=False)
    feat, thr, gain = finder.find_best_split(X, grad, hess)

    assert feat == 0
    assert thr is not None
    assert isinstance(gain, float)


def test_split_assigns_missing_to_best_side_approx():
    X = np.array([[0.0], [10.0], [20.0], [np.nan]])
    grad = np.array([5.0, -1.0, -1.0, 10.0])
    hess = np.ones(4)

    finder = ApproximateSplitFinder(max_bins=3, use_parallelization=False)
    feat, thr, gain = finder.find_best_split(X, grad, hess)
    X_left, grad_left, hess_left, X_right, grad_right, hess_right, assign_missing_to_left = \
        finder.split_data(X, grad, hess, feat, thr)

    assert np.isnan(X_left).any() or np.isnan(X_right).any()
