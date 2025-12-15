"""Gradient boosting core implementation."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Tuple, Callable
from myXGBoost.trees.decision_tree import DecisionTree
from myXGBoost.loss.base import LossFunction
from joblib import Parallel, delayed
from myXGBoost.utils.parallel import compute_gradients_parallel


class BoosterBase(ABC):
    """
    Abstract base class for boosting implementations.
    
    This class defines the interface that all booster implementations should follow.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the booster to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted booster."""
        pass



def _train_single_tree(
    X: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    row_indices: np.ndarray,
    feature_indices: np.ndarray,
    tree_params: dict
) -> Tuple[DecisionTree, np.ndarray]:
    """
    Train a single tree on a subset of data (helper for parallel execution).
    
    Returns
    -------
    tree : DecisionTree
        Trained tree.
    preds : ndarray
        Predictions on the full dataset X.
    """
    X_sampled = np.asfortranarray(X[row_indices])
    grad_sampled = grad[row_indices]
    hess_sampled = hess[row_indices]
    
    tree = DecisionTree(**tree_params)
    tree.fit(X_sampled, grad_sampled, hess_sampled, feature_indices)
    
    # Predict on all samples for updating residuals
    preds = tree.predict(X)
    return tree, preds


class GradientBooster(BoosterBase):
    """
    Gradient boosting implementation.
    
    Implements the additive model:
    - Start with initial prediction
    - Iteratively add trees
    - Update prediction: y_pred += learning_rate * tree.predict(X)
    
    Uses hybrid split finding for automatic optimization selection.
    For small datasets uses exact greedy algorithm.
    For large datasets uses approximate histogram-based algorithm.
    
    Parameters
    ----------
    loss_function : LossFunction
        Loss function to use (e.g., MSELoss, LogisticLoss).
    n_estimators : int, default=100
        Number of boosting rounds (trees).
    learning_rate : float, default=0.1
        Learning rate (shrinkage).
    max_depth : int, default=6
        Maximum depth of trees.
    min_child_weight : float, default=1.0
        Minimum sum of hessians in a child node.
    gamma : float, default=0.0
        Minimum loss reduction.
    reg_lambda : float, default=1.0
        L2 regularization parameter.
    subsample : float, default=1.0
        Row subsampling ratio.
    colsample_bytree : float, default=1.0
        Column subsampling ratio.
    random_state : int, optional
        Random seed for reproducibility.
    use_hybrid_split_finder : bool, default=True
        Whether to use adaptive split finding (exact/approximate hybrid).
    exact_threshold : int, default=10000
        Switch to approximate method above this sample count.
    max_bins : int, default=256
        Maximum bins for histogram construction.
    use_parallel_gradients : bool, default=False
        Whether to compute gradients in parallel across data chunks.
        Default False for backward compatibility.
    n_jobs_gradients : int, default=-1
        Number of parallel jobs for gradient computation. -1 means use all cores.
        Only used if use_parallel_gradients=True.
    """
    
    def __init__(
        self,
        loss_function: LossFunction,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_lambda: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: Optional[int] = None,
        use_hybrid_split_finder: bool = True,
        exact_threshold: int = 10000,
        max_bins: int = 256,
        use_parallel_gradients: bool = False,
        n_jobs_gradients: int = -1
    ):
        self.loss_function = loss_function
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_hybrid_split_finder = use_hybrid_split_finder
        self.exact_threshold = exact_threshold
        self.max_bins = max_bins
        self.use_parallel_gradients = use_parallel_gradients
        self.n_jobs_gradients = n_jobs_gradients
        
        # Model state
        self.trees: List[DecisionTree] = []  # For binary/regression
        self.trees_multiclass: List[List[DecisionTree]] = []  # For multiclass: trees_multiclass[class_idx][tree_idx]
        self.initial_prediction: Optional[float] = None  # For binary/regression
        self.initial_predictions_multiclass: Optional[np.ndarray] = None  # For multiclass: (n_classes,)
        self.n_features_: Optional[int] = None
        self.n_classes_: Optional[int] = None  # Number of classes (None for regression, 2 for binary, >2 for multiclass)
        self.best_iteration_: Optional[int] = None
        
        # Early stopping
        self.early_stopping_rounds: Optional[int] = None
        self.eval_sets: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.eval_metric: Optional[Callable] = None
        self.eval_results: List[dict] = []
    
    def _calculate_initial_prediction(self, y: np.ndarray, n_classes: Optional[int] = None):
        """
        Calculate initial prediction.
        
        For regression: mean of y
        For binary classification: log-odds (log(p / (1-p)) where p = mean(y))
        For multiclass: log of class probabilities (softmax initialization)
        
        Parameters
        ----------
        y : ndarray
            Target values (encoded as class indices for classification).
        n_classes : int, optional
            Number of classes (None for regression, 2 for binary, >2 for multiclass).
            
        Returns
        -------
        initial_pred : float or ndarray
            Initial prediction value(s). Float for regression/binary, array for multiclass.
        """
        if n_classes is None:
            # Regression: mean
            return float(np.mean(y))
        elif n_classes == 2:
            # Binary classification: log-odds
            p = np.mean(y)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return np.log(p / (1 - p))
        else:
            # Multiclass: log probabilities for each class
            # Initialize with class frequencies
            class_counts = np.bincount(y.astype(int), minlength=n_classes)
            class_probs = (class_counts + 1) / (len(y) + n_classes)  # Laplace smoothing
            # Return log probabilities (will be used as initial logits)
            return np.log(class_probs).astype(np.float64)
    
    def _sample_rows(self, n_samples: int) -> np.ndarray:
        """
        Sample row indices for subsampling.
        
        Parameters
        ----------
        n_samples : int
            Total number of samples.
            
        Returns
        -------
        indices : ndarray
            Sampled row indices.
        """
        if self.subsample >= 1.0:
            return np.arange(n_samples)
        
        n_selected = int(self.subsample * n_samples)
        if n_selected == 0:
            n_selected = 1
        
        return np.random.choice(n_samples, size=n_selected, replace=False)
    
    def _sample_features(self, n_features: int) -> np.ndarray:
        """
        Sample feature indices for column subsampling.
        
        Parameters
        ----------
        n_features : int
            Total number of features.
            
        Returns
        -------
        indices : ndarray
            Sampled feature indices.
        """
        if self.colsample_bytree >= 1.0:
            return np.arange(n_features)
        
        n_selected = int(self.colsample_bytree * n_features)
        if n_selected == 0:
            n_selected = 1
        
        return np.random.choice(n_features, size=n_selected, replace=False)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        eval_metric: Optional[Callable] = None,
        eval_names: Optional[List[str]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Fit the gradient boosting model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        sample_weight : ndarray, optional
            Sample weights.
        eval_set : list of tuples, optional
            Validation sets for early stopping. Each tuple is (X_eval, y_eval).
        eval_metric : callable or Metric, optional
            Metric function for evaluation. Can be a callable or Metric object.
        eval_names : list of str, optional
            Names for evaluation sets. Default: ['eval_0', 'eval_1', ...].
        early_stopping_rounds : int, optional
            Number of rounds with no improvement to trigger early stopping.
        verbose : bool, default=False
            If True, prints progress.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asfortranarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Determine if classification and number of classes
        from myXGBoost.loss.base import ClassificationLoss
        from myXGBoost.loss.softmax_loss import SoftmaxLoss
        is_classification = isinstance(self.loss_function, ClassificationLoss)
        
        if is_classification:
            # Determine number of classes
            if isinstance(self.loss_function, SoftmaxLoss):
                self.n_classes_ = self.loss_function.n_classes
            else:
                # Binary classification
                self.n_classes_ = 2
            y_encoded = y.copy()
        else:
            # Regression
            self.n_classes_ = None
            y_encoded = y
        
        # Calculate initial prediction
        if self.n_classes_ is not None and self.n_classes_ > 2:
            # Multiclass
            self.initial_predictions_multiclass = self._calculate_initial_prediction(y_encoded, self.n_classes_)
            self.initial_prediction = None
            # Initialize per-class tree lists
            self.trees_multiclass = [[] for _ in range(self.n_classes_)]
            self.trees = []
        else:
            # Binary or regression
            self.initial_prediction = self._calculate_initial_prediction(y_encoded, self.n_classes_)
            self.initial_predictions_multiclass = None
            self.trees = []
            self.trees_multiclass = []
        
        # Initialize predictions
        if self.n_classes_ is not None and self.n_classes_ > 2:
            # Multiclass: y_pred is (n_samples, n_classes)
            y_pred = np.tile(self.initial_predictions_multiclass, (n_samples, 1))
        else:
            # Binary or regression: y_pred is (n_samples,)
            y_pred = np.full(n_samples, self.initial_prediction, dtype=np.float64)
        
        # Store for early stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_sets = eval_set
        self.eval_metric = eval_metric
        self.eval_results = []
        
        # Set default eval names
        if eval_set is not None and eval_names is None:
            eval_names = [f'eval_{i}' for i in range(len(eval_set))]
        
        best_score = float('inf') if (eval_metric is not None and hasattr(eval_metric, 'is_higher_better') and not eval_metric.is_higher_better()) or (callable(eval_metric) and not hasattr(eval_metric, 'is_higher_better')) else float('-inf')
        best_iteration = 0
        no_improvement_count = 0
        
        
        # Iteratively build trees
        # Only create Parallel context for multiclass (where it's actually used)
        # For binary/regression, avoid the overhead of unused parallel context
        is_multiclass = self.n_classes_ is not None and self.n_classes_ > 2
        
        if is_multiclass:
            # Multiclass: train one tree per class IN PARALLEL
            with Parallel(n_jobs=-1) as parallel:
                for iteration in range(self.n_estimators):
                    # Compute gradients and hessians
                    if self.use_parallel_gradients:
                        grad, hess = compute_gradients_parallel(
                            self.loss_function,
                            y_encoded,
                            y_pred,
                            n_jobs=self.n_jobs_gradients
                        )
                    else:
                        # Sequential computation (backward compatible default)
                        grad, hess = self.loss_function.grad_hess(y_encoded, y_pred)
                    
                    # Apply sample weights if provided
                    if sample_weight is not None:
                        grad = grad * sample_weight if grad.ndim == 1 else grad * sample_weight[:, np.newaxis]
                        hess = hess * sample_weight if hess.ndim == 1 else hess * sample_weight[:, np.newaxis]
                    
                    tree_params = {
                        'max_depth': self.max_depth,
                        'min_child_weight': self.min_child_weight,
                        'reg_lambda': self.reg_lambda,
                        'gamma': self.gamma,
                        'use_hybrid_split_finder': self.use_hybrid_split_finder,
                        'exact_threshold': self.exact_threshold,
                        'max_bins': self.max_bins
                    }
                    
                    # Pre-sample indices for each class
                    tasks = []
                    for class_idx in range(self.n_classes_):
                        row_indices = self._sample_rows(n_samples)
                        feature_indices = self._sample_features(n_features)
                        
                        tasks.append((
                            X,
                            grad[:, class_idx],
                            hess[:, class_idx],
                            row_indices,
                            feature_indices,
                            tree_params
                        ))
                    
                    # Execute in parallel reusing the pool
                    results = parallel(
                        delayed(_train_single_tree)(*args) for args in tasks
                    )
                    
                    # Process results
                    for class_idx, (tree, tree_pred) in enumerate(results):
                        # Update predictions for this class
                        y_pred[:, class_idx] += self.learning_rate * tree_pred
                        
                        # Store tree
                        self.trees_multiclass[class_idx].append(tree)
                    
                    # Evaluate on validation sets (same as binary/regression)
                    if eval_set is not None and eval_metric is not None:
                        eval_result = {}
                        for i, (X_eval, y_eval) in enumerate(eval_set):
                            y_pred_eval = self._predict_raw(X_eval)
                            
                            # Support both function-based and Metric class-based metrics
                            if hasattr(eval_metric, 'score'):
                                # Metric class
                                score = eval_metric.score(y_eval, y_pred_eval)
                                metric_name = eval_metric.name
                            else:
                                # Function-based metric
                                score = eval_metric(y_eval, y_pred_eval)
                                metric_name = 'metric'
                            
                            eval_name = eval_names[i] if eval_names else f'eval_{i}'
                            eval_result[eval_name] = score
                        
                        self.eval_results.append(eval_result)
                        
                        # Early stopping check
                        if early_stopping_rounds is not None:
                            # Use first validation set for early stopping
                            eval_name = eval_names[0] if eval_names else 'eval_0'
                            current_score = eval_result.get(eval_name, float('inf'))
                            
                            # Determine if higher or lower is better
                            is_higher_better = True
                            if hasattr(eval_metric, 'is_higher_better'):
                                is_higher_better = eval_metric.is_higher_better()
                            
                            # Check for improvement
                            if is_higher_better:
                                improved = current_score > best_score
                            else:
                                improved = current_score < best_score
                            
                            if improved:
                                best_score = current_score
                                best_iteration = iteration
                                no_improvement_count = 0
                            else:
                                no_improvement_count += 1
                            
                            if no_improvement_count >= early_stopping_rounds:
                                self.best_iteration_ = best_iteration
                                if verbose:
                                    print(f"Early stopping at iteration {iteration + 1}, "
                                          f"best iteration: {best_iteration + 1}")
                                break
                        
                        # Print evaluation results if verbose
                        if verbose:
                            results_str = ", ".join([f"{k}: {v:.6f}" for k, v in eval_result.items()])
                            print(f"Iteration {iteration + 1}: {results_str}")
                    
                    if verbose and eval_set is None and (iteration + 1) % 10 == 0:
                        train_loss = self.loss_function.loss(y_encoded, y_pred)
                        print(f"Iteration {iteration + 1}/{self.n_estimators}, "
                              f"train_loss: {train_loss:.6f}")
        else:
            # Binary or regression: no parallel context overhead
            for iteration in range(self.n_estimators):
                # Compute gradients and hessians
                if self.use_parallel_gradients:
                    grad, hess = compute_gradients_parallel(
                        self.loss_function,
                        y_encoded,
                        y_pred,
                        n_jobs=self.n_jobs_gradients
                    )
                else:
                    # Sequential computation (backward compatible default)
                    grad, hess = self.loss_function.grad_hess(y_encoded, y_pred)
                
                # Apply sample weights if provided
                if sample_weight is not None:
                    grad = grad * sample_weight if grad.ndim == 1 else grad * sample_weight[:, np.newaxis]
                    hess = hess * sample_weight if hess.ndim == 1 else hess * sample_weight[:, np.newaxis]
                
                # Binary or regression: single tree per iteration
                # Row subsampling
                row_indices = self._sample_rows(n_samples)
                X_sampled = np.asfortranarray(X[row_indices])
                grad_sampled = grad[row_indices]
                hess_sampled = hess[row_indices]
                
                # Column subsampling
                feature_indices = self._sample_features(n_features)
                
                # Build tree
                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                    use_hybrid_split_finder=self.use_hybrid_split_finder,
                    exact_threshold=self.exact_threshold,
                    max_bins=self.max_bins
                )
                tree.fit(X_sampled, grad_sampled, hess_sampled, feature_indices)
                
                # Get tree predictions for all samples
                tree_pred = tree.predict(X)
                
                # Update predictions
                y_pred += self.learning_rate * tree_pred
                
                # Store tree
                self.trees.append(tree)
                
                # Evaluate on validation sets
                if eval_set is not None and eval_metric is not None:
                    eval_result = {}
                    for i, (X_eval, y_eval) in enumerate(eval_set):
                        y_pred_eval = self._predict_raw(X_eval)
                        
                        # Support both function-based and Metric class-based metrics
                        if hasattr(eval_metric, 'score'):
                            # Metric class
                            score = eval_metric.score(y_eval, y_pred_eval)
                            metric_name = eval_metric.name
                        else:
                            # Function-based metric
                            score = eval_metric(y_eval, y_pred_eval)
                            metric_name = 'metric'
                        
                        eval_name = eval_names[i] if eval_names else f'eval_{i}'
                        eval_result[eval_name] = score
                    
                    self.eval_results.append(eval_result)
                    
                    # Early stopping check
                    if early_stopping_rounds is not None:
                        # Use first validation set for early stopping
                        eval_name = eval_names[0] if eval_names else 'eval_0'
                        current_score = eval_result.get(eval_name, float('inf'))
                        
                        # Determine if higher or lower is better
                        is_higher_better = True
                        if hasattr(eval_metric, 'is_higher_better'):
                            is_higher_better = eval_metric.is_higher_better()
                        
                        # Check for improvement
                        if is_higher_better:
                            improved = current_score > best_score
                        else:
                            improved = current_score < best_score
                        
                        if improved:
                            best_score = current_score
                            best_iteration = iteration
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                        
                        if no_improvement_count >= early_stopping_rounds:
                            self.best_iteration_ = best_iteration
                            if verbose:
                                print(f"Early stopping at iteration {iteration + 1}, "
                                      f"best iteration: {best_iteration + 1}")
                            break
                    
                    # Print evaluation results if verbose
                    if verbose:
                        results_str = ", ".join([f"{k}: {v:.6f}" for k, v in eval_result.items()])
                        print(f"Iteration {iteration + 1}: {results_str}")
                
                if verbose and eval_set is None and (iteration + 1) % 10 == 0:
                    train_loss = self.loss_function.loss(y_encoded, y_pred)
                    print(f"Iteration {iteration + 1}/{self.n_estimators}, "
                          f"train_loss: {train_loss:.6f}")
            
            if self.best_iteration_ is None:
                if self.n_classes_ is not None and self.n_classes_ > 2:
                    self.best_iteration_ = len(self.trees_multiclass[0]) - 1
                else:
                    self.best_iteration_ = len(self.trees) - 1

        
        return self

    
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Predict raw values (before transformation).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray
            Raw predictions. Shape (n_samples,) for binary/regression, 
            (n_samples, n_classes) for multiclass.
        """
        if self.initial_prediction is None and self.initial_predictions_multiclass is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        if self.n_classes_ is not None and self.n_classes_ > 2:
            # Multiclass prediction
            y_pred = np.tile(self.initial_predictions_multiclass, (n_samples, 1))
            
            # Add contributions from all trees for each class
            n_trees = len(self.trees_multiclass[0]) if self.best_iteration_ is None else self.best_iteration_ + 1
            for class_idx in range(self.n_classes_):
                for i in range(n_trees):
                    tree_pred = self.trees_multiclass[class_idx][i].predict(X)
                    y_pred[:, class_idx] += self.learning_rate * tree_pred
        else:
            # Binary or regression prediction
            y_pred = np.full(n_samples, self.initial_prediction, dtype=np.float64)
            
            # Add contributions from all trees
            n_trees = len(self.trees) if self.best_iteration_ is None else self.best_iteration_ + 1
            for i in range(n_trees):
                tree_pred = self.trees[i].predict(X)
                y_pred += self.learning_rate * tree_pred
        
        return y_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        
        For regression: returns raw predictions.
        For classification: returns class probabilities (after sigmoid).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray
            Predicted values.
        """
        y_pred_raw = self._predict_raw(X)
        
        # For classification, apply sigmoid; for regression, return as is
        # We'll determine this based on the loss function
        # For now, return raw predictions (will be handled by estimator classes)
        return y_pred_raw
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classification only).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        y_pred_raw = self._predict_raw(X)
        
        if self.n_classes_ is not None and self.n_classes_ > 2:
            # Multiclass: apply softmax
            from myXGBoost.loss.softmax_loss import softmax
            return softmax(y_pred_raw)
        else:
            # Binary: apply sigmoid
            from myXGBoost.loss.classification import sigmoid
            proba_positive = sigmoid(y_pred_raw)
            proba_negative = 1 - proba_positive
            return np.column_stack([proba_negative, proba_positive])
