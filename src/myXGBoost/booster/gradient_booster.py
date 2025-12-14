"""Gradient boosting core implementation."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Tuple, Callable
from myXGBoost.trees.decision_tree import DecisionTree
from myXGBoost.loss.base import LossFunction


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


class GradientBooster(BoosterBase):
    """
    Gradient boosting implementation.
    
    Implements the additive model:
    - Start with initial prediction
    - Iteratively add trees
    - Update prediction: y_pred += learning_rate * tree.predict(X)
    
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
        random_state: Optional[int] = None
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
        
        # Model state
        self.trees: List[DecisionTree] = []
        self.initial_prediction: Optional[float] = None
        self.n_features_: Optional[int] = None
        self.best_iteration_: Optional[int] = None
        
        # Early stopping
        self.early_stopping_rounds: Optional[int] = None
        self.eval_sets: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.eval_metric: Optional[Callable] = None
        self.eval_results: List[dict] = []
    
    def _calculate_initial_prediction(self, y: np.ndarray, is_classification: bool = False) -> float:
        """
        Calculate initial prediction.
        
        For regression: mean of y
        For classification: log-odds (log(p / (1-p)) where p = mean(y))
        
        Parameters
        ----------
        y : ndarray
            Target values.
        is_classification : bool, default=False
            Whether this is a classification task.
            
        Returns
        -------
        initial_pred : float
            Initial prediction value.
        """
        if is_classification:
            # For binary classification: log-odds
            # p = mean(y), log-odds = log(p / (1-p))
            p = np.mean(y)
            # Avoid log(0) or log(inf)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return np.log(p / (1 - p))
        else:
            # For regression: mean
            return float(np.mean(y))
    
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
            Validation sets for early stopping.
        eval_metric : callable, optional
            Metric function for evaluation.
        early_stopping_rounds : int, optional
            Number of rounds with no improvement to trigger early stopping.
        verbose : bool, default=False
            If True, prints progress.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Determine if classification (binary)
        # Check if loss function is a classification loss
        from myXGBoost.loss.base import ClassificationLoss
        is_classification = isinstance(self.loss_function, ClassificationLoss)
        
        # For classification, y should be encoded as 0/1
        if is_classification:
            y_encoded = y.copy()
        else:
            y_encoded = y
        
        # Calculate initial prediction
        self.initial_prediction = self._calculate_initial_prediction(y_encoded, is_classification)
        
        # Initialize predictions
        y_pred = np.full(n_samples, self.initial_prediction, dtype=np.float64)
        
        # Store for early stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_sets = eval_set
        self.eval_metric = eval_metric
        self.eval_results = []
        best_score = float('inf')
        best_iteration = 0
        no_improvement_count = 0
        
        # Iteratively build trees
        for iteration in range(self.n_estimators):
            # Compute gradients and hessians
            grad, hess = self.loss_function.grad_hess(y_encoded, y_pred)
            
            # Apply sample weights if provided
            if sample_weight is not None:
                grad = grad * sample_weight
                hess = hess * sample_weight
            
            # Row subsampling
            row_indices = self._sample_rows(n_samples)
            X_sampled = X[row_indices]
            grad_sampled = grad[row_indices]
            hess_sampled = hess[row_indices]
            
            # Column subsampling
            feature_indices = self._sample_features(n_features)
            
            # Build tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X_sampled, grad_sampled, hess_sampled, feature_indices)
            
            # Get tree predictions for all samples (not just sampled)
            tree_pred = tree.predict(X)
            
            # Update predictions: y_pred += learning_rate * tree_pred
            y_pred += self.learning_rate * tree_pred
            
            # Store tree
            self.trees.append(tree)
            
            # Evaluate on validation sets
            if eval_set is not None and eval_metric is not None:
                eval_result = {}
                for i, (X_eval, y_eval) in enumerate(eval_set):
                    y_pred_eval = self._predict_raw(X_eval)
                    score = eval_metric(y_eval, y_pred_eval)
                    eval_result[f'val_{i}'] = score
                
                self.eval_results.append(eval_result)
                
                # Early stopping check
                if early_stopping_rounds is not None:
                    # Use first validation set for early stopping
                    current_score = eval_result.get('val_0', float('inf'))
                    if current_score < best_score:
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
            
            if verbose and (iteration + 1) % 10 == 0:
                train_loss = self.loss_function.loss(y_encoded, y_pred)
                print(f"Iteration {iteration + 1}/{self.n_estimators}, "
                      f"train_loss: {train_loss:.6f}")
        
        if self.best_iteration_ is None:
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
        y_pred : ndarray of shape (n_samples,)
            Raw predictions.
        """
        if self.initial_prediction is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Start with initial prediction
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
        proba : ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        y_pred_raw = self._predict_raw(X)
        
        # Apply sigmoid to get probabilities
        from myXGBoost.loss.classification import sigmoid
        proba_positive = sigmoid(y_pred_raw)
        proba_negative = 1 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
