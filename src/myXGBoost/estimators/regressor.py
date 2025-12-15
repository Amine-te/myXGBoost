"""XGBRegressor: Gradient boosting regressor following sklearn API."""

import numpy as np
from myXGBoost.base.base import BaseEstimator, RegressorMixin
from myXGBoost.booster.gradient_booster import BoosterBase
from myXGBoost.utils.validation import check_X_y, check_array
from myXGBoost.utils.params import validate_booster_params


class XGBRegressor(BaseEstimator, RegressorMixin, BoosterBase):
    """
    XGBoost-style gradient boosting regressor.
    
    This class implements a gradient boosting regressor following the
    sklearn API design pattern.
    
    Parameters
    ----------
    learning_rate : float, default=0.1
        Boosting learning rate (also known as eta).
    n_estimators : int, default=100
        Number of boosting rounds (trees).
    max_depth : int, default=6
        Maximum tree depth for base learners.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight (hessian) needed in a child.
    gamma : float, default=0.0
        Minimum loss reduction required to make a further partition
        on a leaf node of the tree (regularization parameter).
    subsample : float, default=1.0
        Subsample ratio of the training instances. Setting it to 0.5
        means that XGBoost would randomly sample half of the training
        data prior to growing trees.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    random_state : int or None, default=None
        Random number seed for reproducibility.
    verbose : bool, default=False
        If True, prints progress information during training.
        
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Only defined if the input
        features have names.
    booster_ : BoosterBase
        The underlying booster model (to be implemented).
    """
    
    def __init__(
        self,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        min_child_weight=1.0,
        gamma=0.0,
        reg_lambda=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=None,
        verbose=False,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None, verbose=None):
        """
        Fit the gradient boosting regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        eval_set : list of tuples (X, y), default=None
            Validation sets for early stopping.
        eval_metric : str or callable, default=None
            Metric to use for evaluation.
        verbose : bool, default=None
            If True, prints progress. If None, uses self.verbose.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate and convert inputs
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, y_numeric=True)
        
        # Validate hyperparameters
        validate_booster_params(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
        )
        
        # Store metadata
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns)
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Use provided verbose or instance verbose
        if verbose is None:
            verbose = self.verbose
        
        # Initialize and train the booster
        from myXGBoost.booster.gradient_booster import GradientBooster
        from myXGBoost.loss.regression import MSELoss
        
        self.booster_ = GradientBooster(
            loss_function=MSELoss(),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state
        )
        
        self.booster_.fit(X, y, sample_weight, eval_set, eval_metric, None, verbose)
        
        return self
    
    def predict(self, X):
        """
        Predict target values for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Check if fit has been called
        if not hasattr(self, 'n_features_in_'):
            raise ValueError("This XGBRegressor instance is not fitted yet.")
        
        # Validate input
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but XGBRegressor is expecting "
                f"{self.n_features_in_} features as input."
            )
        
        # Use booster to predict
        return self.booster_.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Note: This method is not applicable for regression.
        It raises a NotImplementedError.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Raises
        ------
        NotImplementedError
            This method is only available for classification.
        """
        raise NotImplementedError(
            "predict_proba is not available for regression. "
            "Use predict() instead."
        )
