"""Parameter validation and default values."""

import numpy as np


def validate_booster_params(
    learning_rate=None,
    n_estimators=None,
    max_depth=None,
    min_child_weight=None,
    gamma=None,
    subsample=None,
    colsample_bytree=None,
):
    """
    Validate hyperparameters for gradient boosting models.
    
    Parameters
    ----------
    learning_rate : float, optional
        Learning rate (eta). Must be > 0.
    n_estimators : int, optional
        Number of boosting rounds. Must be > 0.
    max_depth : int, optional
        Maximum tree depth. Must be > 0.
    min_child_weight : float, optional
        Minimum child weight. Must be >= 0.
    gamma : float, optional
        Minimum loss reduction. Must be >= 0.
    subsample : float, optional
        Subsample ratio. Must be in (0, 1].
    colsample_bytree : float, optional
        Column subsample ratio. Must be in (0, 1].
        
    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    """
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {learning_rate!r}."
            )
    
    if n_estimators is not None:
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be > 0, got {n_estimators!r}."
            )
    
    if max_depth is not None:
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError(
                f"max_depth must be > 0, got {max_depth!r}."
            )
    
    if min_child_weight is not None:
        if not isinstance(min_child_weight, (int, float)) or min_child_weight < 0:
            raise ValueError(
                f"min_child_weight must be >= 0, got {min_child_weight!r}."
            )
    
    if gamma is not None:
        if not isinstance(gamma, (int, float)) or gamma < 0:
            raise ValueError(
                f"gamma must be >= 0, got {gamma!r}."
            )
    
    if subsample is not None:
        if not isinstance(subsample, (int, float)) or subsample <= 0 or subsample > 1:
            raise ValueError(
                f"subsample must be in (0, 1], got {subsample!r}."
            )
    
    if colsample_bytree is not None:
        if not isinstance(colsample_bytree, (int, float)) or colsample_bytree <= 0 or colsample_bytree > 1:
            raise ValueError(
                f"colsample_bytree must be in (0, 1], got {colsample_bytree!r}."
            )
