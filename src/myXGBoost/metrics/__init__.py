"""Evaluation metrics."""

# Regression metrics
from myXGBoost.metrics.regression import rmse, mae, r2_score, RMSE, MAE, R2Score

# Classification metrics
from myXGBoost.metrics.classification import (
    accuracy_score, log_loss, auc_score, 
    Accuracy, LogLoss, AUC
)

__all__ = [
    # Regression functions
    "rmse",
    "mae",
    "r2_score",
    # Regression classes
    "RMSE",
    "MAE",
    "R2Score",
    # Classification functions
    "accuracy_score",
    "log_loss",
    "auc_score",
    # Classification classes
    "Accuracy",
    "LogLoss",
    "AUC",
]
