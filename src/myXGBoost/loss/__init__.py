"""Loss functions for regression and classification."""

from myXGBoost.loss.base import LossFunction, RegressionLoss, ClassificationLoss
from myXGBoost.loss.regression import MSELoss
from myXGBoost.loss.classification import LogisticLoss

__all__ = [
    "LossFunction",
    "RegressionLoss",
    "ClassificationLoss",
    "MSELoss",
    "LogisticLoss",
]

