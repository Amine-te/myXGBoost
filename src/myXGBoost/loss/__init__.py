"""Loss functions for regression and classification."""

from myXGBoost.loss.base import LossFunction, RegressionLoss, ClassificationLoss
from myXGBoost.loss.regression import MSELoss
from myXGBoost.loss.classification import LogisticLoss
from myXGBoost.loss.softmax_loss import SoftmaxLoss

__all__ = [
    "LossFunction",
    "RegressionLoss",
    "ClassificationLoss",
    "MSELoss",
    "LogisticLoss",
    "SoftmaxLoss",
]

