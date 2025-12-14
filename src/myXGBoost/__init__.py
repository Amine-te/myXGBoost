"""
myXGBoost: A modular XGBoost implementation following sklearn design patterns.
"""

from myXGBoost.version import __version__
from myXGBoost.estimators import XGBRegressor, XGBClassifier

__all__ = [
    "__version__",
    "XGBRegressor",
    "XGBClassifier",
]

