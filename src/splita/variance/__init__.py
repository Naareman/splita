"""Variance reduction techniques for A/B testing."""

from splita.variance.adaptive_winsorization import AdaptiveWinsorizer
from splita.variance.cupac import CUPAC
from splita.variance.cuped import CUPED
from splita.variance.double_ml import DoubleML
from splita.variance.multivariate_cuped import MultivariateCUPED
from splita.variance.outliers import OutlierHandler
from splita.variance.regression_adjustment import RegressionAdjustment

__all__ = [
    "CUPAC",
    "CUPED",
    "AdaptiveWinsorizer",
    "DoubleML",
    "MultivariateCUPED",
    "OutlierHandler",
    "RegressionAdjustment",
]
