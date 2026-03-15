"""Variance reduction techniques for A/B testing."""

from splita.variance.adaptive_winsorization import AdaptiveWinsorizer
from splita.variance.cluster_bootstrap import ClusterBootstrap
from splita.variance.cupac import CUPAC
from splita.variance.cuped import CUPED
from splita.variance.double_ml import DoubleML
from splita.variance.inex import InExperimentVR
from splita.variance.multivariate_cuped import MultivariateCUPED
from splita.variance.outliers import OutlierHandler
from splita.variance.post_stratification import PostStratification
from splita.variance.regression_adjustment import RegressionAdjustment
from splita.variance.nonstationary import NonstationaryAdjustment
from splita.variance.ppi import PredictionPoweredInference
from splita.variance.robust_estimators import RobustMeanEstimator
from splita.variance.trimmed_mean import TrimmedMeanEstimator

__all__ = [
    "CUPAC",
    "CUPED",
    "AdaptiveWinsorizer",
    "ClusterBootstrap",
    "DoubleML",
    "InExperimentVR",
    "MultivariateCUPED",
    "OutlierHandler",
    "PostStratification",
    "RegressionAdjustment",
    "NonstationaryAdjustment",
    "PredictionPoweredInference",
    "RobustMeanEstimator",
    "TrimmedMeanEstimator",
]
