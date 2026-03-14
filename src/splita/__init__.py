"""splita — A/B test analysis that is correct by default, informative by design."""

__version__ = "0.1.0"

from splita._types import (
    BanditResult,
    BoundaryResult,
    CorrectionResult,
    ExperimentResult,
    GSResult,
    SampleSizeResult,
    SRMResult,
    mSPRTResult,
    mSPRTState,
)
from splita.core import Experiment, MultipleCorrection, SampleSize, SRMCheck

__all__ = [
    # Result types
    "BanditResult",
    "BoundaryResult",
    "CorrectionResult",
    # Core classes
    "Experiment",
    "ExperimentResult",
    "GSResult",
    "MultipleCorrection",
    "SRMCheck",
    "SRMResult",
    "SampleSize",
    "SampleSizeResult",
    "mSPRTResult",
    "mSPRTState",
]
