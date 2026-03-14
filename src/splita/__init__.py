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
from splita.bandits import LinTS, ThompsonSampler
from splita.core import Experiment, MultipleCorrection, SampleSize, SRMCheck
from splita.sequential import GroupSequential, mSPRT

__all__ = [
    # Result types
    "BanditResult",
    "BoundaryResult",
    "CorrectionResult",
    # Core classes
    "Experiment",
    "ExperimentResult",
    "GSResult",
    "GroupSequential",
    "LinTS",
    "MultipleCorrection",
    "SRMCheck",
    "SRMResult",
    "SampleSize",
    "SampleSizeResult",
    "ThompsonSampler",
    "mSPRT",
    "mSPRTResult",
    "mSPRTState",
]
