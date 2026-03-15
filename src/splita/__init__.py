"""splita — A/B test analysis that is correct by default, informative by design."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("splita")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# ─── Core classes (most-used, top-level imports) ─────────────────────
from splita.audit_trail import audit_trail
from splita.auto import auto
from splita.bandits import LinTS, ThompsonSampler
from splita.check import check
from splita.compare import compare
from splita.core import (
    BayesianExperiment,
    Experiment,
    MultipleCorrection,
    SampleSize,
    SRMCheck,
)
from splita.diagnose import diagnose
from splita.errors import (
    InsufficientDataError,
    NotFittedError,
    SplitaError,
    ValidationError,
)
from splita.explain import explain
from splita.meta_analysis import meta_analysis
from splita.monitor import monitor
from splita.power_report import power_report
from splita.recommend import recommend
from splita.report import report
from splita.sequential import GroupSequential, mSPRT
from splita.simulate import simulate
from splita.variance import CUPAC, CUPED, OutlierHandler
from splita.verbose import verbose
from splita.what_if import what_if

__all__ = [
    "CUPAC",
    "CUPED",
    "BayesianExperiment",
    "Experiment",
    "GroupSequential",
    "InsufficientDataError",
    "LinTS",
    "MultipleCorrection",
    "NotFittedError",
    "OutlierHandler",
    "SRMCheck",
    "SampleSize",
    "SplitaError",
    "ThompsonSampler",
    "ValidationError",
    "audit_trail",
    "auto",
    "check",
    "compare",
    "diagnose",
    "explain",
    "mSPRT",
    "meta_analysis",
    "monitor",
    "power_report",
    "recommend",
    "report",
    "simulate",
    "verbose",
    "what_if",
]
