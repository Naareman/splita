"""Experiment diagnostics.

Novelty detection, A/A tests, effect time series, pre-experiment validation.
"""

from splita.diagnostics.aa_test import AATest
from splita.diagnostics.effect_timeseries import EffectTimeSeries
from splita.diagnostics.nonstationarity import NonStationaryDetector
from splita.diagnostics.novelty import NoveltyCurve
from splita.diagnostics.pre_experiment import MetricSensitivity, VarianceEstimator

__all__ = [
    "AATest",
    "EffectTimeSeries",
    "MetricSensitivity",
    "NonStationaryDetector",
    "NoveltyCurve",
    "VarianceEstimator",
]
