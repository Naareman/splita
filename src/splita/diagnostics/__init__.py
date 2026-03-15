"""Experiment diagnostics.

Novelty detection, A/A tests, effect time series, pre-experiment validation,
flicker detection.
"""

from splita.diagnostics.aa_test import AATest
from splita.diagnostics.carryover import CarryoverDetector
from splita.diagnostics.effect_timeseries import EffectTimeSeries
from splita.diagnostics.flicker import FlickerDetector
from splita.diagnostics.nonstationarity import NonStationaryDetector
from splita.diagnostics.novelty import NoveltyCurve
from splita.diagnostics.phacking import PHackingDetector
from splita.diagnostics.pre_experiment import MetricSensitivity, VarianceEstimator
from splita.diagnostics.randomization import RandomizationValidator

__all__ = [
    "AATest",
    "CarryoverDetector",
    "EffectTimeSeries",
    "FlickerDetector",
    "MetricSensitivity",
    "NonStationaryDetector",
    "NoveltyCurve",
    "PHackingDetector",
    "RandomizationValidator",
    "VarianceEstimator",
]
