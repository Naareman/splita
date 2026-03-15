"""Visualization module for splita A/B test results.

Requires matplotlib as an optional dependency.
Install with: ``pip install splita[viz]``
"""

from splita.viz.plots import (
    effect_over_time,
    forest_plot,
    funnel_chart,
    metric_comparison,
    power_curve,
)

__all__ = [
    "effect_over_time",
    "forest_plot",
    "funnel_chart",
    "metric_comparison",
    "power_curve",
]
