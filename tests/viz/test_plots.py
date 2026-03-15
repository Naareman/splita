"""Tests for splita.viz.plots — verify each function runs and returns a Figure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Non-interactive backend for CI

from matplotlib.figure import Figure  # noqa: E402

from splita.viz.plots import (  # noqa: E402
    effect_over_time,
    forest_plot,
    funnel_chart,
    metric_comparison,
    power_curve,
)


# ---------------------------------------------------------------------------
# Lightweight stubs — avoids coupling tests to real Experiment pipelines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeExperimentResult:
    lift: float
    ci_lower: float
    ci_upper: float
    significant: bool


@dataclass(frozen=True)
class _FakeEffectTimeSeriesResult:
    time_points: list
    is_stable: bool
    final_lift: float
    final_pvalue: float


@dataclass(frozen=True)
class _FakeFunnelResult:
    step_results: list
    bottleneck_step: str
    overall_lift: float


# ---------------------------------------------------------------------------
# forest_plot
# ---------------------------------------------------------------------------


class TestForestPlot:
    def test_returns_figure(self):
        results = [
            _FakeExperimentResult(lift=0.05, ci_lower=0.01, ci_upper=0.09, significant=True),
            _FakeExperimentResult(lift=-0.02, ci_lower=-0.06, ci_upper=0.02, significant=False),
        ]
        fig = forest_plot(results)
        assert isinstance(fig, Figure)

    def test_custom_labels(self):
        results = [
            _FakeExperimentResult(lift=0.03, ci_lower=0.00, ci_upper=0.06, significant=True),
        ]
        fig = forest_plot(results, labels=["CTR"])
        assert isinstance(fig, Figure)

    def test_with_provided_ax(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        results = [
            _FakeExperimentResult(lift=0.1, ci_lower=0.02, ci_upper=0.18, significant=True),
        ]
        fig = forest_plot(results, ax=ax)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# effect_over_time
# ---------------------------------------------------------------------------


class TestEffectOverTime:
    def _make_ts(self, n=5):
        return _FakeEffectTimeSeriesResult(
            time_points=[
                {
                    "timestamp": i,
                    "cumulative_lift": 0.01 * i,
                    "pvalue": 0.05,
                    "ci_lower": 0.01 * i - 0.02,
                    "ci_upper": 0.01 * i + 0.02,
                    "n_control": 100 * (i + 1),
                    "n_treatment": 100 * (i + 1),
                }
                for i in range(n)
            ],
            is_stable=True,
            final_lift=0.04,
            final_pvalue=0.03,
        )

    def test_returns_figure(self):
        fig = effect_over_time(self._make_ts())
        assert isinstance(fig, Figure)

    def test_single_point(self):
        fig = effect_over_time(self._make_ts(n=1))
        assert isinstance(fig, Figure)

    def test_with_provided_ax(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        fig = effect_over_time(self._make_ts(), ax=ax)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# power_curve
# ---------------------------------------------------------------------------


class TestPowerCurve:
    def test_returns_figure(self):
        fig = power_curve(
            baseline=0.10,
            mde_range=np.linspace(0.01, 0.10, 10),
            n_per_variant=1000,
        )
        assert isinstance(fig, Figure)

    def test_small_mde(self):
        fig = power_curve(
            baseline=0.05,
            mde_range=[0.001, 0.005, 0.01],
            n_per_variant=5000,
            alpha=0.01,
        )
        assert isinstance(fig, Figure)

    def test_with_provided_ax(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        fig = power_curve(
            baseline=0.10,
            mde_range=[0.01, 0.02, 0.03],
            n_per_variant=2000,
            ax=ax,
        )
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# funnel_chart
# ---------------------------------------------------------------------------


class TestFunnelChart:
    def _make_funnel(self):
        return _FakeFunnelResult(
            step_results=[
                {"name": "Visit", "control_rate": 1.0, "treatment_rate": 1.0},
                {"name": "Signup", "control_rate": 0.40, "treatment_rate": 0.45},
                {"name": "Purchase", "control_rate": 0.10, "treatment_rate": 0.13},
            ],
            bottleneck_step="Signup",
            overall_lift=0.03,
        )

    def test_returns_figure(self):
        fig = funnel_chart(self._make_funnel())
        assert isinstance(fig, Figure)

    def test_single_step(self):
        fr = _FakeFunnelResult(
            step_results=[{"name": "Convert", "control_rate": 0.5, "treatment_rate": 0.55}],
            bottleneck_step="Convert",
            overall_lift=0.05,
        )
        fig = funnel_chart(fr)
        assert isinstance(fig, Figure)

    def test_with_provided_ax(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        fig = funnel_chart(self._make_funnel(), ax=ax)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# metric_comparison
# ---------------------------------------------------------------------------


class TestMetricComparison:
    def _make_dict(self):
        return {
            "CTR": _FakeExperimentResult(lift=0.02, ci_lower=0.005, ci_upper=0.035, significant=True),
            "Revenue": _FakeExperimentResult(lift=-0.5, ci_lower=-1.2, ci_upper=0.2, significant=False),
            "Sessions": _FakeExperimentResult(lift=1.0, ci_lower=0.1, ci_upper=1.9, significant=True),
        }

    def test_returns_figure(self):
        fig = metric_comparison(self._make_dict())
        assert isinstance(fig, Figure)

    def test_single_metric(self):
        d = {"Bounce": _FakeExperimentResult(lift=-0.01, ci_lower=-0.03, ci_upper=0.01, significant=False)}
        fig = metric_comparison(d)
        assert isinstance(fig, Figure)

    def test_with_provided_ax(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        fig = metric_comparison(self._make_dict(), ax=ax)
        assert isinstance(fig, Figure)
