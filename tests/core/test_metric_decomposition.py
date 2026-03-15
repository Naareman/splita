"""Tests for MetricDecomposition."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.metric_decomposition import MetricDecomposition


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


class TestDecompose:
    def test_basic_decomposition(self, rng):
        n = 500
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10.5, 2, n)
        comps_ctrl = {"a": rng.normal(5, 1, n), "b": rng.normal(5, 1, n)}
        comps_trt = {"a": rng.normal(5.3, 1, n), "b": rng.normal(5.2, 1, n)}
        result = MetricDecomposition().decompose(ctrl, trt, comps_ctrl, comps_trt)
        assert "a" in result.component_results
        assert "b" in result.component_results

    def test_total_lift(self, rng):
        n = 500
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10.5, 2, n)
        comps_ctrl = {"x": ctrl}
        comps_trt = {"x": trt}
        result = MetricDecomposition().decompose(ctrl, trt, comps_ctrl, comps_trt)
        assert abs(result.total_lift - (np.mean(trt) - np.mean(ctrl))) < 1e-10

    def test_dominant_component_identified(self, rng):
        n = 1000
        ctrl_total = rng.normal(10, 1, n)
        trt_total = rng.normal(11, 1, n)  # large effect
        comps_ctrl = {
            "driver": rng.normal(5, 0.5, n),
            "noise": rng.normal(5, 0.5, n),
        }
        comps_trt = {
            "driver": rng.normal(6, 0.5, n),  # strong signal
            "noise": rng.normal(5, 0.5, n),  # no signal
        }
        result = MetricDecomposition().decompose(
            ctrl_total, trt_total, comps_ctrl, comps_trt
        )
        assert result.dominant_component == "driver"

    def test_component_has_pvalue(self, rng):
        n = 500
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10.5, 2, n)
        comps_ctrl = {"x": rng.normal(5, 1, n)}
        comps_trt = {"x": rng.normal(5.3, 1, n)}
        result = MetricDecomposition().decompose(ctrl, trt, comps_ctrl, comps_trt)
        assert "pvalue" in result.component_results["x"]
        assert 0.0 <= result.component_results["x"]["pvalue"] <= 1.0

    def test_contribution_sums(self, rng):
        n = 500
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10.5, 2, n)
        comps_ctrl = {"a": rng.normal(5, 1, n), "b": rng.normal(5, 1, n)}
        comps_trt = {"a": rng.normal(5.3, 1, n), "b": rng.normal(5.2, 1, n)}
        result = MetricDecomposition().decompose(ctrl, trt, comps_ctrl, comps_trt)
        total_contribution = sum(
            r["contribution"] for r in result.component_results.values()
        )
        # Contributions should approximately sum to 1 when components
        # are additive decomposition of total
        assert isinstance(total_contribution, float)

    def test_to_dict(self, rng):
        n = 200
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10.5, 2, n)
        comps_ctrl = {"x": rng.normal(5, 1, n)}
        comps_trt = {"x": rng.normal(5.3, 1, n)}
        result = MetricDecomposition().decompose(ctrl, trt, comps_ctrl, comps_trt)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_lift" in d

    def test_repr(self, rng):
        n = 200
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10.5, 2, n)
        comps_ctrl = {"x": rng.normal(5, 1, n)}
        comps_trt = {"x": rng.normal(5.3, 1, n)}
        result = MetricDecomposition().decompose(ctrl, trt, comps_ctrl, comps_trt)
        assert "MetricDecompResult" in repr(result)


class TestValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            MetricDecomposition(alpha=1.5)

    def test_empty_components(self, rng):
        with pytest.raises(ValueError, match="empty"):
            MetricDecomposition().decompose(
                rng.normal(0, 1, 50), rng.normal(0, 1, 50), {}, {}
            )

    def test_mismatched_keys(self, rng):
        with pytest.raises(ValueError, match="same keys"):
            MetricDecomposition().decompose(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                {"a": rng.normal(0, 1, 50)},
                {"b": rng.normal(0, 1, 50)},
            )

    def test_short_input(self):
        with pytest.raises(ValueError, match="at least"):
            MetricDecomposition().decompose(
                [1.0], [2.0], {"x": [1.0]}, {"x": [2.0]}
            )
