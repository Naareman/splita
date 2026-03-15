"""Tests for OECBuilder."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import OECResult
from splita.core.oec import OECBuilder


# ── helpers ──────────────────────────────────────────────────────────


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic OECBuilder behaviour."""

    def test_detects_positive_effect(self):
        """Detects a clear positive effect across metrics."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("rev", rng.normal(10, 2, 500), rng.normal(11, 2, 500))
        builder.add_metric("clicks", rng.normal(5, 1, 500), rng.normal(5.5, 1, 500))
        result = builder.run()
        assert isinstance(result, OECResult)
        assert result.significant
        assert result.oec_lift > 0

    def test_no_effect(self):
        """Zero effect is not significant."""
        rng = _rng(123)
        builder = OECBuilder()
        builder.add_metric("rev", rng.normal(10, 2, 1000), rng.normal(10, 2, 1000))
        builder.add_metric("clicks", rng.normal(5, 1, 1000), rng.normal(5, 1, 1000))
        result = builder.run()
        assert not result.significant

    def test_result_frozen(self):
        """Result is a frozen dataclass."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("m", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        result = builder.run()
        with pytest.raises(AttributeError):
            result.oec_lift = 999.0  # type: ignore[misc]

    def test_to_dict(self):
        """Result serializes to dict."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("m", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        result = builder.run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "oec_lift" in d
        assert "weights" in d

    def test_repr(self):
        """Result has string representation."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("m", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        result = builder.run()
        s = repr(result)
        assert "OECResult" in s


# ── Weights ──────────────────────────────────────────────────────────


class TestWeights:
    """Weight normalisation and contributions."""

    def test_weights_sum_to_one(self):
        """Normalised weights sum to 1."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("a", rng.normal(0, 1, 100), rng.normal(0, 1, 100), weight=3.0)
        builder.add_metric("b", rng.normal(0, 1, 100), rng.normal(0, 1, 100), weight=1.0)
        result = builder.run()
        np.testing.assert_allclose(sum(result.weights), 1.0, atol=1e-10)

    def test_equal_weights(self):
        """Equal raw weights produce equal normalised weights."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("a", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        builder.add_metric("b", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        result = builder.run()
        np.testing.assert_allclose(result.weights[0], result.weights[1], atol=1e-10)

    def test_contributions_count_matches_metrics(self):
        """Number of contributions equals number of metrics."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("a", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        builder.add_metric("b", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        builder.add_metric("c", rng.normal(0, 1, 100), rng.normal(0, 1, 100))
        result = builder.run()
        assert len(result.metric_contributions) == 3


# ── Direction flipping ───────────────────────────────────────────────


class TestDirection:
    """Lower-is-better direction flipping."""

    def test_lower_is_better_flips_sign(self):
        """Lower-is-better metric with decrease => positive contribution."""
        rng = _rng()
        builder = OECBuilder()
        # Latency decreased (good)
        builder.add_metric(
            "latency",
            rng.normal(200, 10, 500),
            rng.normal(180, 10, 500),
            direction="lower_is_better",
        )
        result = builder.run()
        assert result.oec_lift > 0
        assert result.significant


# ── Normalization ────────────────────────────────────────────────────


class TestNormalization:
    """Tests for normalize=True vs False."""

    def test_no_normalize(self):
        """Works without normalisation."""
        rng = _rng()
        builder = OECBuilder(normalize=False)
        builder.add_metric("rev", rng.normal(10, 2, 500), rng.normal(11, 2, 500))
        result = builder.run()
        assert isinstance(result, OECResult)

    def test_normalize_different_scales(self):
        """Normalisation helps with different-scale metrics."""
        rng = _rng()
        builder = OECBuilder(normalize=True)
        builder.add_metric("rev", rng.normal(1000, 100, 500), rng.normal(1050, 100, 500))
        builder.add_metric("ctr", rng.normal(0.05, 0.01, 500), rng.normal(0.055, 0.01, 500))
        result = builder.run()
        assert isinstance(result, OECResult)


# ── CI coverage ──────────────────────────────────────────────────────


class TestCI:
    """Confidence interval tests."""

    def test_ci_contains_lift(self):
        """CI contains the observed OEC lift."""
        rng = _rng()
        builder = OECBuilder()
        builder.add_metric("m", rng.normal(10, 2, 500), rng.normal(10, 2, 500))
        result = builder.run()
        assert result.ci_lower <= result.oec_lift <= result.ci_upper


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_no_metrics_raises(self):
        """run() without metrics raises ValueError."""
        with pytest.raises(ValueError, match="No metrics"):
            OECBuilder().run()

    def test_empty_name_raises(self):
        """Empty metric name raises ValueError."""
        rng = _rng()
        with pytest.raises(ValueError, match="non-empty"):
            OECBuilder().add_metric("", rng.normal(0, 1, 10), rng.normal(0, 1, 10))

    def test_negative_weight_raises(self):
        """Negative weight raises ValueError."""
        rng = _rng()
        with pytest.raises(ValueError, match="weight"):
            OECBuilder().add_metric("m", rng.normal(0, 1, 10), rng.normal(0, 1, 10), weight=-1.0)

    def test_invalid_direction_raises(self):
        """Invalid direction raises ValueError."""
        rng = _rng()
        with pytest.raises(ValueError, match="direction"):
            OECBuilder().add_metric(
                "m", rng.normal(0, 1, 10), rng.normal(0, 1, 10), direction="up"
            )

    def test_control_too_short_raises(self):
        """Control with < 2 elements raises ValueError."""
        with pytest.raises(ValueError, match="at least"):
            OECBuilder().add_metric("m", [1.0], [1.0, 2.0])

    def test_normalize_non_bool_raises(self):
        """Non-bool normalize raises TypeError."""
        with pytest.raises(TypeError, match="bool"):
            OECBuilder(normalize="yes")  # type: ignore[arg-type]


# ── E2E scenario ─────────────────────────────────────────────────────


class TestE2E:
    """End-to-end scenario."""

    def test_ecommerce_oec(self):
        """E-commerce OEC: revenue up, latency down, bounce rate down."""
        rng = _rng(99)
        builder = OECBuilder()
        builder.add_metric("revenue", rng.normal(50, 10, 1000), rng.normal(52, 10, 1000), weight=3.0)
        builder.add_metric("latency_ms", rng.normal(300, 50, 1000), rng.normal(280, 50, 1000),
                           weight=1.0, direction="lower_is_better")
        builder.add_metric("bounce_rate", rng.normal(0.4, 0.1, 1000), rng.normal(0.38, 0.1, 1000),
                           weight=2.0, direction="lower_is_better")
        result = builder.run()
        assert result.significant
        assert result.oec_lift > 0
        assert len(result.metric_contributions) == 3
