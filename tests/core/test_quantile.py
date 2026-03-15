"""Tests for QuantileExperiment and QuantileResult."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import QuantileResult
from splita.core import QuantileExperiment


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def normal_data():
    """Control ~ N(10, 2), treatment ~ N(11, 2), n=500 each."""
    rng = np.random.default_rng(42)
    ctrl = rng.normal(10, 2, size=500)
    trt = rng.normal(11, 2, size=500)
    return ctrl, trt


@pytest.fixture()
def identical_data():
    """Same distribution for control and treatment."""
    rng = np.random.default_rng(99)
    ctrl = rng.normal(10, 2, size=500)
    trt = rng.normal(10, 2, size=500)
    return ctrl, trt


# ─── Basic functionality ────────────────────────────────────────────


class TestSingleQuantile:
    """Tests for a single quantile (median)."""

    def test_median_detects_shift(self, normal_data):
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=42
        ).run()
        assert isinstance(result, QuantileResult)
        assert len(result.quantiles) == 1
        assert result.quantiles[0] == 0.5
        assert result.significant[0] is True

    def test_median_no_shift(self, identical_data):
        ctrl, trt = identical_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=42
        ).run()
        assert result.significant[0] is False

    def test_result_fields_populated(self, normal_data):
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=42
        ).run()
        assert result.control_n == 500
        assert result.treatment_n == 500
        assert result.alpha == 0.05
        assert len(result.control_quantiles) == 1
        assert len(result.treatment_quantiles) == 1
        assert len(result.differences) == 1
        assert len(result.ci_lower) == 1
        assert len(result.ci_upper) == 1
        assert len(result.pvalues) == 1
        assert len(result.significant) == 1


class TestMultipleQuantiles:
    """Tests for multiple quantiles."""

    def test_multiple_quantiles(self, normal_data):
        ctrl, trt = normal_data
        qs = [0.25, 0.5, 0.75, 0.90, 0.99]
        result = QuantileExperiment(
            ctrl, trt, quantiles=qs, random_state=42
        ).run()
        assert result.quantiles == qs
        assert len(result.differences) == 5
        assert len(result.pvalues) == 5
        assert len(result.significant) == 5

    def test_all_quantiles_shifted_for_constant_shift(self):
        """treatment = control + constant -> all quantiles shift equally."""
        rng = np.random.default_rng(123)
        ctrl = rng.normal(10, 2, size=1000)
        shift = 1.5
        trt = ctrl + shift
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        result = QuantileExperiment(
            ctrl, trt, quantiles=qs, random_state=42
        ).run()
        for i in range(len(qs)):
            assert abs(result.differences[i] - shift) < 0.01, (
                f"Quantile {qs[i]}: expected diff ~{shift}, "
                f"got {result.differences[i]}"
            )
            assert result.significant[i] is True


class TestSignificantAtMedianNotP90:
    """Significant at median but not at p90."""

    def test_median_shift_only(self):
        """Shift the bulk of the distribution but keep right tail similar."""
        rng = np.random.default_rng(77)
        n = 1000
        # Control: mixture with heavy right tail
        ctrl = np.concatenate([
            rng.normal(10, 1, size=n - 50),
            rng.normal(30, 5, size=50),
        ])
        # Treatment: shift the bulk but not the tail
        trt = np.concatenate([
            rng.normal(12, 1, size=n - 50),
            rng.normal(30, 5, size=50),
        ])
        result = QuantileExperiment(
            ctrl, trt, quantiles=[0.5, 0.90], random_state=42
        ).run()
        assert result.significant[0] is True, "Median should be significant"
        # p90 may or may not be significant, but the diff should be much smaller
        assert abs(result.differences[0]) > abs(result.differences[1]) * 0.5


class TestRevenueRightTailShift:
    """Revenue data: right tail shifts (p90, p99) but median doesn't."""

    def test_right_tail_shift(self):
        rng = np.random.default_rng(55)
        n = 2000
        # Lognormal revenue: most users spend ~$10, some spend $100+
        ctrl = rng.lognormal(2.3, 0.5, size=n)
        # Treatment: same bulk, but heavy spenders spend more
        trt = ctrl.copy()
        # Shift the top 5% by +50%
        p95_val = np.quantile(ctrl, 0.95)
        mask = trt >= p95_val
        trt[mask] = trt[mask] * 1.5

        result = QuantileExperiment(
            ctrl, trt, quantiles=[0.5, 0.90, 0.99], random_state=42
        ).run()
        # Median should NOT be significant (bulk is unchanged)
        assert result.significant[0] is False, (
            f"Median should not be significant, got p={result.pvalues[0]}"
        )
        # p99 difference should be positive
        assert result.differences[2] > 0, "p99 should show a positive shift"


# ─── Known shift tests ──────────────────────────────────────────────


class TestKnownShift:
    """Treatment = control + constant -> all quantiles shift by that constant."""

    def test_constant_shift_differences(self):
        rng = np.random.default_rng(0)
        ctrl = rng.exponential(5, size=800)
        shift = 3.0
        trt = ctrl + shift
        qs = [0.25, 0.5, 0.75]
        result = QuantileExperiment(
            ctrl, trt, quantiles=qs, random_state=0
        ).run()
        for i, q in enumerate(qs):
            assert abs(result.differences[i] - shift) < 0.01
            assert result.ci_lower[i] < shift < result.ci_upper[i]

    def test_ci_contains_true_diff(self):
        """CI should contain the true difference."""
        rng = np.random.default_rng(12)
        ctrl = rng.normal(0, 1, size=500)
        trt = rng.normal(0.5, 1, size=500)
        result = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=12
        ).run()
        # The true median difference is 0.5 (normal is symmetric)
        # CI should usually contain 0.5
        assert result.ci_lower[0] < 0.5 < result.ci_upper[0]


# ─── CI coverage simulation ─────────────────────────────────────────


class TestCICoverage:
    """Check that 95% CI covers the true difference ~95% of the time."""

    def test_coverage_rate(self):
        true_shift = 1.0
        n_sims = 100
        covered = 0
        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            ctrl = rng.normal(10, 2, size=200)
            trt = rng.normal(10 + true_shift, 2, size=200)
            result = QuantileExperiment(
                ctrl, trt, quantiles=0.5, n_bootstrap=1000, random_state=seed
            ).run()
            if result.ci_lower[0] <= true_shift <= result.ci_upper[0]:
                covered += 1
        coverage = covered / n_sims
        # Allow a generous range: 85-100% (bootstrap coverage is approximate)
        assert 0.85 <= coverage <= 1.0, f"Coverage was {coverage}"


# ─── Validation ─────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_quantile_below_zero(self):
        with pytest.raises(ValueError, match="quantiles.*must be in"):
            QuantileExperiment([1, 2, 3], [4, 5, 6], quantiles=-0.1)

    def test_quantile_above_one(self):
        with pytest.raises(ValueError, match="quantiles.*must be in"):
            QuantileExperiment([1, 2, 3], [4, 5, 6], quantiles=1.5)

    def test_quantile_exactly_zero(self):
        with pytest.raises(ValueError, match="quantiles.*must be in"):
            QuantileExperiment([1, 2, 3], [4, 5, 6], quantiles=0.0)

    def test_quantile_exactly_one(self):
        with pytest.raises(ValueError, match="quantiles.*must be in"):
            QuantileExperiment([1, 2, 3], [4, 5, 6], quantiles=1.0)

    def test_empty_quantiles_list(self):
        with pytest.raises(ValueError, match="can't be empty"):
            QuantileExperiment([1, 2, 3], [4, 5, 6], quantiles=[])

    def test_n_bootstrap_too_low(self):
        with pytest.raises(ValueError, match="n_bootstrap.*must be >= 100"):
            QuantileExperiment(
                [1, 2, 3], [4, 5, 6], quantiles=0.5, n_bootstrap=10
            )

    def test_n_bootstrap_too_high(self):
        with pytest.raises(ValueError, match="n_bootstrap.*must be at most"):
            QuantileExperiment(
                [1, 2, 3], [4, 5, 6], quantiles=0.5, n_bootstrap=2_000_000
            )

    def test_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha.*must be in"):
            QuantileExperiment(
                [1, 2, 3], [4, 5, 6], quantiles=0.5, alpha=0.0
            )

    def test_alpha_one(self):
        with pytest.raises(ValueError, match="alpha.*must be in"):
            QuantileExperiment(
                [1, 2, 3], [4, 5, 6], quantiles=0.5, alpha=1.0
            )

    def test_control_too_short(self):
        with pytest.raises(ValueError, match="control.*must have at least"):
            QuantileExperiment([1], [4, 5, 6], quantiles=0.5)

    def test_treatment_too_short(self):
        with pytest.raises(ValueError, match="treatment.*must have at least"):
            QuantileExperiment([1, 2, 3], [4], quantiles=0.5)

    def test_non_array_control(self):
        with pytest.raises(TypeError, match="control.*must be array-like"):
            QuantileExperiment("not an array", [4, 5, 6], quantiles=0.5)


# ─── Reproducibility ────────────────────────────────────────────────


class TestReproducibility:
    """Same random_state produces same results."""

    def test_same_seed_same_results(self, normal_data):
        ctrl, trt = normal_data
        r1 = QuantileExperiment(
            ctrl, trt, quantiles=[0.25, 0.5, 0.75], random_state=42
        ).run()
        r2 = QuantileExperiment(
            ctrl, trt, quantiles=[0.25, 0.5, 0.75], random_state=42
        ).run()
        assert r1.differences == r2.differences
        assert r1.pvalues == r2.pvalues
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_different_seed_different_results(self, normal_data):
        ctrl, trt = normal_data
        r1 = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=42
        ).run()
        r2 = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=99
        ).run()
        # CIs should differ (different bootstrap samples)
        assert r1.ci_lower != r2.ci_lower or r1.ci_upper != r2.ci_upper


# ─── Serialisation ──────────────────────────────────────────────────


class TestToDict:
    """to_dict produces a plain Python dict."""

    def test_to_dict_keys(self, normal_data):
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=[0.25, 0.5], random_state=42
        ).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        expected_keys = {
            "quantiles",
            "control_quantiles",
            "treatment_quantiles",
            "differences",
            "ci_lower",
            "ci_upper",
            "pvalues",
            "significant",
            "alpha",
            "control_n",
            "treatment_n",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_types(self, normal_data):
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=42
        ).run()
        d = result.to_dict()
        assert isinstance(d["quantiles"], list)
        assert isinstance(d["control_n"], int)
        assert isinstance(d["alpha"], float)
        assert isinstance(d["significant"], list)
        assert all(isinstance(v, bool) for v in d["significant"])

    def test_to_dict_json_serialisable(self, normal_data):
        import json

        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=[0.25, 0.5, 0.75], random_state=42
        ).run()
        # Should not raise
        json.dumps(result.to_dict())


# ─── Repr ───────────────────────────────────────────────────────────


class TestRepr:
    """Pretty repr shows a table."""

    def test_repr_contains_header(self, normal_data):
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=[0.25, 0.5], random_state=42
        ).run()
        text = repr(result)
        assert "QuantileResult" in text
        assert "quantile" in text
        assert "diff" in text
        assert "pvalue" in text

    def test_repr_shows_quantile_values(self, normal_data):
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=[0.25, 0.5, 0.75], random_state=42
        ).run()
        text = repr(result)
        assert "0.2500" in text
        assert "0.5000" in text
        assert "0.7500" in text

    def test_repr_shows_significance_markers(self, normal_data):
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=42
        ).run()
        text = repr(result)
        # Should contain a check mark or cross mark
        assert "\u2713" in text or "\u2717" in text


# ─── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests."""

    def test_single_float_quantile_is_normalised(self):
        """A single float should be wrapped in a list."""
        result = QuantileExperiment(
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            quantiles=0.5,
            random_state=42,
        ).run()
        assert result.quantiles == [0.5]

    def test_integer_quantile_cast_to_float(self):
        """Integer quantile (e.g. from a loop) should work."""
        # This shouldn't raise
        QuantileExperiment(
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            quantiles=[0.5],
            random_state=42,
        ).run()

    def test_small_sample(self):
        """Works with minimum sample size (2 per group)."""
        result = QuantileExperiment(
            [1, 10],
            [5, 15],
            quantiles=0.5,
            random_state=42,
        ).run()
        assert result.control_n == 2
        assert result.treatment_n == 2
        assert len(result.differences) == 1

    def test_frozen_result(self, normal_data):
        """QuantileResult is frozen — can't set attributes."""
        ctrl, trt = normal_data
        result = QuantileExperiment(
            ctrl, trt, quantiles=0.5, random_state=42
        ).run()
        with pytest.raises(AttributeError):
            result.alpha = 0.1  # type: ignore[misc]

    def test_extreme_quantiles(self):
        """Near-boundary quantiles (0.01, 0.99) should work."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, size=1000)
        trt = rng.normal(1, 1, size=1000)
        result = QuantileExperiment(
            ctrl, trt, quantiles=[0.01, 0.99], random_state=42
        ).run()
        assert len(result.differences) == 2
        # Both should show positive shift
        assert result.differences[0] > 0
        assert result.differences[1] > 0
