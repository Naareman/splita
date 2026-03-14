"""Smoke tests — real-world scenarios exercising foundation modules together."""

import math
import warnings

import numpy as np
import pytest

from splita._types import ExperimentResult, SampleSizeResult
from splita._utils import (
    auto_detect_metric,
    cohens_d,
    cohens_h,
    ensure_rng,
    relative_lift,
    to_array,
)
from splita._validation import (
    check_array_like,
    check_in_range,
    check_one_of,
    check_same_length,
)

# ─── Scenario 1: Validate array pipeline ──────────────────────────────


class TestArrayPipeline:
    """Create realistic A/B test data and pipe through array utilities."""

    def test_conversion_data_pipeline(self):
        rng = np.random.default_rng(101)
        # 1000 users per group, ~10% conversion rate
        control = rng.binomial(1, 0.10, size=1000).astype(float)
        treatment = rng.binomial(1, 0.10, size=1000).astype(float)

        # Inject a few NaN values
        control[50] = float("nan")
        control[200] = float("nan")
        treatment[999] = float("nan")

        # to_array preserves NaN (it does not clean)
        ctrl_arr = to_array(control.tolist(), "control")
        assert ctrl_arr.dtype == np.float64, "to_array should produce float64"

        # check_array_like cleans NaN
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctrl_clean = check_array_like(ctrl_arr.tolist(), "control")
        assert len(w) == 1, "should warn about NaN removal"
        assert "NaN" in str(w[0].message), "warning should mention NaN"
        assert not np.any(np.isnan(ctrl_clean)), "NaN should be removed"
        assert len(ctrl_clean) == 998, "2 NaN removed from 1000 -> 998"

        # auto_detect_metric on binary data
        metric = auto_detect_metric(ctrl_clean)
        assert metric == "conversion", (
            f"binary data should be 'conversion', got {metric!r}"
        )

    def test_continuous_data_pipeline(self):
        rng = np.random.default_rng(202)
        # Revenue data: mean ~$25, std ~$40 (right-skewed as revenue often is)
        revenue = rng.exponential(scale=25.0, size=1000)

        arr = to_array(revenue.tolist(), "revenue")
        cleaned = check_array_like(arr.tolist(), "revenue")
        metric = auto_detect_metric(cleaned)
        assert metric == "continuous", (
            f"revenue data should be 'continuous', got {metric!r}"
        )
        assert len(cleaned) == 1000, "no NaN injected, length should stay 1000"


# ─── Scenario 2: Effect size calculations on realistic data ───────────


class TestEffectSizeRealistic:
    """Verify effect size functions produce sensible values for typical A/B tests."""

    def test_cohens_h_small_conversion_lift(self):
        # Control 10%, treatment 12% — a realistic 20% relative lift
        h = cohens_h(0.10, 0.12)
        assert h > 0, "treatment > control should yield positive Cohen's h"
        assert 0.04 < h < 0.08, f"expected small h around 0.06, got {h:.4f}"

    def test_cohens_d_small_revenue_lift(self):
        rng = np.random.default_rng(303)
        control = rng.normal(loc=25.0, scale=40.0, size=5000)
        treatment = rng.normal(loc=27.0, scale=40.0, size=5000)

        d = cohens_d(control, treatment)
        assert d > 0, "treatment mean > control mean should yield positive Cohen's d"
        assert 0.02 < d < 0.10, f"expected small d around 0.05, got {d:.4f}"

    def test_relative_lift_percentage(self):
        lift = relative_lift(25.0, 27.0)
        assert abs(lift - 0.08) < 0.001, f"expected ~8% relative lift, got {lift:.4f}"


# ─── Scenario 3: Validation catches bad inputs ────────────────────────


class TestValidationCatchesBadInputs:
    """Verify validators produce informative errors for common mistakes."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match=r"alpha.*must be in"):
            check_in_range(1.5, "alpha", 0.0, 1.0)

    def test_alpha_nan_rejected(self):
        with pytest.raises(ValueError, match=r"finite number"):
            check_in_range(float("nan"), "alpha", 0.0, 1.0)

    def test_check_one_of_suggests_alternative(self):
        with pytest.raises(ValueError, match=r"did you mean") as exc_info:
            check_one_of("bayesian", "method", ["frequentist", "bayesian_bootstrap"])
        # The suggestion should point to the closest match
        assert "bayesian_bootstrap" in str(exc_info.value), (
            "should suggest 'bayesian_bootstrap' for 'bayesian'"
        )

    def test_check_same_length_mismatched(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"same length"):
            check_same_length(a, b, "control", "treatment")

    def test_check_array_like_handles_list_input(self):
        # Lists should be handled gracefully (common user input)
        result = check_array_like([1.0, 2.0, 3.0], "data")
        assert isinstance(result, np.ndarray), "list should be converted to ndarray"
        assert len(result) == 3, "length should be preserved"


# ─── Scenario 4: Dataclass serialization roundtrip ────────────────────


class TestDataclassSerialization:
    """Verify result types serialize numpy values and produce readable output."""

    def test_experiment_result_to_dict_no_numpy(self):
        result = ExperimentResult(
            control_mean=np.float64(0.10),
            treatment_mean=np.float64(0.12),
            lift=np.float64(0.02),
            relative_lift=np.float64(0.20),
            pvalue=np.float64(0.03),
            statistic=np.float64(2.17),
            ci_lower=np.float64(0.001),
            ci_upper=np.float64(0.039),
            significant=True,
            alpha=0.05,
            method="z_test",
            metric="conversion",
            control_n=np.int64(5000),
            treatment_n=np.int64(5000),
            power=np.float64(0.82),
            effect_size=np.float64(0.063),
        )

        d = result.to_dict()

        # Every value should be a plain Python type, not numpy
        for key, val in d.items():
            assert not isinstance(val, (np.integer, np.floating, np.bool_)), (
                f"to_dict()[{key!r}] is {type(val).__name__}, "
                "expected plain Python type"
            )

        # Spot-check key values
        assert d["control_n"] == 5000, "control_n should roundtrip as int"
        assert isinstance(d["control_n"], int), "control_n should be plain int"
        assert abs(d["pvalue"] - 0.03) < 1e-10, "pvalue should roundtrip correctly"

    def test_experiment_result_repr_readable(self):
        result = ExperimentResult(
            control_mean=0.10,
            treatment_mean=0.12,
            lift=0.02,
            relative_lift=0.20,
            pvalue=0.03,
            statistic=2.17,
            ci_lower=0.001,
            ci_upper=0.039,
            significant=True,
            alpha=0.05,
            method="z_test",
            metric="conversion",
            control_n=5000,
            treatment_n=5000,
            power=0.82,
            effect_size=0.063,
        )
        text = repr(result)
        assert "ExperimentResult" in text, "repr should start with class name"
        assert "z_test" in text, "repr should include method"
        assert "conversion" in text, "repr should include metric"

    def test_sample_size_result_duration_chain(self):
        ssr = SampleSizeResult(
            n_per_variant=5000,
            n_total=10000,
            alpha=0.05,
            power=0.80,
            mde=0.02,
            relative_mde=0.20,
            baseline=0.10,
            metric="conversion",
            effect_size=0.063,
            days_needed=None,
        )
        assert ssr.days_needed is None, "days_needed should be None before .duration()"

        with_days = ssr.duration(daily_users=1000)
        assert with_days.days_needed is not None, (
            "days_needed should be populated after .duration()"
        )
        assert with_days.days_needed == 10, (
            f"10000 total / 1000 daily = 10 days, got {with_days.days_needed}"
        )
        # Original should be unmodified (frozen)
        assert ssr.days_needed is None, "original SampleSizeResult should be unchanged"


# ─── Scenario 5: RNG reproducibility ──────────────────────────────────


class TestRNGReproducibility:
    """Verify RNG seeding gives deterministic and independent streams."""

    def test_same_seed_same_output(self):
        rng1 = ensure_rng(42)
        rng2 = ensure_rng(42)
        val1 = rng1.random()
        val2 = rng2.random()
        assert val1 == val2, (
            f"same seed should produce identical first value: {val1} != {val2}"
        )

    def test_none_gives_different_output(self):
        rng1 = ensure_rng(None)
        rng2 = ensure_rng(None)
        # Draw several values to reduce the (astronomically small) chance of collision
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]
        assert vals1 != vals2, "None seed should produce different streams (OS entropy)"

    def test_passthrough_generator(self):
        gen = np.random.default_rng(99)
        result = ensure_rng(gen)
        assert result is gen, "passing a Generator should return the same object"


# ─── Scenario 6: Edge cases that should work ──────────────────────────


class TestEdgeCases:
    """Verify boundary conditions and extreme values are handled correctly."""

    def test_very_small_conversion_rate(self):
        # 0.1% baseline — common in e-commerce purchase funnels
        h = cohens_h(0.001, 0.002)
        assert h > 0, "doubled conversion (0.1% -> 0.2%) should be positive h"
        assert math.isfinite(h), "result should be finite"

    def test_very_large_sample_sizes(self):
        # 1M per group — large-scale experiment
        rng = np.random.default_rng(404)
        data = rng.binomial(1, 0.05, size=1_000_000)
        arr = to_array(data.tolist(), "big_data")
        assert len(arr) == 1_000_000, "should handle 1M elements"
        metric = auto_detect_metric(arr)
        assert metric == "conversion", "binary 1M-element array should be 'conversion'"

    def test_cohens_h_boundary_zero(self):
        h = cohens_h(0.0, 0.0)
        assert h == 0.0, f"h(0,0) should be exactly 0, got {h}"

    def test_cohens_h_boundary_one(self):
        h = cohens_h(1.0, 1.0)
        assert h == 0.0, f"h(1,1) should be exactly 0, got {h}"

    def test_cohens_h_full_range(self):
        h = cohens_h(0.0, 1.0)
        assert h > 0, "h(0, 1) should be positive"
        assert math.isfinite(h), "h(0, 1) should be finite"

    def test_relative_lift_negative_baseline(self):
        # Negative baseline (e.g., P&L metric)
        lift = relative_lift(-100.0, -80.0)
        # (-80 - -100) / |-100| = 20/100 = 0.2
        assert abs(lift - 0.20) < 0.001, (
            f"lift from -100 to -80 should be +0.20, got {lift:.4f}"
        )

    def test_duration_very_small_traffic_fraction(self):
        ssr = SampleSizeResult(
            n_per_variant=10000,
            n_total=20000,
            alpha=0.05,
            power=0.80,
            mde=0.02,
            relative_mde=0.20,
            baseline=0.10,
            metric="conversion",
            effect_size=0.063,
            days_needed=None,
        )
        # Only 1% of traffic allocated
        with_days = ssr.duration(daily_users=10000, traffic_fraction=0.01)
        # 20000 / (10000 * 0.01) = 200 days
        assert with_days.days_needed == 200, (
            f"expected 200 days at 1% traffic, got {with_days.days_needed}"
        )
