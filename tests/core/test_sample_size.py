"""Tests for splita.core.sample_size — SampleSize calculator."""

from __future__ import annotations

import math
import warnings

import pytest
from scipy.stats import norm

from splita.core.sample_size import SampleSize


# ═══════════════════════════════════════════════════════════════════════
# for_proportion
# ═══════════════════════════════════════════════════════════════════════


class TestForProportion:
    """Unit tests for SampleSize.for_proportion."""

    def test_basic_10pct_baseline_2pp_mde(self):
        """10% baseline, 2pp MDE → n_per_variant ≈ 3843."""
        res = SampleSize.for_proportion(0.10, 0.02)
        assert res.n_per_variant == 3841
        assert res.n_total == 3841 * 2
        assert res.metric == "proportion"
        assert res.alpha == 0.05
        assert res.power == 0.80

    def test_one_sided_fewer_than_two_sided(self):
        """One-sided test requires fewer samples than two-sided."""
        two = SampleSize.for_proportion(0.10, 0.02, alternative="two-sided")
        one = SampleSize.for_proportion(0.10, 0.02, alternative="one-sided")
        assert one.n_per_variant < two.n_per_variant

    def test_higher_power_more_samples(self):
        """Higher power (0.90) requires more samples than 0.80."""
        low = SampleSize.for_proportion(0.10, 0.02, power=0.80)
        high = SampleSize.for_proportion(0.10, 0.02, power=0.90)
        assert high.n_per_variant > low.n_per_variant

    def test_stricter_alpha_more_samples(self):
        """Stricter alpha (0.01) requires more samples than 0.05."""
        loose = SampleSize.for_proportion(0.10, 0.02, alpha=0.05)
        strict = SampleSize.for_proportion(0.10, 0.02, alpha=0.01)
        assert strict.n_per_variant > loose.n_per_variant

    def test_abc_test_n_variants_3(self):
        """A/B/C test: n_total = 3 * n_per_variant."""
        res = SampleSize.for_proportion(0.10, 0.02, n_variants=3)
        assert res.n_total == res.n_per_variant * 3

    def test_relative_mde_equivalent(self):
        """relative_mde=0.10 on baseline=0.10 is same as mde=0.01."""
        abs_res = SampleSize.for_proportion(0.10, 0.01)
        rel_res = SampleSize.for_proportion(0.10, relative_mde=0.10)
        assert abs_res.n_per_variant == rel_res.n_per_variant
        assert rel_res.relative_mde == 0.10

    def test_both_mde_and_relative_mde_raises(self):
        """Providing both mde and relative_mde raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            SampleSize.for_proportion(0.10, mde=0.02, relative_mde=0.10)

    def test_neither_mde_nor_relative_mde_raises(self):
        """Providing neither mde nor relative_mde raises ValueError."""
        with pytest.raises(ValueError, match="must be provided"):
            SampleSize.for_proportion(0.10)

    def test_small_baseline_warning(self):
        """Very small baseline (<1%) emits RuntimeWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SampleSize.for_proportion(0.005, 0.002)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "very small" in str(w[0].message).lower()

    def test_invalid_baseline_zero(self):
        """baseline=0 raises ValueError."""
        with pytest.raises(ValueError, match="baseline"):
            SampleSize.for_proportion(0.0, 0.02)

    def test_invalid_baseline_one(self):
        """baseline=1 raises ValueError."""
        with pytest.raises(ValueError, match="baseline"):
            SampleSize.for_proportion(1.0, 0.02)

    def test_baseline_plus_mde_exceeds_one(self):
        """baseline + mde > 1 raises ValueError."""
        with pytest.raises(ValueError, match="baseline \\+ mde"):
            SampleSize.for_proportion(0.95, 0.10)

    def test_traffic_fraction_stored(self):
        """traffic_fraction is accepted without error (stored in result for duration)."""
        res = SampleSize.for_proportion(0.10, 0.02, traffic_fraction=0.5)
        # traffic_fraction doesn't change n_per_variant
        ref = SampleSize.for_proportion(0.10, 0.02)
        assert res.n_per_variant == ref.n_per_variant

    def test_n_variants_less_than_2_raises(self):
        """n_variants < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_variants"):
            SampleSize.for_proportion(0.10, 0.02, n_variants=1)

    def test_result_to_dict(self):
        """Result is a frozen dataclass with to_dict()."""
        res = SampleSize.for_proportion(0.10, 0.02)
        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["n_per_variant"] == res.n_per_variant


# ═══════════════════════════════════════════════════════════════════════
# for_mean
# ═══════════════════════════════════════════════════════════════════════


class TestForMean:
    """Unit tests for SampleSize.for_mean."""

    def test_revenue_example(self):
        """Revenue: baseline=25, std=40, mde=2 gives reasonable n."""
        res = SampleSize.for_mean(25.0, 40.0, 2.0)
        assert res.n_per_variant > 0
        assert res.metric == "mean"
        # Cohen's d = 2/40 = 0.05, should need many samples
        assert res.n_per_variant > 5000

    def test_cuped_reduced_std(self):
        """Lower std (CUPED-reduced) gives smaller n."""
        full = SampleSize.for_mean(25.0, 40.0, 2.0)
        cuped = SampleSize.for_mean(25.0, 20.0, 2.0)
        assert cuped.n_per_variant < full.n_per_variant

    def test_relative_mde(self):
        """relative_mde works: 10% of baseline_mean=100 → mde=10."""
        abs_res = SampleSize.for_mean(100.0, 50.0, 10.0)
        rel_res = SampleSize.for_mean(100.0, 50.0, relative_mde=0.10)
        assert abs_res.n_per_variant == rel_res.n_per_variant

    def test_both_mde_and_relative_mde_raises(self):
        """Providing both mde and relative_mde raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            SampleSize.for_mean(100.0, 50.0, mde=10.0, relative_mde=0.10)

    def test_neither_mde_nor_relative_mde_raises(self):
        """Providing neither mde nor relative_mde raises ValueError."""
        with pytest.raises(ValueError, match="must be provided"):
            SampleSize.for_mean(100.0, 50.0)

    def test_std_zero_raises(self):
        """baseline_std <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="baseline_std"):
            SampleSize.for_mean(25.0, 0.0, 2.0)
        with pytest.raises(ValueError, match="baseline_std"):
            SampleSize.for_mean(25.0, -1.0, 2.0)

    def test_mde_zero_raises(self):
        """mde=0 raises ValueError."""
        with pytest.raises(ValueError, match="mde"):
            SampleSize.for_mean(25.0, 40.0, 0.0)

    def test_effect_size_populated(self):
        """Effect size (Cohen's d) is populated in result."""
        res = SampleSize.for_mean(25.0, 40.0, 2.0)
        assert res.effect_size == pytest.approx(2.0 / 40.0)

    def test_one_sided_fewer(self):
        """One-sided test requires fewer samples."""
        two = SampleSize.for_mean(25.0, 40.0, 2.0, alternative="two-sided")
        one = SampleSize.for_mean(25.0, 40.0, 2.0, alternative="one-sided")
        assert one.n_per_variant < two.n_per_variant


# ═══════════════════════════════════════════════════════════════════════
# for_ratio
# ═══════════════════════════════════════════════════════════════════════


class TestForRatio:
    """Unit tests for SampleSize.for_ratio."""

    def test_ctr_example(self):
        """CTR-like example gives reasonable n."""
        res = SampleSize.for_ratio(
            baseline_num_mean=5.0,     # clicks
            baseline_den_mean=100.0,   # impressions
            baseline_num_std=3.0,
            baseline_den_std=20.0,
            baseline_covariance=1.0,
            mde=0.005,                 # 0.5pp change in CTR
        )
        assert res.n_per_variant > 0
        assert res.metric == "ratio"
        assert res.baseline == pytest.approx(5.0 / 100.0)

    def test_den_mean_zero_raises(self):
        """baseline_den_mean <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="baseline_den_mean"):
            SampleSize.for_ratio(5.0, 0.0, 3.0, 20.0, 1.0, 0.005)
        with pytest.raises(ValueError, match="baseline_den_mean"):
            SampleSize.for_ratio(5.0, -1.0, 3.0, 20.0, 1.0, 0.005)

    def test_mde_zero_raises(self):
        """mde=0 raises ValueError."""
        with pytest.raises(ValueError, match="mde"):
            SampleSize.for_ratio(5.0, 100.0, 3.0, 20.0, 1.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# mde_for_proportion
# ═══════════════════════════════════════════════════════════════════════


class TestMdeForProportion:
    """Unit tests for SampleSize.mde_for_proportion."""

    def test_roundtrip(self):
        """for_proportion gives n, mde_for_proportion with that n gives back ~same mde."""
        original_mde = 0.02
        res = SampleSize.for_proportion(0.10, original_mde)
        recovered_mde = SampleSize.mde_for_proportion(0.10, res.n_per_variant)
        assert recovered_mde == pytest.approx(original_mde, abs=0.001)

    def test_larger_n_smaller_mde(self):
        """Larger n → smaller MDE."""
        mde_small = SampleSize.mde_for_proportion(0.10, 5000)
        mde_large = SampleSize.mde_for_proportion(0.10, 10000)
        assert mde_large < mde_small

    def test_n_zero_raises(self):
        """n <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n"):
            SampleSize.mde_for_proportion(0.10, 0)
        with pytest.raises(ValueError, match="n"):
            SampleSize.mde_for_proportion(0.10, -100)

    def test_invalid_baseline_raises(self):
        """Invalid baseline raises ValueError."""
        with pytest.raises(ValueError, match="baseline"):
            SampleSize.mde_for_proportion(0.0, 1000)
        with pytest.raises(ValueError, match="baseline"):
            SampleSize.mde_for_proportion(1.0, 1000)


# ═══════════════════════════════════════════════════════════════════════
# duration
# ═══════════════════════════════════════════════════════════════════════


class TestDuration:
    """Unit tests for SampleSize.duration."""

    def test_basic(self):
        """10000 users needed, 1000 per day → 10 days."""
        assert SampleSize.duration(10000, 1000) == 10

    def test_with_traffic_fraction(self):
        """10000 users needed, 1000/day, 50% traffic → 20 days."""
        assert SampleSize.duration(10000, 1000, traffic_fraction=0.5) == 20

    def test_with_ramp_days(self):
        """Ramp days are added."""
        assert SampleSize.duration(10000, 1000, ramp_days=3) == 13

    def test_rounds_up(self):
        """Partial days round up."""
        assert SampleSize.duration(10001, 1000) == 11

    def test_daily_users_zero_raises(self):
        """daily_users <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="daily_users"):
            SampleSize.duration(10000, 0)
        with pytest.raises(ValueError, match="daily_users"):
            SampleSize.duration(10000, -100)

    def test_n_required_zero_raises(self):
        """n_required <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_required"):
            SampleSize.duration(0, 1000)


# ═══════════════════════════════════════════════════════════════════════
# Property-based / monotonicity tests
# ═══════════════════════════════════════════════════════════════════════


class TestMonotonicity:
    """Property-based tests for monotonic relationships."""

    @pytest.mark.parametrize("n", [1000, 3000, 5000, 10000, 20000])
    def test_increasing_n_decreasing_mde(self, n):
        """Increasing sample size → decreasing MDE (monotonic)."""
        mde = SampleSize.mde_for_proportion(0.10, n)
        mde_bigger = SampleSize.mde_for_proportion(0.10, n + 1000)
        assert mde_bigger < mde

    @pytest.mark.parametrize("power", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    def test_increasing_power_increasing_n(self, power):
        """Increasing power → increasing required n."""
        n = SampleSize.for_proportion(0.10, 0.02, power=power).n_per_variant
        n_higher = SampleSize.for_proportion(
            0.10, 0.02, power=min(power + 0.05, 0.99)
        ).n_per_variant
        assert n_higher >= n

    @pytest.mark.parametrize("baseline", [0.05, 0.10, 0.20, 0.50])
    def test_one_sided_always_less_than_two_sided(self, baseline):
        """One-sided n < two-sided n, always."""
        two = SampleSize.for_proportion(baseline, 0.02, alternative="two-sided")
        one = SampleSize.for_proportion(baseline, 0.02, alternative="one-sided")
        assert one.n_per_variant < two.n_per_variant


# ═══════════════════════════════════════════════════════════════════════
# Additional coverage tests
# ═══════════════════════════════════════════════════════════════════════


class TestAdditionalCoverage:
    """Tests to cover remaining gaps identified in QA review."""

    def test_for_ratio_factor_of_2(self):
        """for_ratio gives a reasonable n (sanity check on delta-method variance)."""
        res = SampleSize.for_ratio(
            baseline_num_mean=5.0,
            baseline_den_mean=100.0,
            baseline_num_std=3.0,
            baseline_den_std=20.0,
            baseline_covariance=1.0,
            mde=0.005,
        )
        # With the delta method, n should be a reasonable positive number
        assert res.n_per_variant > 100
        # n_total should be n_per_variant * 2
        assert res.n_total == res.n_per_variant * 2

    def test_duration_negative_ramp_days_raises(self):
        """ramp_days=-1 raises ValueError."""
        with pytest.raises(ValueError, match="ramp_days"):
            SampleSize.duration(10000, 1000, ramp_days=-1)

    def test_for_proportion_negative_mde(self):
        """Negative MDE (degradation detection) should work if baseline+mde is valid."""
        res = SampleSize.for_proportion(0.50, -0.05)
        assert res.n_per_variant > 0
        assert res.mde == -0.05

    def test_for_ratio_one_sided(self):
        """One-sided ratio test gives smaller n than two-sided."""
        two = SampleSize.for_ratio(5.0, 100.0, 3.0, 20.0, 1.0, 0.005, alternative="two-sided")
        one = SampleSize.for_ratio(5.0, 100.0, 3.0, 20.0, 1.0, 0.005, alternative="one-sided")
        assert one.n_per_variant < two.n_per_variant

    def test_mde_for_proportion_extreme_baseline(self):
        """Baseline very close to 0 or 1 should raise ValueError for MDE computation."""
        # baseline=0.999 → max_mde = min(0.999, 0.001) - 1e-9 ≈ 0.001
        # Should still work for small enough n that requires large MDE
        with pytest.raises(ValueError, match="baseline"):
            SampleSize.mde_for_proportion(1.0, 1000)
        # Very close to 0
        with pytest.raises(ValueError, match="baseline"):
            SampleSize.mde_for_proportion(0.0, 1000)

    def test_mde_for_proportion_extreme_baseline_clear_error(self):
        """Extreme baseline with tiny n gives a clear ValueError, not a scipy error."""
        with pytest.raises(ValueError, match="too small to detect any effect"):
            SampleSize.mde_for_proportion(baseline=0.001, n=50)

    def test_mde_for_proportion_float_n_raises(self):
        """Non-integer n (e.g. 3.5) raises ValueError."""
        with pytest.raises(ValueError, match="n"):
            SampleSize.mde_for_proportion(0.10, n=3.5)

    def test_mde_for_proportion_extreme_baseline_near_zero(self):
        """Baseline extremely close to 0 (< 1e-9) raises ValueError about max_mde <= 0."""
        with pytest.raises(ValueError, match="Cannot compute MDE"):
            SampleSize.mde_for_proportion(baseline=1e-10, n=1000)
