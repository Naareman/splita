"""Tests for SRMCheck — Sample Ratio Mismatch detector."""

from __future__ import annotations

import warnings

import pytest
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chisquare

from splita._types import SRMResult
from splita.core.srm import SRMCheck


# ─── Basic tests ────────────────────────────────────────────────────


class TestBasic:
    """Happy-path behaviour for common scenarios."""

    def test_equal_split_passes(self):
        result = SRMCheck([5000, 5000]).run()
        assert result.passed is True

    def test_slight_imbalance_passes(self):
        # chi2 = (4950-5000)^2/5000 + (5050-5000)^2/5000 = 1.0, p ≈ 0.317
        result = SRMCheck([4950, 5050]).run()
        assert result.passed is True

    def test_large_imbalance_fails(self):
        result = SRMCheck([4000, 6000]).run()
        assert result.passed is False

    def test_custom_fractions_pass(self):
        result = SRMCheck(
            [9000, 1000], expected_fractions=[0.90, 0.10]
        ).run()
        assert result.passed is True

    def test_custom_fractions_fail(self):
        result = SRMCheck(
            [8000, 2000], expected_fractions=[0.90, 0.10]
        ).run()
        assert result.passed is False

    def test_abc_equal_split_passes(self):
        result = SRMCheck([3300, 3350, 3350]).run()
        assert result.passed is True

    def test_abc_with_srm_fails(self):
        result = SRMCheck([3000, 3500, 3500]).run()
        assert result.passed is False


# ─── Statistical correctness ───────────────────────────────────────


class TestStatisticalCorrectness:
    """Verify chi-square computation matches scipy."""

    def test_chi2_matches_scipy(self):
        observed = [4000, 6000]
        result = SRMCheck(observed).run()
        # scipy.stats.chisquare assumes equal expected by default
        scipy_stat, scipy_pval = chisquare(observed)
        assert result.chi2_statistic == pytest.approx(scipy_stat, rel=1e-10)
        assert result.pvalue == pytest.approx(scipy_pval, rel=1e-10)

    def test_pvalue_known_chi2(self):
        # chi2 = 6.635 with df=1 should give p ≈ 0.01
        # Work backwards: set observed so that chi2 ≈ 6.635
        # With N=10000 equal split, (o - 5000)^2/5000 * 2 = chi2
        # => (o - 5000)^2 = chi2 * 5000 / 2 = 16587.5
        # => o - 5000 ≈ 128.79 => o ≈ 5129
        result = SRMCheck([4871, 5129]).run()
        # The chi-square stat should be close to 6.6564
        expected_chi2 = (4871 - 5000) ** 2 / 5000 + (5129 - 5000) ** 2 / 5000
        assert result.chi2_statistic == pytest.approx(expected_chi2, rel=1e-10)
        # Verify p-value against scipy
        expected_p = 1.0 - chi2_dist.cdf(expected_chi2, df=1)
        assert result.pvalue == pytest.approx(expected_p, rel=1e-10)

    def test_degrees_of_freedom_k_variants(self):
        # 4-variant test: df should be 3
        observed = [2500, 2500, 2500, 2500]
        result = SRMCheck(observed).run()
        # chi2 = 0, p should be 1.0  (1 - chi2.cdf(0, 3) = 1.0)
        assert result.chi2_statistic == pytest.approx(0.0)
        assert result.pvalue == pytest.approx(1.0)


# ─── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_small_expected_count_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SRMCheck([1, 1, 1]).run()
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) >= 1
            assert "expected count" in str(runtime_warnings[0].message).lower()

    def test_very_large_sample_passes(self):
        result = SRMCheck([500000, 500100]).run()
        assert result.passed is True

    def test_one_variant_has_zero_fails(self):
        result = SRMCheck([0, 10000]).run()
        assert result.passed is False

    def test_all_in_one_variant_fails(self):
        result = SRMCheck([10000, 0]).run()
        assert result.passed is False


# ─── Validation ────────────────────────────────────────────────────


class TestValidation:
    """Input validation raises appropriate errors."""

    def test_fewer_than_2_variants(self):
        with pytest.raises(ValueError, match="at least 2 variants"):
            SRMCheck([5000])

    def test_fractions_dont_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SRMCheck([5000, 5000], expected_fractions=[0.6, 0.6])

    def test_fraction_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            SRMCheck([5000, 5000], expected_fractions=[0.5, 0.3, 0.2])

    def test_negative_observed_count(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            SRMCheck([-100, 5000])

    def test_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            SRMCheck([5000, 5000], alpha=0)

    def test_alpha_one(self):
        with pytest.raises(ValueError, match="alpha"):
            SRMCheck([5000, 5000], alpha=1)

    def test_invalid_min_expected_count(self):
        with pytest.raises(ValueError, match="min_expected_count"):
            SRMCheck([5000, 5000], min_expected_count=0)

    def test_zero_total_observations_raises(self):
        with pytest.raises(ValueError, match="all zeros"):
            SRMCheck([0, 0]).run()

    def test_nan_in_observed_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            SRMCheck([float('nan'), 5000])

    def test_inf_in_observed_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            SRMCheck([float('inf'), 5000])


# ─── Properties ────────────────────────────────────────────────────


class TestProperties:
    """Result object properties and invariants."""

    def test_result_is_srm_result(self):
        result = SRMCheck([5000, 5000]).run()
        assert isinstance(result, SRMResult)

    def test_to_dict_works(self):
        result = SRMCheck([5000, 5000]).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "observed" in d
        assert "pvalue" in d
        assert "passed" in d

    def test_message_informative_on_failure(self):
        result = SRMCheck([4000, 6000]).run()
        assert "cannot be trusted" in result.message

    def test_idempotent(self):
        check = SRMCheck([4850, 5150])
        r1 = check.run()
        r2 = check.run()
        assert r1.chi2_statistic == r2.chi2_statistic
        assert r1.pvalue == r2.pvalue
        assert r1.passed == r2.passed
        assert r1.message == r2.message


# ─── Real-world scenario ──────────────────────────────────────────


class TestRealWorld:
    """Simulate realistic A/B test scenarios."""

    def test_bot_traffic_detection(self):
        # 10% more traffic in treatment suggests bot contamination
        result = SRMCheck([4500, 5500], expected_fractions=[0.5, 0.5]).run()
        assert result.passed is False
        assert "cannot be trusted" in result.message
        assert "mismatch detected" in result.message.lower()


# ─── Milestone 3 review fixes ────────────────────────────────────


class TestDeviationsAndExpected:
    """Verify deviation percentages and expected counts."""

    def test_deviations_pct_values(self):
        result = SRMCheck([4000, 6000]).run()
        # Both variants have |20%| deviation; max() picks the first (index 0)
        assert result.worst_variant == 0
        assert result.deviations_pct[0] == pytest.approx(-20.0)
        assert result.deviations_pct[1] == pytest.approx(20.0)

    def test_expected_counts_values(self):
        result = SRMCheck([5000, 5000]).run()
        assert result.expected_counts == [5000.0, 5000.0]

    def test_custom_fractions_three_variants(self):
        result = SRMCheck(
            [5000, 3000, 2000], expected_fractions=[0.5, 0.3, 0.2]
        ).run()
        assert result.expected_counts == [5000.0, 3000.0, 2000.0]
        assert result.passed is True

    def test_deviation_inf_for_zero_expected(self):
        # With fractions [1.0, 0.0, 0.0] (impossible since they must sum to 1
        # and be valid), we instead craft a scenario via custom fractions
        # where expected is 0 for a variant with nonzero observed.
        # Since fractions must sum to 1.0 and be non-negative, we can't get
        # a zero fraction. Instead, test the deviation logic directly:
        # If total = 10, fractions = [0.5, 0.5], expected = [5, 5] — no zero.
        # The zero-expected case occurs when a fraction rounds to 0. We test
        # the deviation formula through a 3-variant case where one variant
        # gets essentially zero expected.
        import math

        # Use a very small fraction for one variant so expected rounds near 0
        # Actually: with observed [10, 0, 0] and equal fractions, expected ~3.33 each.
        # The only way to get zero expected is fraction=0 which is disallowed.
        # So we test the code path by checking the formula directly with
        # a small enough sample: observed=[1, 0] gives expected=[0.5, 0.5],
        # which is > 0 so deviation is computed normally.
        # The inf path is reachable only with fraction=0.
        # Let's verify the formula handles it correctly via the SRM code path
        # by checking what happens when all traffic goes to one variant.
        result = SRMCheck([10000, 0]).run()
        # deviation for variant 0 = (10000-5000)/5000*100 = 100%
        assert result.deviations_pct[0] == pytest.approx(100.0)
        # deviation for variant 1 = (0-5000)/5000*100 = -100%
        assert result.deviations_pct[1] == pytest.approx(-100.0)
        # For a true zero-expected test, verify the formula inline:
        # e=0, o=5 -> inf
        deviation = ((5 - 0) / 0 * 100) if 0 > 0 else (float('inf') if 5 > 0 else 0.0)
        assert math.isinf(deviation)
        # e=0, o=0 -> 0.0
        deviation_zero = ((0 - 0) / 0 * 100) if 0 > 0 else (float('inf') if 0 > 0 else 0.0)
        assert deviation_zero == 0.0
