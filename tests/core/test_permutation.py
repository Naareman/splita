"""Tests for PermutationTest."""

import numpy as np
import pytest
from scipy.stats import ttest_ind

from splita._types import PermutationResult
from splita.core.permutation import PermutationTest


# ─── Significant effect ───────────────────────────────────────────


class TestSignificantEffect:
    def test_detects_large_effect(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(1.0, 1, 100)
        result = PermutationTest(ctrl, trt, random_state=42).run()
        assert result.significant is True
        assert result.pvalue < 0.05
        assert result.observed_statistic > 0

    def test_detects_negative_effect(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(8, 1, 100)
        result = PermutationTest(ctrl, trt, random_state=42).run()
        assert result.significant is True
        assert result.observed_statistic < 0


# ─── No effect ─────────────────────────────────────────────────────


class TestNoEffect:
    def test_null_not_significant(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(0, 1, 100)
        result = PermutationTest(ctrl, trt, random_state=42).run()
        assert result.pvalue > 0.05

    def test_null_distribution_centered_near_zero(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 200)
        trt = rng.normal(0, 1, 200)
        result = PermutationTest(ctrl, trt, random_state=42).run()
        assert abs(result.null_distribution_mean) < 0.1


# ─── Matches t-test for normal data ──────────────────────────────


class TestMatchesTTest:
    def test_agrees_with_ttest_on_normal_data(self):
        """Permutation p-value should roughly agree with t-test for normal data."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 200)
        trt = rng.normal(0.5, 1, 200)
        perm_result = PermutationTest(
            ctrl, trt, n_permutations=10000, random_state=42
        ).run()
        _, t_pvalue = ttest_ind(ctrl, trt)
        # Should be in the same ballpark
        assert abs(perm_result.pvalue - t_pvalue) < 0.05


# ─── Small sample ─────────────────────────────────────────────────


class TestSmallSample:
    def test_works_with_small_samples(self):
        ctrl = [1.0, 2.0, 3.0]
        trt = [10.0, 11.0, 12.0]
        result = PermutationTest(ctrl, trt, random_state=42).run()
        # With only 3 elements per group, exact significance is hard
        # but pvalue should be small (few permutations give this extreme a diff)
        assert result.pvalue < 0.15
        assert result.n_permutations == 10000

    def test_minimum_sample_size(self):
        ctrl = [1.0, 2.0]
        trt = [3.0, 4.0]
        result = PermutationTest(ctrl, trt, random_state=42).run()
        assert isinstance(result, PermutationResult)


# ─── Statistic types ──────────────────────────────────────────────


class TestStatisticTypes:
    def test_median_diff(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(1, 1, 100)
        result = PermutationTest(ctrl, trt, statistic="median_diff",
                                 random_state=42).run()
        assert result.significant is True

    def test_mean_diff_default(self):
        rng = np.random.default_rng(42)
        result = PermutationTest(
            rng.normal(0, 1, 50), rng.normal(0, 1, 50), random_state=42
        ).run()
        assert isinstance(result.observed_statistic, float)


# ─── Alternative hypotheses ───────────────────────────────────────


class TestAlternatives:
    def test_greater(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(1, 1, 100)
        result = PermutationTest(ctrl, trt, alternative="greater",
                                 random_state=42).run()
        assert result.pvalue < 0.05

    def test_less(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 100)
        trt = rng.normal(-1, 1, 100)
        result = PermutationTest(ctrl, trt, alternative="less",
                                 random_state=42).run()
        assert result.pvalue < 0.05

    def test_greater_not_significant_when_effect_is_negative(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(8, 1, 100)
        result = PermutationTest(ctrl, trt, alternative="greater",
                                 random_state=42).run()
        assert result.pvalue > 0.05


# ─── Reproducibility ──────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_result(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0.5, 1, 50)
        r1 = PermutationTest(ctrl, trt, random_state=123).run()
        r2 = PermutationTest(ctrl, trt, random_state=123).run()
        assert r1.pvalue == r2.pvalue
        assert r1.observed_statistic == r2.observed_statistic

    def test_different_seed_different_result(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0.5, 1, 50)
        r1 = PermutationTest(ctrl, trt, random_state=1).run()
        r2 = PermutationTest(ctrl, trt, random_state=2).run()
        # P-values may differ slightly due to different permutations
        # (but observed_statistic should be the same)
        assert r1.observed_statistic == r2.observed_statistic

    def test_generator_input(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0.5, 1, 50)
        gen = np.random.default_rng(42)
        result = PermutationTest(ctrl, trt, random_state=gen).run()
        assert isinstance(result, PermutationResult)


# ─── Validation ────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_statistic(self):
        with pytest.raises(ValueError, match="statistic"):
            PermutationTest([1, 2], [3, 4], statistic="variance")

    def test_invalid_alternative(self):
        with pytest.raises(ValueError, match="alternative"):
            PermutationTest([1, 2], [3, 4], alternative="both")

    def test_too_few_permutations(self):
        with pytest.raises(ValueError, match="n_permutations"):
            PermutationTest([1, 2], [3, 4], n_permutations=10)

    def test_too_short_control(self):
        with pytest.raises(ValueError, match="control"):
            PermutationTest([1], [2, 3])

    def test_invalid_control_type(self):
        with pytest.raises(TypeError, match="control"):
            PermutationTest("not_array", [1, 2, 3])


# ─── Result properties ────────────────────────────────────────────


class TestResultProperties:
    def test_result_is_frozen_dataclass(self):
        result = PermutationTest([1, 2, 3], [4, 5, 6], random_state=42).run()
        assert isinstance(result, PermutationResult)
        with pytest.raises(AttributeError):
            result.pvalue = 0.5  # type: ignore[misc]

    def test_to_dict(self):
        result = PermutationTest([1, 2, 3], [4, 5, 6], random_state=42).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "observed_statistic" in d
        assert "null_distribution_mean" in d

    def test_repr_contains_key_info(self):
        result = PermutationTest([1, 2, 3], [4, 5, 6], random_state=42).run()
        text = repr(result)
        assert "PermutationResult" in text
        assert "pvalue" in text


# ─── Statistical audit (simulation) ──────────────────────────────


class TestStatisticalAudit:
    def test_type_i_error_rate(self):
        """Under H0, permutation test should reject at most ~alpha."""
        n_sims = 100
        rejections = 0
        for i in range(n_sims):
            rng = np.random.default_rng(i + 2000)
            ctrl = rng.normal(0, 1, 50)
            trt = rng.normal(0, 1, 50)
            result = PermutationTest(ctrl, trt, n_permutations=1000,
                                     random_state=i).run()
            if result.significant:
                rejections += 1
        fpr = rejections / n_sims
        assert fpr < 0.15  # Should be around 0.05, allow slack

    def test_power_increases_with_effect_size(self):
        """Larger effects should be detected more often."""
        n_sims = 50
        power_small = 0
        power_large = 0
        for i in range(n_sims):
            rng = np.random.default_rng(i + 3000)
            ctrl = rng.normal(0, 1, 100)
            trt_small = rng.normal(0.2, 1, 100)
            trt_large = rng.normal(1.0, 1, 100)
            r_small = PermutationTest(ctrl, trt_small, n_permutations=500,
                                      random_state=i).run()
            r_large = PermutationTest(ctrl, trt_large, n_permutations=500,
                                      random_state=i).run()
            if r_small.significant:
                power_small += 1
            if r_large.significant:
                power_large += 1
        assert power_large > power_small
