"""Tests for TrimmedMeanEstimator."""

import numpy as np
import pytest
from scipy.stats import ttest_ind

from splita._types import TrimmedMeanResult
from splita.variance.trimmed_mean import TrimmedMeanEstimator


# ─── Basic functionality ──────────────────────────────────────────


class TestBasic:
    def test_no_effect(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(100, 10, 500)
        trt = rng.normal(100, 10, 500)
        result = TrimmedMeanEstimator().fit_transform(ctrl, trt)
        assert not result.significant
        assert result.pvalue > 0.05
        assert result.ci_lower < 0 < result.ci_upper

    def test_known_effect_recovery(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(100, 10, 1000)
        trt = rng.normal(105, 10, 1000)
        result = TrimmedMeanEstimator().fit_transform(ctrl, trt)
        assert result.significant
        assert result.pvalue < 0.05
        # ATE should be close to 5
        assert 3.0 < result.ate < 7.0

    def test_result_is_frozen_dataclass(self):
        rng = np.random.default_rng(42)
        result = TrimmedMeanEstimator().fit_transform(
            rng.normal(0, 1, 100), rng.normal(0, 1, 100)
        )
        assert isinstance(result, TrimmedMeanResult)
        with pytest.raises(AttributeError):
            result.ate = 0.0  # type: ignore[misc]

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        result = TrimmedMeanEstimator().fit_transform(
            rng.normal(0, 1, 100), rng.normal(0, 1, 100)
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "trim_fraction" in d


# ─── Variance reduction ───────────────────────────────────────────


class TestVarianceReduction:
    def test_reduces_variance_with_outliers(self):
        """Trimmed mean should have smaller SE than raw when outliers are present."""
        rng = np.random.default_rng(42)
        ctrl = np.concatenate([rng.normal(100, 5, 480), rng.normal(500, 100, 20)])
        trt = np.concatenate([rng.normal(105, 5, 480), rng.normal(500, 100, 20)])

        # Raw SE
        raw_se = np.sqrt(
            np.var(ctrl, ddof=1) / len(ctrl) + np.var(trt, ddof=1) / len(trt)
        )

        # Trimmed SE
        result = TrimmedMeanEstimator(trim_fraction=0.10).fit_transform(ctrl, trt)
        assert result.se < raw_se

    def test_trimming_removes_expected_count(self):
        rng = np.random.default_rng(42)
        n = 200
        ctrl = rng.normal(0, 1, n)
        trt = rng.normal(0, 1, n)
        trim = 0.10
        result = TrimmedMeanEstimator(trim_fraction=trim).fit_transform(ctrl, trt)
        expected_n = n - 2 * int(n * trim / 2)
        assert result.n_trimmed_control == expected_n
        assert result.n_trimmed_treatment == expected_n

    def test_zero_trim_matches_raw(self):
        """With trim_fraction=0 (exclusive bounds prevent 0, use very small)."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, 100)
        trt = rng.normal(10.5, 2, 100)
        # Use very small trim that removes 0 observations
        result = TrimmedMeanEstimator(trim_fraction=0.001).fit_transform(ctrl, trt)
        assert result.n_trimmed_control == 100
        assert result.n_trimmed_treatment == 100


# ─── Different trim fractions ─────────────────────────────────────


class TestTrimFractions:
    def test_small_trim(self):
        rng = np.random.default_rng(42)
        result = TrimmedMeanEstimator(trim_fraction=0.02).fit_transform(
            rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)
        )
        assert result.trim_fraction == 0.02

    def test_large_trim(self):
        rng = np.random.default_rng(42)
        result = TrimmedMeanEstimator(trim_fraction=0.40).fit_transform(
            rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)
        )
        assert result.trim_fraction == 0.40
        assert result.n_trimmed_control < 200
        assert result.n_trimmed_treatment < 200


# ─── Confidence interval ──────────────────────────────────────────


class TestCI:
    def test_ci_contains_ate(self):
        rng = np.random.default_rng(42)
        result = TrimmedMeanEstimator().fit_transform(
            rng.normal(0, 1, 200), rng.normal(0.3, 1, 200)
        )
        assert result.ci_lower <= result.ate <= result.ci_upper

    def test_ci_coverage(self):
        """95% CI should contain true effect ~95% of the time."""
        true_effect = 1.0
        covers = 0
        n_sims = 200
        for i in range(n_sims):
            rng = np.random.default_rng(i)
            ctrl = rng.normal(0, 1, 100)
            trt = rng.normal(true_effect, 1, 100)
            result = TrimmedMeanEstimator(alpha=0.05).fit_transform(ctrl, trt)
            if result.ci_lower <= true_effect <= result.ci_upper:
                covers += 1
        coverage = covers / n_sims
        assert 0.88 < coverage < 1.0


# ─── Validation ────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            TrimmedMeanEstimator(alpha=1.5)

    def test_invalid_trim_fraction(self):
        with pytest.raises(ValueError, match="trim_fraction"):
            TrimmedMeanEstimator(trim_fraction=1.5)

    def test_too_short_control(self):
        with pytest.raises(ValueError, match="control"):
            TrimmedMeanEstimator().fit_transform([1], [1, 2, 3])

    def test_invalid_control_type(self):
        with pytest.raises(TypeError, match="control"):
            TrimmedMeanEstimator().fit_transform("not_array", [1, 2, 3])

    def test_excessive_trimming(self):
        # n=10, trim_fraction=0.99 => k=int(10*0.99/2)=4, leaves 2
        # n=4, trim_fraction=0.99 => k=int(4*0.99/2)=1, leaves 2
        # Need trim_fraction that leaves <2 elements
        # n=10, trim_fraction=0.99 => k=4, leaves 2 (still ok)
        # Use n=6, trim_fraction=0.99 => k=int(6*0.99/2)=2, leaves 2 (still ok)
        # Actually, need to ensure the trim function leaves <2
        # n=3, trim_fraction=0.99 => k=int(3*0.99/2)=1, leaves 1 => triggers error
        with pytest.raises(ValueError, match="Trimming removed"):
            TrimmedMeanEstimator(trim_fraction=0.99).fit_transform(
                [1, 2, 3], [4, 5, 6]
            )


# ─── Statistical audit (simulation) ──────────────────────────────


class TestStatisticalAudit:
    def test_type_i_error_rate(self):
        """Under H0, should reject at most ~alpha."""
        alpha = 0.05
        n_sims = 200
        rejections = 0
        for i in range(n_sims):
            rng = np.random.default_rng(i + 1000)
            ctrl = rng.normal(10, 2, 100)
            trt = rng.normal(10, 2, 100)
            result = TrimmedMeanEstimator(alpha=alpha).fit_transform(ctrl, trt)
            if result.significant:
                rejections += 1
        fpr = rejections / n_sims
        # Allow some slack: should be below 2 * alpha
        assert fpr < 2 * alpha


# ─── Repr ──────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_key_info(self):
        rng = np.random.default_rng(42)
        result = TrimmedMeanEstimator().fit_transform(
            rng.normal(0, 1, 100), rng.normal(0, 1, 100)
        )
        text = repr(result)
        assert "TrimmedMeanResult" in text
        assert "ate" in text
        assert "pvalue" in text
