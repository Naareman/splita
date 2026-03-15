"""Tests for the advisory/recommendation system."""

from __future__ import annotations

import io
import sys
import warnings

import numpy as np
import pytest

from splita._advisory import (
    advise_method_choice,
    advise_multiple_testing,
    advise_sample_size,
    advise_sequential,
    advise_variance_reduction,
    info,
)
from splita.core.experiment import Experiment
from splita.variance.cuped import CUPED
from splita.verbose import verbose


# ── advise_method_choice ─────────────────────────────────────────────


class TestAdviseMethodChoice:
    """Tests for advise_method_choice."""

    def test_skewed_data_ttest_warns(self):
        """High skewness + ttest should warn to use mannwhitney/bootstrap."""
        rng = np.random.default_rng(42)
        # Generate highly skewed data (exponential)
        data = rng.exponential(scale=1.0, size=200)
        data = np.concatenate([data, [100.0] * 10])  # add extreme values

        with pytest.warns(RuntimeWarning, match="high skewness"):
            advise_method_choice(data, "ttest", "continuous", len(data))

    def test_low_skew_ttest_no_warning(self):
        """Normal data + ttest should not warn."""
        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, size=200)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_method_choice(data, "ttest", "continuous", len(data))

    def test_large_n_mannwhitney_suggests_ttest(self):
        """Large n + low skewness + mannwhitney should suggest ttest."""
        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, size=6000)

        with pytest.warns(RuntimeWarning, match="t-test would be more powerful"):
            advise_method_choice(data, "mannwhitney", "continuous", len(data))

    def test_small_n_mannwhitney_no_warning(self):
        """Small n + mannwhitney should not trigger the large-n advisory."""
        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, size=100)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_method_choice(data, "mannwhitney", "continuous", len(data))

    def test_small_n_ztest_warns(self):
        """Small n + ztest should warn about normal approximation."""
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with pytest.warns(RuntimeWarning, match="z-test normal approximation"):
            advise_method_choice(data, "ztest", "conversion", len(data))

    def test_large_n_ztest_no_warning(self):
        """Large n + ztest should not warn."""
        rng = np.random.default_rng(42)
        data = rng.binomial(1, 0.5, size=100)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_method_choice(data, "ztest", "conversion", len(data))

    def test_conversion_metric_skips_skewness_check(self):
        """Skewness check only applies to continuous metrics."""
        data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 10)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not warn about skewness for conversion metric
            advise_method_choice(data, "ttest", "conversion", len(data))


# ── advise_sample_size ───────────────────────────────────────────────


class TestAdviseSampleSize:
    """Tests for advise_sample_size."""

    def test_small_sample_warns(self):
        """n < 30 should warn."""
        with pytest.warns(RuntimeWarning, match="Small sample size"):
            advise_sample_size(20, 20, "continuous")

    def test_adequate_sample_no_warning(self):
        """n >= 30 should not warn about sample size."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_sample_size(100, 100, "continuous")

    def test_imbalanced_groups_warns(self):
        """Ratio > 3:1 should warn."""
        with pytest.warns(RuntimeWarning, match="imbalanced"):
            advise_sample_size(100, 400, "continuous")

    def test_balanced_groups_no_warning(self):
        """Roughly balanced groups should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_sample_size(100, 120, "continuous")

    def test_both_small_and_imbalanced(self):
        """Both warnings can fire at once."""
        with pytest.warns(RuntimeWarning):
            advise_sample_size(10, 50, "continuous")


# ── advise_variance_reduction ────────────────────────────────────────


class TestAdviseVarianceReduction:
    """Tests for advise_variance_reduction."""

    def test_low_cuped_reduction_warns(self):
        """Variance reduction < 5% with CUPED should suggest CUPAC."""
        with pytest.warns(RuntimeWarning, match="CUPAC"):
            advise_variance_reduction(0.03, "CUPED", has_pre_data=True)

    def test_good_cuped_reduction_no_warning(self):
        """Variance reduction >= 5% should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_variance_reduction(0.30, "CUPED", has_pre_data=True)

    def test_non_cuped_method_no_warning(self):
        """Low reduction with non-CUPED method should not fire CUPED advisory."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_variance_reduction(0.03, "regression", has_pre_data=True)


# ── advise_multiple_testing ──────────────────────────────────────────


class TestAdviseMultipleTesting:
    """Tests for advise_multiple_testing."""

    def test_multiple_uncorrected_warns(self):
        """Multiple metrics without correction should warn."""
        with pytest.warns(RuntimeWarning, match="without correction"):
            advise_multiple_testing(5, corrected=False)

    def test_corrected_no_warning(self):
        """Corrected multiple testing should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_multiple_testing(5, corrected=True)

    def test_single_metric_no_warning(self):
        """Single metric should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_multiple_testing(1, corrected=False)


# ── advise_sequential ────────────────────────────────────────────────


class TestAdviseSequential:
    """Tests for advise_sequential."""

    def test_peeking_without_sequential_warns(self):
        """Multiple peeks without sequential testing should warn."""
        with pytest.warns(RuntimeWarning, match="peeked"):
            advise_sequential(3, is_sequential=False)

    def test_peeking_with_sequential_no_warning(self):
        """Sequential testing should not warn about peeking."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_sequential(3, is_sequential=True)

    def test_single_peek_no_warning(self):
        """Single analysis (no peeking) should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            advise_sequential(1, is_sequential=False)


# ── verbose context manager ──────────────────────────────────────────


class TestVerbose:
    """Tests for the verbose() context manager."""

    def test_verbose_enables_info(self):
        """verbose() should enable info messages."""
        import splita._advisory as adv

        assert not adv._VERBOSE
        with verbose():
            assert adv._VERBOSE
        assert not adv._VERBOSE

    def test_verbose_restores_state(self):
        """verbose() should restore prior state even on exception."""
        import splita._advisory as adv

        assert not adv._VERBOSE
        with pytest.raises(ValueError):
            with verbose():
                assert adv._VERBOSE
                raise ValueError("test error")
        assert not adv._VERBOSE

    def test_info_prints_when_verbose(self):
        """info() should print to stderr when verbose."""
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with verbose():
                info("test message")
        finally:
            sys.stderr = old_stderr
        assert "[splita] test message" in captured.getvalue()

    def test_info_silent_when_not_verbose(self):
        """info() should not print when not verbose."""
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            info("should not appear")
        finally:
            sys.stderr = old_stderr
        assert captured.getvalue() == ""


# ── integration: Experiment ──────────────────────────────────────────


class TestExperimentAdvisory:
    """Test advisory integration with Experiment class."""

    def test_small_sample_experiment_warns(self):
        """Experiment with small n should emit sample size warning."""
        ctrl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trt = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        with pytest.warns(RuntimeWarning, match="Small sample size"):
            Experiment(ctrl, trt)

    def test_explicit_ttest_on_skewed_data_warns(self):
        """User explicitly choosing ttest on skewed data should warn at run()."""
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1.0, size=500)
        data = np.concatenate([data, [200.0] * 20])
        ctrl = data[:260]
        trt = data[260:]

        exp = Experiment(ctrl, trt, method="ttest")
        with pytest.warns(RuntimeWarning, match="high skewness"):
            exp.run()

    def test_auto_method_no_method_choice_warning(self):
        """Auto method should not trigger advise_method_choice."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, size=200)
        trt = rng.normal(10.5, 2, size=200)

        # Should not emit method-choice warning (auto mode skips it)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp = Experiment(ctrl, trt)
            exp.run()

    def test_verbose_experiment(self):
        """verbose() should print info during Experiment.run()."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 2, size=100)
        trt = rng.normal(10.5, 2, size=100)

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with verbose():
                Experiment(ctrl, trt).run()
        finally:
            sys.stderr = old_stderr
        output = captured.getvalue()
        assert "Auto-detected metric" in output
        assert "Selected method" in output


# ── integration: CUPED ───────────────────────────────────────────────


class TestCUPEDAdvisory:
    """Test advisory integration with CUPED class."""

    def test_low_variance_reduction_warns(self):
        """CUPED with near-zero correlation should warn about low reduction."""
        rng = np.random.default_rng(42)
        # pre and post are uncorrelated
        ctrl_pre = rng.normal(10, 2, size=100)
        ctrl_post = rng.normal(10, 2, size=100)
        trt_pre = rng.normal(10, 2, size=100)
        trt_post = rng.normal(10.5, 2, size=100)

        cuped = CUPED()
        with pytest.warns(RuntimeWarning):
            cuped.fit_transform(ctrl_post, trt_post, ctrl_pre, trt_pre)

    def test_good_variance_reduction_no_extra_warning(self):
        """CUPED with good correlation should not warn about low reduction."""
        rng = np.random.default_rng(42)
        ctrl_pre = rng.normal(10, 2, size=100)
        ctrl_post = ctrl_pre + rng.normal(0, 0.5, size=100)
        trt_pre = rng.normal(10, 2, size=100)
        trt_post = trt_pre + 0.5 + rng.normal(0, 0.5, size=100)

        cuped = CUPED()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cuped.fit_transform(ctrl_post, trt_post, ctrl_pre, trt_pre)
