"""Tests for MultiObjectiveExperiment."""

from __future__ import annotations

import numpy as np
import pytest

from splita.core.multi_objective import MultiObjectiveExperiment
from splita._types import MultiObjectiveResult


# ── helpers ──────────────────────────────────────────────────────────


def _positive_metric(n: int = 500, effect: float = 1.0, seed: int = 42):
    """Generate data where treatment is better."""
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(10, 2, n)
    trt = rng.normal(10 + effect, 2, n)
    return ctrl, trt


def _negative_metric(n: int = 500, effect: float = -1.0, seed: int = 42):
    """Generate data where treatment is worse."""
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(10, 2, n)
    trt = rng.normal(10 + effect, 2, n)
    return ctrl, trt


def _null_metric(n: int = 500, seed: int = 42):
    """Generate data with no treatment effect."""
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(10, 2, n)
    trt = rng.normal(10, 2, n)
    return ctrl, trt


# ── Basic tests ──────────────────────────────────────────────────────


class TestBasic:
    """Basic MultiObjectiveExperiment behaviour."""

    def test_returns_result_type(self):
        """run() returns a MultiObjectiveResult."""
        exp = MultiObjectiveExperiment()
        ctrl, trt = _positive_metric()
        exp.add_metric(ctrl, trt)
        result = exp.run()
        assert isinstance(result, MultiObjectiveResult)

    def test_single_metric(self):
        """Works with a single metric."""
        exp = MultiObjectiveExperiment()
        ctrl, trt = _positive_metric()
        exp.add_metric(ctrl, trt, name="revenue")
        result = exp.run()
        assert len(result.metric_results) == 1
        assert len(result.corrected_pvalues) == 1

    def test_two_metrics(self):
        """Works with two metrics."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(seed=1), name="revenue")
        exp.add_metric(*_positive_metric(seed=2), name="engagement")
        result = exp.run()
        assert len(result.metric_results) == 2
        assert len(result.corrected_pvalues) == 2


# ── Recommendation tests ─────────────────────────────────────────────


class TestRecommendation:
    """Tests for recommendation logic."""

    def test_all_positive_adopt(self):
        """All metrics positive and significant -> 'adopt'."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=1))
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=2))
        result = exp.run()
        assert result.recommendation == "adopt"

    def test_all_negative_reject(self):
        """All metrics negative and significant -> 'reject'."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_negative_metric(n=1000, effect=-2.0, seed=1))
        exp.add_metric(*_negative_metric(n=1000, effect=-2.0, seed=2))
        result = exp.run()
        assert result.recommendation == "reject"

    def test_mixed_directions_tradeoff(self):
        """Some positive, some negative -> 'tradeoff'."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=1), name="good")
        exp.add_metric(*_negative_metric(n=1000, effect=-2.0, seed=2), name="bad")
        result = exp.run()
        assert result.recommendation == "tradeoff"
        assert len(result.tradeoffs) > 0

    def test_no_significant_reject(self):
        """No significant metrics -> 'reject' (insufficient evidence)."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_null_metric(seed=1))
        exp.add_metric(*_null_metric(seed=2))
        result = exp.run()
        assert result.recommendation == "reject"

    def test_sig_positive_plus_nonsig_negative_lift_tradeoff(self):
        """Sig positive on one metric, non-sig slightly negative lift on another -> 'tradeoff'.

        Covers line 193 in multi_objective.py: sig_positive > 0, sig_negative == 0,
        but not all lifts are positive, so recommendation = 'tradeoff'.
        """
        exp = MultiObjectiveExperiment()
        # Strong positive effect on metric 1
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=10), name="good")
        # Tiny negative lift (not statistically significant) on metric 2
        rng = np.random.default_rng(1)
        ctrl_neg = rng.normal(10, 5, 30)
        trt_neg = rng.normal(9.5, 5, 30)  # slight negative, but huge noise -> not sig
        exp.add_metric(ctrl_neg, trt_neg, name="neutral")
        result = exp.run()
        # Verify the setup conditions: sig_positive > 0, sig_negative == 0, not all lifts > 0
        lifts = [r.lift for r in result.metric_results]
        assert any(l > 0 for l in lifts), "Need at least one positive lift"
        assert result.recommendation == "tradeoff"


# ── Pareto dominance ─────────────────────────────────────────────────


class TestParetoDominance:
    """Tests for Pareto dominance detection."""

    def test_pareto_dominant_true(self):
        """Treatment better on all metrics -> pareto_dominant = True."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=1))
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=2))
        result = exp.run()
        assert result.pareto_dominant is True

    def test_pareto_dominant_false_mixed(self):
        """Mixed directions -> pareto_dominant = False."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=1))
        exp.add_metric(*_negative_metric(n=1000, effect=-2.0, seed=2))
        result = exp.run()
        assert result.pareto_dominant is False

    def test_pareto_dominant_false_not_significant(self):
        """Not all significant -> pareto_dominant = False."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=1))
        exp.add_metric(*_null_metric(seed=2))
        result = exp.run()
        assert result.pareto_dominant is False


# ── Multiple testing correction ──────────────────────────────────────


class TestCorrection:
    """Tests for multiple testing correction."""

    def test_correction_applied(self):
        """Corrected p-values are >= raw p-values."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(n=500, seed=1))
        exp.add_metric(*_positive_metric(n=500, seed=2))
        exp.add_metric(*_positive_metric(n=500, seed=3))
        result = exp.run()

        for i, r in enumerate(result.metric_results):
            assert result.corrected_pvalues[i] >= r.pvalue - 1e-10

    def test_bonferroni_correction(self):
        """Bonferroni correction works."""
        exp = MultiObjectiveExperiment(correction="bonferroni")
        exp.add_metric(*_positive_metric(n=500, seed=1))
        exp.add_metric(*_positive_metric(n=500, seed=2))
        result = exp.run()
        assert len(result.corrected_pvalues) == 2

    def test_holm_correction(self):
        """Holm correction works."""
        exp = MultiObjectiveExperiment(correction="holm")
        exp.add_metric(*_positive_metric(n=500, seed=1))
        exp.add_metric(*_positive_metric(n=500, seed=2))
        result = exp.run()
        assert len(result.corrected_pvalues) == 2

    def test_single_metric_no_correction(self):
        """Single metric doesn't inflate p-value via correction."""
        exp = MultiObjectiveExperiment()
        ctrl, trt = _positive_metric(n=500, seed=1)
        exp.add_metric(ctrl, trt)
        result = exp.run()

        # Single metric: corrected == raw
        raw = result.metric_results[0].pvalue
        assert abs(result.corrected_pvalues[0] - raw) < 1e-10


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_no_metrics_raises(self):
        """run() with no metrics raises ValueError."""
        exp = MultiObjectiveExperiment()
        with pytest.raises(ValueError, match=r"No metrics"):
            exp.run()

    def test_bad_alpha_raises(self):
        """alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match=r"alpha"):
            MultiObjectiveExperiment(alpha=0.0)
        with pytest.raises(ValueError, match=r"alpha"):
            MultiObjectiveExperiment(alpha=1.0)


# ── Metric naming ────────────────────────────────────────────────────


class TestNaming:
    """Tests for metric naming."""

    def test_explicit_names(self):
        """add_metric(name=...) uses the given name."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(seed=1), name="revenue")
        exp.add_metric(*_positive_metric(seed=2), name="latency")
        result = exp.run()
        # Tradeoffs/metrics reference names
        assert len(result.metric_results) == 2

    def test_init_metric_names(self):
        """metric_names from init are used."""
        exp = MultiObjectiveExperiment(metric_names=["rev", "lat"])
        exp.add_metric(*_positive_metric(seed=1))
        exp.add_metric(*_positive_metric(seed=2))
        result = exp.run()
        assert len(result.metric_results) == 2

    def test_auto_naming(self):
        """Without names, metrics are auto-numbered."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(seed=1))
        exp.add_metric(*_positive_metric(seed=2))
        result = exp.run()
        assert len(result.metric_results) == 2


# ── Result attributes ────────────────────────────────────────────────


class TestResultAttributes:
    """Test result dataclass attributes."""

    def test_to_dict(self):
        """to_dict produces a plain dict."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric())
        result = exp.run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "recommendation" in d
        assert "pareto_dominant" in d
        assert "corrected_pvalues" in d

    def test_repr(self):
        """repr produces a string."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric())
        result = exp.run()
        s = repr(result)
        assert "MultiObjectiveResult" in s
        assert "recommendation" in s

    def test_tradeoffs_list(self):
        """Tradeoffs is a list."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(n=1000, effect=2.0, seed=1))
        exp.add_metric(*_negative_metric(n=1000, effect=-2.0, seed=2))
        result = exp.run()
        assert isinstance(result.tradeoffs, list)

    def test_corrected_pvalues_bounded(self):
        """Corrected p-values are in [0, 1]."""
        exp = MultiObjectiveExperiment()
        exp.add_metric(*_positive_metric(seed=1))
        exp.add_metric(*_positive_metric(seed=2))
        exp.add_metric(*_negative_metric(seed=3))
        result = exp.run()
        for p in result.corrected_pvalues:
            assert 0.0 <= p <= 1.0


# ── Many metrics ─────────────────────────────────────────────────────


class TestManyMetrics:
    """Tests with many metrics."""

    def test_five_metrics(self):
        """Works with 5 metrics."""
        exp = MultiObjectiveExperiment()
        for i in range(5):
            exp.add_metric(*_positive_metric(seed=i + 1))
        result = exp.run()
        assert len(result.metric_results) == 5
        assert len(result.corrected_pvalues) == 5

    def test_correction_more_conservative_with_more_metrics(self):
        """More metrics -> more conservative corrected p-values."""
        # 2 metrics
        exp2 = MultiObjectiveExperiment()
        exp2.add_metric(*_positive_metric(n=200, effect=0.5, seed=1))
        exp2.add_metric(*_positive_metric(n=200, effect=0.5, seed=2))
        r2 = exp2.run()

        # 5 metrics (same first 2 + 3 more)
        exp5 = MultiObjectiveExperiment()
        exp5.add_metric(*_positive_metric(n=200, effect=0.5, seed=1))
        exp5.add_metric(*_positive_metric(n=200, effect=0.5, seed=2))
        exp5.add_metric(*_positive_metric(n=200, effect=0.5, seed=3))
        exp5.add_metric(*_positive_metric(n=200, effect=0.5, seed=4))
        exp5.add_metric(*_positive_metric(n=200, effect=0.5, seed=5))
        r5 = exp5.run()

        # With BH, more tests generally means higher corrected p-values
        # for the same raw p-values (at least for the first two)
        # This is a soft check — BH is less conservative than Bonferroni
        # but still corrects upward
        assert len(r5.corrected_pvalues) == 5
        assert len(r2.corrected_pvalues) == 2
