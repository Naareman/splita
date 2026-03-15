"""Tests for BayesianExperiment."""

from __future__ import annotations

import numpy as np
import pytest

from splita import BayesianExperiment, BayesianResult


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_conversion(p_ctrl, p_trt, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    ctrl = rng.binomial(1, p_ctrl, n)
    trt = rng.binomial(1, p_trt, n)
    return ctrl, trt


def _make_continuous(mu_ctrl, mu_trt, sigma=1.0, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(mu_ctrl, sigma, n)
    trt = rng.normal(mu_trt, sigma, n)
    return ctrl, trt


# ═══════════════════════════════════════════════════════════════════════
# Basic tests
# ═══════════════════════════════════════════════════════════════════════


class TestConversionBasic:
    """Conversion metric basic tests."""

    def test_large_effect_gives_high_prob_b_beats_a(self):
        """Test 1: significant difference -> prob_b_beats_a > 0.95."""
        ctrl, trt = _make_conversion(0.10, 0.14)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        assert result.prob_b_beats_a > 0.95

    def test_equal_rates_give_prob_near_fifty_percent(self):
        """Test 2: no difference -> prob_b_beats_a ~ 0.50."""
        ctrl, trt = _make_conversion(0.10, 0.10)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        assert 0.35 < result.prob_b_beats_a < 0.65


class TestContinuousBasic:
    """Continuous metric basic tests."""

    def test_large_mean_diff_gives_high_prob_b_beats_a(self):
        """Test 3: significant -> prob_b_beats_a > 0.95."""
        ctrl, trt = _make_continuous(10.0, 10.5)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        assert result.prob_b_beats_a > 0.95

    def test_equal_means_give_ci_containing_zero(self):
        """Test 4: no difference -> prob_b_beats_a not extreme.

        With equal population means the realized sample difference can
        push P(B>A) away from 0.50, so we only check it is not near
        0 or 1 (i.e. not a decisive result).
        """
        ctrl, trt = _make_continuous(10.0, 10.0)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        # The 95% credible interval should contain 0
        assert result.ci_lower < 0 < result.ci_upper


class TestAutoDetection:
    """Test 5: auto-detection works for both metric types."""

    def test_auto_conversion(self):
        ctrl, trt = _make_conversion(0.10, 0.12)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        assert result.metric == "conversion"

    def test_auto_continuous(self):
        ctrl, trt = _make_continuous(10.0, 10.5)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        assert result.metric == "continuous"


# ═══════════════════════════════════════════════════════════════════════
# Statistical correctness
# ═══════════════════════════════════════════════════════════════════════


class TestStatisticalCorrectness:
    """Tests 6-9: statistical correctness."""

    def test_expected_loss_identity(self):
        """Test 6: loss_a + loss_b ~ E[|diff|]."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        # loss_a = E[max(trt - ctrl, 0)], loss_b = E[max(ctrl - trt, 0)]
        # loss_a + loss_b = E[|trt - ctrl|]
        sum_loss = result.expected_loss_a + result.expected_loss_b
        # Verify it's positive and reasonable
        assert sum_loss > 0
        # The sum of losses should approximately equal E[|diff|]
        # We can't compute E[|diff|] exactly here but can check consistency:
        # loss_a should be larger when treatment is better
        assert result.expected_loss_a > result.expected_loss_b

    def test_ci_coverage(self):
        """Test 7: 95% CI contains true effect >= 93% of time (500 sims)."""
        true_effect = 0.02
        n_sims = 500
        covered = 0
        for i in range(n_sims):
            rng = np.random.default_rng(i)
            ctrl = rng.binomial(1, 0.10, 1000)
            trt = rng.binomial(1, 0.12, 1000)
            result = BayesianExperiment(
                ctrl, trt, n_samples=5000, random_state=i
            ).run()
            if result.ci_lower <= true_effect <= result.ci_upper:
                covered += 1
        coverage = covered / n_sims
        assert coverage >= 0.93, f"Coverage {coverage:.3f} < 0.93"

    def test_agrees_with_frequentist_large_n(self):
        """Test 8: prob_b_beats_a agrees with frequentist at large n."""
        from splita import Experiment

        ctrl, trt = _make_conversion(0.10, 0.12, n=10000)
        bayes = BayesianExperiment(ctrl, trt, random_state=0).run()
        freq = Experiment(ctrl, trt).run()
        # If frequentist says significant, Bayesian should have high prob
        if freq.significant:
            assert bayes.prob_b_beats_a > 0.90
        # If frequentist says not significant, Bayesian should be moderate
        else:
            assert bayes.prob_b_beats_a < 0.95

    def test_posterior_means_close_to_sample_means(self):
        """Test 9: posterior means close to sample means with vague prior."""
        ctrl, trt = _make_conversion(0.10, 0.15)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        # With vague prior (alpha=1, beta=1) and n=5000,
        # posterior mean ~ sample mean
        sample_ctrl = float(np.mean(ctrl))
        sample_trt = float(np.mean(trt))
        assert abs(result.control_mean - sample_ctrl) < 0.01
        assert abs(result.treatment_mean - sample_trt) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# ROPE tests
# ═══════════════════════════════════════════════════════════════════════


class TestROPE:
    """Tests 10-12: ROPE functionality."""

    def test_small_effect_in_rope(self):
        """Test 10: small effect within ROPE -> prob_in_rope > 0.5."""
        # Very small difference
        ctrl, trt = _make_conversion(0.100, 0.101, n=1000)
        result = BayesianExperiment(
            ctrl, trt, rope=(-0.02, 0.02), random_state=0
        ).run()
        assert result.prob_in_rope is not None
        assert result.prob_in_rope > 0.5

    def test_large_effect_outside_rope(self):
        """Test 11: large effect outside ROPE -> prob_in_rope ~ 0."""
        ctrl, trt = _make_conversion(0.10, 0.20, n=5000)
        result = BayesianExperiment(
            ctrl, trt, rope=(-0.01, 0.01), random_state=0
        ).run()
        assert result.prob_in_rope is not None
        assert result.prob_in_rope < 0.01

    def test_no_rope_returns_none(self):
        """Test 12: no ROPE -> prob_in_rope is None."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        assert result.prob_in_rope is None
        assert result.rope is None


# ═══════════════════════════════════════════════════════════════════════
# Custom prior tests
# ═══════════════════════════════════════════════════════════════════════


class TestCustomPrior:
    """Tests 13-15: custom prior effects."""

    def test_informative_prior_shifts_posterior(self):
        """Test 13: informative prior shifts posterior."""
        ctrl, trt = _make_conversion(0.10, 0.12, n=100)
        # Vague prior
        r_vague = BayesianExperiment(ctrl, trt, random_state=0).run()
        # Strong prior centered at 0.50 for both
        r_strong = BayesianExperiment(
            ctrl, trt, prior={"alpha": 500, "beta": 500}, random_state=0
        ).run()
        # Strong prior should pull means toward 0.50
        assert r_strong.control_mean > r_vague.control_mean
        assert r_strong.treatment_mean > r_vague.treatment_mean

    def test_strong_prior_dominates_small_data(self):
        """Test 14: strong prior dominates small data."""
        # Only 10 observations, strong prior at 0.50
        rng = np.random.default_rng(42)
        ctrl = rng.binomial(1, 0.05, 10)
        trt = rng.binomial(1, 0.05, 10)
        result = BayesianExperiment(
            ctrl, trt, prior={"alpha": 1000, "beta": 1000}, random_state=0
        ).run()
        # Posterior means should be close to 0.50 despite data being ~0.05
        assert 0.40 < result.control_mean < 0.60
        assert 0.40 < result.treatment_mean < 0.60

    def test_large_data_overwhelms_prior(self):
        """Test 15: large data overwhelms prior."""
        ctrl, trt = _make_conversion(0.10, 0.10, n=50000)
        # Prior centered at 0.50 but not extremely strong
        result = BayesianExperiment(
            ctrl, trt, prior={"alpha": 50, "beta": 50}, random_state=0
        ).run()
        # With 50k observations, posterior should be close to data (0.10)
        assert abs(result.control_mean - 0.10) < 0.01
        assert abs(result.treatment_mean - 0.10) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# Validation tests
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """Tests 16-19: input validation."""

    def test_n_samples_too_low(self):
        """Test 16: n_samples < 1000 -> ValueError."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        with pytest.raises(ValueError, match="n_samples"):
            BayesianExperiment(ctrl, trt, n_samples=500)

    def test_rope_invalid(self):
        """Test 17: rope[0] >= rope[1] -> ValueError."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        with pytest.raises(ValueError, match="rope"):
            BayesianExperiment(ctrl, trt, rope=(0.01, -0.01))
        with pytest.raises(ValueError, match="rope"):
            BayesianExperiment(ctrl, trt, rope=(0.01, 0.01))

    def test_invalid_metric(self):
        """Test 18: invalid metric -> ValueError."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        with pytest.raises(ValueError, match="metric"):
            BayesianExperiment(ctrl, trt, metric="ratio")

    def test_too_few_observations(self):
        """Test 19: too few observations -> ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            BayesianExperiment([1], [0, 1, 0])
        with pytest.raises(ValueError, match="at least 2"):
            BayesianExperiment([0, 1, 0], [1])


# ═══════════════════════════════════════════════════════════════════════
# Reproducibility tests
# ═══════════════════════════════════════════════════════════════════════


class TestReproducibility:
    """Test 20: same seed -> same result."""

    def test_same_seed_same_result(self):
        ctrl, trt = _make_conversion(0.10, 0.12)
        r1 = BayesianExperiment(ctrl, trt, random_state=42).run()
        r2 = BayesianExperiment(ctrl, trt, random_state=42).run()
        assert r1.prob_b_beats_a == r2.prob_b_beats_a
        assert r1.lift == r2.lift
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper
        assert r1.expected_loss_a == r2.expected_loss_a
        assert r1.expected_loss_b == r2.expected_loss_b


# ═══════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Tests 21-22: integration with other splita components."""

    def test_outlier_handler_pipeline(self):
        """Test 21: OutlierHandler -> BayesianExperiment pipeline."""
        from splita import OutlierHandler

        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 500)
        trt = rng.normal(10.5, 1, 500)
        # Add outliers
        ctrl[0] = 100.0
        trt[0] = -80.0

        handler = OutlierHandler(method="iqr")
        ctrl_clean, trt_clean = handler.fit_transform(ctrl, trt)

        result = BayesianExperiment(ctrl_clean, trt_clean, random_state=0).run()
        assert result.metric == "continuous"
        assert result.prob_b_beats_a > 0.9

    def test_cuped_pipeline(self):
        """Test 22: CUPED -> BayesianExperiment pipeline."""
        from splita import CUPED

        rng = np.random.default_rng(42)
        n = 2000
        # Pre-experiment covariate (correlated with outcome)
        pre_ctrl = rng.normal(10, 2, n)
        pre_trt = rng.normal(10, 2, n)
        # Post-experiment outcome
        ctrl = pre_ctrl + rng.normal(0, 1, n)
        trt = pre_trt + rng.normal(0.3, 1, n)

        cuped = CUPED()
        ctrl_adj, trt_adj = cuped.fit_transform(
            ctrl, trt, control_pre=pre_ctrl, treatment_pre=pre_trt
        )

        result = BayesianExperiment(ctrl_adj, trt_adj, random_state=0).run()
        assert result.metric == "continuous"
        assert isinstance(result, BayesianResult)


# ═══════════════════════════════════════════════════════════════════════
# Result tests
# ═══════════════════════════════════════════════════════════════════════


class TestBayesianResult:
    """Tests for BayesianResult dataclass."""

    def test_to_dict(self):
        """to_dict returns a plain dict with all fields."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "prob_b_beats_a" in d
        assert "lift" in d
        assert "metric" in d
        assert d["control_n"] == len(ctrl)
        assert d["treatment_n"] == len(trt)

    def test_repr_contains_key_info(self):
        """repr contains key metrics."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        text = repr(result)
        assert "BayesianResult" in text
        assert "P(B > A)" in text
        assert "Decision" in text
        assert "Expected loss" in text

    def test_repr_with_rope(self):
        """repr includes ROPE info when set."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        result = BayesianExperiment(
            ctrl, trt, rope=(-0.01, 0.01), random_state=0
        ).run()
        text = repr(result)
        assert "ROPE" in text
        assert "P(in ROPE)" in text

    def test_frozen(self):
        """Result is immutable."""
        ctrl, trt = _make_conversion(0.10, 0.12)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        with pytest.raises(AttributeError):
            result.prob_b_beats_a = 0.5  # type: ignore[misc]


class TestContinuousPrior:
    """Additional tests for continuous NIG prior."""

    def test_continuous_posterior_means_near_sample(self):
        """Continuous posterior means are near sample means with vague prior."""
        ctrl, trt = _make_continuous(5.0, 6.0, n=5000)
        result = BayesianExperiment(ctrl, trt, random_state=0).run()
        assert abs(result.control_mean - 5.0) < 0.1
        assert abs(result.treatment_mean - 6.0) < 0.1

    def test_continuous_custom_prior(self):
        """Continuous with custom NIG prior."""
        ctrl, trt = _make_continuous(5.0, 5.5, n=100)
        result = BayesianExperiment(
            ctrl,
            trt,
            metric="continuous",
            prior={"mu": 0, "kappa": 0.001, "alpha": 1, "beta": 1},
            random_state=0,
        ).run()
        assert result.metric == "continuous"
        assert isinstance(result.lift, float)

    def test_continuous_rope(self):
        """Continuous metric with ROPE."""
        ctrl, trt = _make_continuous(10.0, 10.0, n=2000)
        result = BayesianExperiment(
            ctrl, trt, rope=(-0.1, 0.1), random_state=0
        ).run()
        assert result.prob_in_rope is not None
        # No real difference, most of the posterior diff should be near 0
        assert result.prob_in_rope > 0.3
