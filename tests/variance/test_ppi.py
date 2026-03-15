"""Tests for PredictionPoweredInference (PPI)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.variance.ppi import PredictionPoweredInference


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestPPIBasic:
    """Basic functionality tests."""

    def test_known_mean(self, rng):
        """PPI should estimate the true mean."""
        true_mean = 5.0
        n_labeled = 100
        n_unlabeled = 5000

        y = rng.normal(true_mean, 1, n_labeled)
        f_labeled = y + rng.normal(0, 0.3, n_labeled)
        f_unlabeled = rng.normal(true_mean, 1, n_unlabeled) + rng.normal(
            0, 0.3, n_unlabeled
        )

        r = PredictionPoweredInference().fit(y, f_labeled, f_unlabeled)
        assert abs(r.mean_estimate - true_mean) < 1.0

    def test_ci_contains_true_mean(self, rng):
        """CI should contain the true mean."""
        true_mean = 3.0
        n_labeled = 200
        n_unlabeled = 10000

        y = rng.normal(true_mean, 1, n_labeled)
        f_labeled = y + rng.normal(0, 0.2, n_labeled)
        f_unlabeled = rng.normal(true_mean, 1, n_unlabeled) + rng.normal(
            0, 0.2, n_unlabeled
        )

        r = PredictionPoweredInference().fit(y, f_labeled, f_unlabeled)
        assert r.ci_lower < true_mean < r.ci_upper

    def test_se_decreases_with_more_unlabeled(self, rng):
        """SE should decrease with more unlabeled data."""
        y = rng.normal(5, 1, 100)
        f_labeled = y + rng.normal(0, 0.3, 100)

        f_small = rng.normal(5, 1, 1000) + rng.normal(0, 0.3, 1000)
        f_large = rng.normal(5, 1, 50000) + rng.normal(0, 0.3, 50000)

        r_small = PredictionPoweredInference().fit(y, f_labeled, f_small)
        r_large = PredictionPoweredInference().fit(y, f_labeled, f_large)
        assert r_large.se <= r_small.se

    def test_n_counts(self, rng):
        """n_labeled and n_unlabeled should match input sizes."""
        n_l, n_u = 50, 2000
        y = rng.normal(0, 1, n_l)
        fl = rng.normal(0, 1, n_l)
        fu = rng.normal(0, 1, n_u)

        r = PredictionPoweredInference().fit(y, fl, fu)
        assert r.n_labeled == n_l
        assert r.n_unlabeled == n_u

    def test_perfect_predictions(self, rng):
        """With perfect predictions, PPI should still work."""
        y = rng.normal(5, 1, 100)
        f_labeled = y.copy()  # perfect
        f_unlabeled = rng.normal(5, 1, 5000)

        r = PredictionPoweredInference().fit(y, f_labeled, f_unlabeled)
        assert abs(r.mean_estimate - 5.0) < 1.0

    def test_ci_order(self, rng):
        y = rng.normal(0, 1, 50)
        fl = rng.normal(0, 1, 50)
        fu = rng.normal(0, 1, 500)

        r = PredictionPoweredInference().fit(y, fl, fu)
        assert r.ci_lower < r.ci_upper

    def test_to_dict(self, rng):
        y = rng.normal(0, 1, 50)
        fl = rng.normal(0, 1, 50)
        fu = rng.normal(0, 1, 500)
        r = PredictionPoweredInference().fit(y, fl, fu)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "mean_estimate" in d

    def test_repr(self, rng):
        y = rng.normal(0, 1, 50)
        fl = rng.normal(0, 1, 50)
        fu = rng.normal(0, 1, 500)
        r = PredictionPoweredInference().fit(y, fl, fu)
        assert "PPIResult" in repr(r)


class TestPPIValidation:
    """Validation and error tests."""

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            PredictionPoweredInference(alpha=0.0)

    def test_mismatched_labeled(self, rng):
        with pytest.raises(ValueError, match="same length"):
            PredictionPoweredInference().fit(
                [1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0, 3.0]
            )

    def test_too_few_labeled(self):
        with pytest.raises(ValueError, match="at least"):
            PredictionPoweredInference().fit([1.0], [1.0], [1.0, 2.0, 3.0])

    def test_non_array_input(self):
        with pytest.raises(TypeError):
            PredictionPoweredInference().fit("bad", [1, 2, 3], [1, 2, 3])
