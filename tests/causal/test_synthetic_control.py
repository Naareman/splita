"""Tests for SyntheticControl (M17: Causal Inference — Synthetic Control)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.synthetic_control import SyntheticControl


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_sc_data(rng, n_pre=10, n_post=5, n_donors=3, effect=5.0):
    """Helper to create synthetic control test data.

    Donor 0 is designed to perfectly match the treated unit pre-treatment.
    """
    treated_pre = np.arange(1, n_pre + 1, dtype=float) + rng.normal(0, 0.1, n_pre)
    treated_post = np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) + effect + rng.normal(0, 0.1, n_post)

    donors_pre = np.zeros((n_pre, n_donors))
    donors_post = np.zeros((n_post, n_donors))

    # Donor 0: matches treated pre very closely
    donors_pre[:, 0] = np.arange(1, n_pre + 1, dtype=float) + rng.normal(0, 0.1, n_pre)
    donors_post[:, 0] = np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) + rng.normal(0, 0.1, n_post)

    # Donor 1: offset
    donors_pre[:, 1] = np.arange(1, n_pre + 1, dtype=float) + 10
    donors_post[:, 1] = np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) + 10

    # Donor 2: different trend
    donors_pre[:, 2] = np.arange(1, n_pre + 1, dtype=float) * 0.5
    donors_post[:, 2] = np.arange(n_pre + 1, n_pre + n_post + 1, dtype=float) * 0.5

    return treated_pre, treated_post, donors_pre, donors_post


class TestSyntheticControlBasic:
    """Basic functionality tests."""

    def test_positive_effect_detected(self, rng):
        """A clear positive treatment effect should be recovered."""
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(
            rng, effect=5.0
        )
        sc = SyntheticControl()
        sc.fit(treated_pre, treated_post, donors_pre, donors_post)
        r = sc.result()
        assert r.effect > 0

    def test_no_effect(self, rng):
        """When there is no effect, the estimated effect should be near zero."""
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(
            rng, effect=0.0
        )
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert abs(r.effect) < 2.0

    def test_weights_sum_to_one(self, rng):
        """Weights should sum to 1."""
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert abs(sum(r.weights) - 1.0) < 1e-6

    def test_weights_non_negative(self, rng):
        """All weights should be non-negative."""
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert all(w >= 0 for w in r.weights)

    def test_best_donor_gets_highest_weight(self, rng):
        """The closest-matching donor should receive the highest weight."""
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        # Donor 0 is designed to match the treated unit
        assert r.weights[0] > r.weights[1]
        assert r.weights[0] > r.weights[2]

    def test_pre_treatment_rmse_low_for_good_fit(self, rng):
        """Pre-treatment RMSE should be low when donors can match treated."""
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert r.pre_treatment_rmse < 1.0

    def test_effect_series_length(self, rng):
        """Effect series should have length equal to post-treatment periods."""
        n_post = 7
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(
            rng, n_post=n_post
        )
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert len(r.effect_series) == n_post

    def test_donor_contributions_dict(self, rng):
        """Donor contributions should map each donor index to its weight."""
        n_donors = 4
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(
            rng, n_donors=n_donors
        )
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert len(r.donor_contributions) == n_donors
        assert set(r.donor_contributions.keys()) == set(range(n_donors))

    def test_single_donor(self, rng):
        """Should work with a single donor (weight must be 1.0)."""
        treated_pre = np.arange(1, 11, dtype=float)
        treated_post = np.arange(11, 16, dtype=float) + 3.0
        donors_pre = np.arange(1, 11, dtype=float).reshape(-1, 1)
        donors_post = np.arange(11, 16, dtype=float).reshape(-1, 1)

        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert abs(r.weights[0] - 1.0) < 1e-6
        assert abs(r.effect - 3.0) < 0.1

    def test_fit_returns_self(self, rng):
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        sc = SyntheticControl()
        result = sc.fit(treated_pre, treated_post, donors_pre, donors_post)
        assert result is sc

    def test_negative_effect(self, rng):
        """A negative treatment effect should yield negative effect."""
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(
            rng, effect=-5.0
        )
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert r.effect < 0


class TestSyntheticControlValidation:
    """Tests for input validation."""

    def test_result_before_fit(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            SyntheticControl().result()

    def test_mismatched_pre_periods(self, rng):
        with pytest.raises(ValueError, match="same number of time periods"):
            SyntheticControl().fit(
                np.arange(10, dtype=float),
                np.arange(5, dtype=float),
                np.zeros((8, 2)),  # 8 != 10
                np.zeros((5, 2)),
            )

    def test_mismatched_post_periods(self, rng):
        with pytest.raises(ValueError, match="same number of time periods"):
            SyntheticControl().fit(
                np.arange(10, dtype=float),
                np.arange(5, dtype=float),
                np.zeros((10, 2)),
                np.zeros((3, 2)),  # 3 != 5
            )

    def test_mismatched_donors(self, rng):
        with pytest.raises(ValueError, match="same number of donors"):
            SyntheticControl().fit(
                np.arange(10, dtype=float),
                np.arange(5, dtype=float),
                np.zeros((10, 2)),
                np.zeros((5, 3)),  # 3 != 2
            )

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            SyntheticControl(alpha=1.5)

    def test_too_few_pre_periods(self, rng):
        with pytest.raises(ValueError, match="at least"):
            SyntheticControl().fit(
                np.array([1.0]),
                np.arange(5, dtype=float),
                np.zeros((1, 2)),
                np.zeros((5, 2)),
            )

    def test_non_array_donors(self, rng):
        """Should accept list-of-lists for donors."""
        treated_pre = np.arange(1, 6, dtype=float)
        treated_post = np.arange(6, 9, dtype=float) + 2.0
        donors_pre = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        donors_post = [[6, 7], [7, 8], [8, 9]]
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert len(r.weights) == 2


class TestSyntheticControlResult:
    """Tests for result properties."""

    def test_to_dict(self, rng):
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "effect" in d
        assert "weights" in d
        assert "pre_treatment_rmse" in d
        assert "effect_series" in d

    def test_repr(self, rng):
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        assert "SyntheticControlResult" in repr(r)

    def test_frozen_dataclass(self, rng):
        treated_pre, treated_post, donors_pre, donors_post = _make_sc_data(rng)
        r = SyntheticControl().fit(
            treated_pre, treated_post, donors_pre, donors_post
        ).result()
        with pytest.raises(AttributeError):
            r.effect = 999  # type: ignore[misc]
