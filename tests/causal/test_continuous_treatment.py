"""Tests for ContinuousTreatmentEffect (dose-response)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.continuous_treatment import ContinuousTreatmentEffect


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestDoseResponseBasic:
    """Basic functionality tests."""

    def test_linear_relationship(self, rng):
        """Should detect a positive linear dose-response."""
        n = 300
        dose = rng.uniform(0, 10, n)
        outcome = 2.0 * dose + rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect().fit(outcome, dose).result()

        assert r.slope_at_mean > 0
        assert r.r_squared > 0.5
        assert len(r.dose_response_curve) > 0

    def test_quadratic_relationship(self, rng):
        """Should capture a non-linear dose-response with optimal dose."""
        n = 500
        dose = rng.uniform(0, 10, n)
        # Optimal dose is at 5 (parabola peaks at x=5)
        outcome = -(dose - 5.0) ** 2 + 25 + rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect().fit(outcome, dose).result()

        # Optimal dose should be near 5
        assert abs(r.optimal_dose - 5.0) < 2.0
        assert len(r.dose_response_curve) > 0

    def test_no_relationship(self, rng):
        """When there's no dose-response, R-squared should be low."""
        n = 200
        dose = rng.uniform(0, 10, n)
        outcome = rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect().fit(outcome, dose).result()

        assert abs(r.r_squared) < 0.1

    def test_negative_slope(self, rng):
        """Should detect a negative dose-response."""
        n = 300
        dose = rng.uniform(0, 10, n)
        outcome = -1.5 * dose + rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect().fit(outcome, dose).result()

        assert r.slope_at_mean < 0

    def test_curve_length_matches_grid(self, rng):
        """Dose-response curve should have n_grid points."""
        n_grid = 30
        n = 200
        dose = rng.uniform(0, 10, n)
        outcome = dose + rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect(n_grid=n_grid).fit(outcome, dose).result()

        assert len(r.dose_response_curve) == n_grid

    def test_curve_tuples(self, rng):
        """Dose-response curve should be list of (dose, effect) tuples."""
        n = 200
        dose = rng.uniform(0, 10, n)
        outcome = dose + rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect().fit(outcome, dose).result()

        for point in r.dose_response_curve:
            assert len(point) == 2
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)

    def test_with_covariates(self, rng):
        """Should handle covariates correctly."""
        n = 300
        x = rng.normal(0, 1, n)
        dose = rng.uniform(0, 10, n)
        outcome = 1.0 * dose + 2.0 * x + rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect().fit(
            outcome, dose, covariates=x
        ).result()

        assert r.n == n
        assert len(r.dose_response_curve) > 0

    def test_custom_bandwidth(self, rng):
        """Should accept a custom bandwidth."""
        n = 200
        dose = rng.uniform(0, 10, n)
        outcome = dose + rng.normal(0, 1, n)

        r = ContinuousTreatmentEffect(bandwidth=2.0).fit(outcome, dose).result()

        assert r.n == n


class TestDoseResponseValidation:
    """Validation and error handling tests."""

    def test_too_few_observations(self, rng):
        """Should reject arrays with fewer than 10 elements."""
        with pytest.raises(ValueError, match="at least 10"):
            ContinuousTreatmentEffect().fit(
                [1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
            )

    def test_mismatched_lengths(self, rng):
        """Arrays must have the same length."""
        with pytest.raises(ValueError, match="same length"):
            ContinuousTreatmentEffect().fit(
                rng.normal(0, 1, 20),
                rng.uniform(0, 10, 15),
            )

    def test_result_before_fit(self):
        """Calling result() before fit() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="fitted"):
            ContinuousTreatmentEffect().result()

    def test_n_grid_too_small(self):
        """n_grid < 3 should raise ValueError."""
        with pytest.raises(ValueError, match="n_grid"):
            ContinuousTreatmentEffect(n_grid=2)

    def test_covariates_wrong_rows(self, rng):
        """Covariates with wrong number of rows should raise ValueError."""
        n = 20
        with pytest.raises(ValueError, match="same number of rows"):
            ContinuousTreatmentEffect().fit(
                rng.normal(0, 1, n),
                rng.uniform(0, 10, n),
                covariates=rng.normal(0, 1, (n + 5, 2)),
            )

    def test_to_dict(self, rng):
        """to_dict should return a plain dictionary."""
        n = 50
        r = ContinuousTreatmentEffect().fit(
            rng.normal(0, 1, n), rng.uniform(0, 10, n)
        ).result()

        d = r.to_dict()
        assert isinstance(d, dict)
        assert "optimal_dose" in d
        assert "dose_response_curve" in d

    def test_repr(self, rng):
        """repr should return a formatted string."""
        n = 50
        r = ContinuousTreatmentEffect().fit(
            rng.normal(0, 1, n), rng.uniform(0, 10, n)
        ).result()
        assert "DoseResponseResult" in repr(r)

    def test_chaining(self, rng):
        """fit() should return self for chaining."""
        cte = ContinuousTreatmentEffect()
        n = 50
        ret = cte.fit(rng.normal(0, 1, n), rng.uniform(0, 10, n))
        assert ret is cte
