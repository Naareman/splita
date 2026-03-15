"""Tests for SurrogateIndex (Tier 3, Item 12)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import SurrogateIndexResult
from splita.causal.surrogate_index import SurrogateIndex


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestSurrogateIndexBasic:
    """Basic functionality tests."""

    def test_valid_multi_surrogate(self, rng):
        """Multiple correlated surrogates should predict long-term effect."""
        n = 300
        # Short-term outcomes correlated with long-term
        X_short = rng.normal(0, 1, (2 * n, 3))
        y_long = 2 * X_short[:, 0] + 0.5 * X_short[:, 1] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        # Treatment shifts short-term outcomes
        X_short[n:, 0] += 1.0

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        assert isinstance(result, SurrogateIndexResult)
        assert result.n_surrogates == 3
        assert result.predicted_effect > 0  # positive treatment effect
        assert result.surrogate_r2 > 0.3
        assert result.is_valid is True

    def test_single_surrogate(self, rng):
        """Single surrogate should work."""
        n = 200
        X_short = rng.normal(0, 1, (2 * n, 1))
        y_long = 3 * X_short[:, 0] + rng.normal(0, 0.3, 2 * n)
        treatment = np.array([0] * n + [1] * n)
        X_short[n:, 0] += 0.5

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        assert result.n_surrogates == 1
        assert result.predicted_effect > 0

    def test_weak_surrogates_warning(self, rng):
        """Weak surrogates should trigger a warning and is_valid=False."""
        n = 200
        # Surrogates uncorrelated with long-term outcome
        X_short = rng.normal(0, 1, (2 * n, 3))
        y_long = rng.normal(0, 1, 2 * n)  # independent of X_short
        treatment = np.array([0] * n + [1] * n)

        with pytest.warns(RuntimeWarning, match="below 0.3"):
            si = SurrogateIndex(random_state=42)
            si = si.fit(X_short, y_long, treatment)

        result = si.predict(X_short[:n], X_short[n:])
        assert result.is_valid is False

    def test_prediction_direction(self, rng):
        """Predicted effect should match actual treatment direction."""
        n = 300
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] * 2 + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        # Treatment increases first surrogate
        X_short[n:, 0] += 2.0

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        assert result.predicted_effect > 0

    def test_ci_contains_estimate(self, rng):
        """CI should contain the point estimate."""
        n = 300
        X_short = rng.normal(0, 1, (2 * n, 3))
        y_long = X_short[:, 0] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)
        X_short[n:, 0] += 1.0

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        assert result.ci_lower <= result.predicted_effect <= result.ci_upper

    def test_se_positive(self, rng):
        """Standard error should be positive."""
        n = 200
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        assert result.se > 0

    def test_custom_cv(self, rng):
        """Custom CV should work."""
        n = 200
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(cv=3, random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        assert isinstance(result, SurrogateIndexResult)

    def test_to_dict(self, rng):
        """Result should serialise to a dictionary."""
        n = 100
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "predicted_effect" in d
        assert "surrogate_r2" in d

    def test_repr(self, rng):
        """Result __repr__ should be a string."""
        n = 100
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])

        assert "SurrogateIndexResult" in repr(result)

    def test_1d_short_term_reshaped(self, rng):
        """1-D short_term_outcomes should be reshaped to (n, 1)."""
        n = 100
        X_short = rng.normal(0, 1, 2 * n)
        y_long = X_short * 2 + rng.normal(0, 0.3, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n].reshape(-1, 1), X_short[n:].reshape(-1, 1))

        assert result.n_surrogates == 1


class TestSurrogateIndexValidation:
    """Validation and error handling tests."""

    def test_predict_before_fit_raises(self):
        """Calling predict() before fit() should raise RuntimeError."""
        si = SurrogateIndex()
        with pytest.raises(RuntimeError, match="fitted"):
            si.predict(np.array([[1]]), np.array([[2]]))

    def test_mismatched_rows_raises(self):
        """Mismatched short_term and long_term rows should raise ValueError."""
        si = SurrogateIndex()
        with pytest.raises(ValueError, match="same number of rows"):
            si.fit(
                np.array([[1, 2], [3, 4]]),
                np.array([1, 2, 3]),
                np.array([0, 1]),
            )

    def test_mismatched_features_predict_raises(self, rng):
        """Mismatched features at predict time should raise ValueError."""
        n = 50
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)

        with pytest.raises(ValueError, match="same number of features"):
            si.predict(rng.normal(0, 1, (n, 3)), rng.normal(0, 1, (n, 3)))

    def test_invalid_cv_raises(self):
        """CV < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="cv"):
            SurrogateIndex(cv=1)

    def test_3d_short_term_raises(self):
        """Line 133: 3-D short_term_outcomes should raise ValueError."""
        si = SurrogateIndex()
        with pytest.raises(ValueError, match="1-D or 2-D"):
            si.fit(
                np.zeros((10, 2, 3)),
                np.zeros(10),
                np.array([0] * 5 + [1] * 5),
            )

    def test_mismatched_treatment_rows_raises(self):
        """Line 154: mismatched treatment length should raise ValueError."""
        si = SurrogateIndex()
        with pytest.raises(ValueError, match="same number of rows"):
            si.fit(
                np.zeros((10, 2)),
                np.zeros(10),
                np.array([0, 1, 0]),  # length 3 != 10
            )

    def test_1d_predict_inputs_reshaped(self, rng):
        """Lines 240, 242: 1-D predict inputs get reshaped."""
        n = 100
        X_short = rng.normal(0, 1, 2 * n)
        y_long = X_short * 2 + rng.normal(0, 0.3, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        # Pass 1-D arrays to predict (should auto-reshape)
        result = si.predict(X_short[:n], X_short[n:])
        assert result.n_surrogates == 1

    def test_mismatched_trt_features_predict_raises(self, rng):
        """Line 253: mismatched trt features at predict should raise ValueError."""
        n = 50
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(random_state=42)
        si = si.fit(X_short, y_long, treatment)
        # ctrl has correct features but trt has wrong number
        with pytest.raises(ValueError, match="same number of features"):
            si.predict(rng.normal(0, 1, (n, 2)), rng.normal(0, 1, (n, 3)))

    def test_custom_estimator(self, rng):
        """Custom estimator passed via constructor should be cloned per fold."""
        from sklearn.linear_model import LinearRegression

        n = 100
        X_short = rng.normal(0, 1, (2 * n, 2))
        y_long = X_short[:, 0] * 2 + rng.normal(0, 0.5, 2 * n)
        treatment = np.array([0] * n + [1] * n)

        si = SurrogateIndex(estimator=LinearRegression(), random_state=42)
        si = si.fit(X_short, y_long, treatment)
        result = si.predict(X_short[:n], X_short[n:])
        assert isinstance(result, SurrogateIndexResult)
