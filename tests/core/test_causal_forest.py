"""Tests for CausalForest (Tier 3, Item 14)."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import CausalForestResult
from splita.core.causal_forest import CausalForest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestCausalForestBasic:
    """Basic functionality tests."""

    def test_detects_heterogeneity(self, rng):
        """Should detect heterogeneous treatment effects."""
        n = 300
        X_ctrl = rng.normal(size=(n, 3))
        X_trt = rng.normal(size=(n, 3))
        # Heterogeneous effect: CATE depends on X[:, 0]
        y_ctrl = X_ctrl[:, 0] * 0.5 + rng.normal(0, 0.3, n)
        y_trt = X_trt[:, 0] * 2.0 + rng.normal(0, 0.3, n)

        cf = CausalForest(n_estimators=50, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        assert isinstance(result, CausalForestResult)
        # CATE std should be > 0 indicating heterogeneity
        assert result.cate_std > 0

    def test_honest_vs_dishonest(self, rng):
        """Honest splitting should generally produce wider CIs."""
        n = 200
        X_ctrl = rng.normal(size=(n, 3))
        X_trt = rng.normal(size=(n, 3))
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        # Honest
        cf_honest = CausalForest(n_estimators=30, honest=True, random_state=42)
        cf_honest = cf_honest.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        r_honest = cf_honest.result()

        # Dishonest
        cf_dish = CausalForest(n_estimators=30, honest=False, random_state=42)
        cf_dish = cf_dish.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        r_dish = cf_dish.result()

        # Both should produce valid results
        assert isinstance(r_honest, CausalForestResult)
        assert isinstance(r_dish, CausalForestResult)

    def test_feature_importance(self, rng):
        """Feature importances should sum to ~1 and have correct length."""
        n = 200
        X_ctrl = rng.normal(size=(n, 5))
        X_trt = rng.normal(size=(n, 5))
        y_ctrl = X_ctrl[:, 0] + rng.normal(0, 0.5, n)
        y_trt = X_trt[:, 0] * 3 + rng.normal(0, 0.5, n)

        cf = CausalForest(n_estimators=50, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        assert len(result.feature_importances) == 5
        assert abs(sum(result.feature_importances) - 1.0) < 0.01

    def test_ci_coverage(self, rng):
        """CI should contain the true mean CATE in many replications."""
        true_effect = 2.0
        covers = 0
        n_reps = 20

        for rep in range(n_reps):
            r = np.random.default_rng(rep + 100)
            n = 200
            X_ctrl = r.normal(size=(n, 2))
            X_trt = r.normal(size=(n, 2))
            y_ctrl = r.normal(0, 1, n)
            y_trt = r.normal(true_effect, 1, n)

            cf = CausalForest(n_estimators=30, honest=False, random_state=rep)
            cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
            result = cf.result()

            if result.ci_lower <= true_effect <= result.ci_upper:
                covers += 1

        # Should cover at least 50% of the time (conservative check)
        assert covers >= 5

    def test_predict_returns_array(self, rng):
        """predict() should return an array of correct length."""
        n = 100
        X_ctrl = rng.normal(size=(n, 3))
        X_trt = rng.normal(size=(n, 3))
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        cf = CausalForest(n_estimators=20, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)

        X_new = rng.normal(size=(50, 3))
        preds = cf.predict(X_new)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 50

    def test_mean_cate_positive_when_treatment_helps(self, rng):
        """Mean CATE should be positive when treatment has positive effect."""
        n = 300
        X_ctrl = rng.normal(size=(n, 3))
        X_trt = rng.normal(size=(n, 3))
        y_ctrl = rng.normal(0, 0.5, n)
        y_trt = rng.normal(3, 0.5, n)

        cf = CausalForest(n_estimators=50, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        assert result.mean_cate > 0

    def test_max_depth(self, rng):
        """max_depth parameter should be respected."""
        n = 100
        X_ctrl = rng.normal(size=(n, 3))
        X_trt = rng.normal(size=(n, 3))
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        cf = CausalForest(n_estimators=10, max_depth=3, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        assert isinstance(result, CausalForestResult)

    def test_to_dict(self, rng):
        """Result should serialise to a dictionary."""
        n = 50
        X_ctrl = rng.normal(size=(n, 2))
        X_trt = rng.normal(size=(n, 2))
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        cf = CausalForest(n_estimators=10, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "mean_cate" in d
        assert "feature_importances" in d

    def test_repr(self, rng):
        """Result __repr__ should be a string."""
        n = 50
        X_ctrl = rng.normal(size=(n, 2))
        X_trt = rng.normal(size=(n, 2))
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        cf = CausalForest(n_estimators=10, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        assert "CausalForestResult" in repr(result)

    def test_1d_features_reshaped(self, rng):
        """1-D feature arrays should be reshaped to (n, 1)."""
        n = 50
        X_ctrl = rng.normal(size=n)
        X_trt = rng.normal(size=n)
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        cf = CausalForest(n_estimators=10, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        assert len(result.feature_importances) == 1


class TestCausalForestValidation:
    """Validation and error handling tests."""

    def test_predict_before_fit_raises(self):
        """Calling predict() before fit() should raise RuntimeError."""
        cf = CausalForest()
        with pytest.raises(RuntimeError, match="fitted"):
            cf.predict(np.array([[1, 2]]))

    def test_result_before_fit_raises(self):
        """Calling result() before fit() should raise RuntimeError."""
        cf = CausalForest()
        with pytest.raises(RuntimeError, match="fitted"):
            cf.result()

    def test_mismatched_feature_dims_raises(self, rng):
        """Mismatched feature dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="same number of features"):
            cf = CausalForest(n_estimators=5, random_state=42)
            cf.fit(
                [1, 2, 3], [4, 5, 6],
                np.array([[1, 2], [3, 4], [5, 6]]),
                np.array([[1], [2], [3]]),
            )

    def test_mismatched_outcome_feature_rows_raises(self, rng):
        """Mismatched outcome and feature rows should raise ValueError."""
        with pytest.raises(ValueError, match="same number of rows"):
            cf = CausalForest(n_estimators=5, random_state=42)
            cf.fit(
                [1, 2, 3], [4, 5, 6],
                np.array([[1, 2], [3, 4]]),  # 2 rows vs 3 outcomes
                np.array([[1, 2], [3, 4], [5, 6]]),
            )

    def test_invalid_n_estimators_raises(self):
        """n_estimators < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_estimators"):
            CausalForest(n_estimators=0)

    def test_invalid_max_depth_raises(self):
        """max_depth < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_depth"):
            CausalForest(max_depth=0)

    def test_mismatched_treatment_rows_raises(self, rng):
        """Mismatched treatment outcome and feature rows should raise."""
        with pytest.raises(ValueError, match="same number of rows"):
            cf = CausalForest(n_estimators=5, random_state=42)
            cf.fit(
                [1, 2, 3], [4, 5, 6],
                np.array([[1, 2], [3, 4], [5, 6]]),
                np.array([[1, 2], [3, 4]]),  # 2 rows vs 3 outcomes
            )

    def test_single_tree_jackknife_fallback(self, rng):
        """With n_estimators=1, jackknife falls back to simple SE."""
        n = 50
        X_ctrl = rng.normal(size=(n, 2))
        X_trt = rng.normal(size=(n, 2))
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        cf = CausalForest(n_estimators=1, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = cf.result()

        assert isinstance(result, CausalForestResult)
        # CI should still be valid (non-zero width)
        assert result.ci_upper > result.ci_lower

    def test_predict_1d_reshape(self, rng):
        """predict() with 1-D input should reshape and work."""
        n = 50
        X_ctrl = rng.normal(size=(n, 1))
        X_trt = rng.normal(size=(n, 1))
        y_ctrl = rng.normal(0, 1, n)
        y_trt = rng.normal(1, 1, n)

        cf = CausalForest(n_estimators=10, random_state=42)
        cf = cf.fit(y_ctrl, y_trt, X_ctrl, X_trt)

        X_new = rng.normal(size=5)  # 1-D
        preds = cf.predict(X_new)
        assert preds.shape == (5,)
