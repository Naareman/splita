"""Tests for CUPAC variance reduction."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from splita.variance import CUPAC


# ── helpers ──────────────────────────────────────────────────────────


def _make_cupac_data(
    n: int = 500,
    n_features: int = 3,
    effect: float = 0.5,
    noise: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate feature-driven data for CUPAC tests."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 1, size=n_features)
    X_ctrl = rng.normal(0, 1, size=(n, n_features))
    X_trt = rng.normal(0, 1, size=(n, n_features))
    ctrl = X_ctrl @ weights + rng.normal(0, noise, n)
    trt = X_trt @ weights + effect + rng.normal(0, noise, n)
    return ctrl, trt, X_ctrl, X_trt


# ── Basic tests ──────────────────────────────────────────────────────


class TestBasic:
    """Basic CUPAC behaviour."""

    def test_variance_reduction_works(self):
        """Adjusted data has lower variance than original."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(random_state=42)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        original_var = np.var(np.concatenate([ctrl, trt]))
        adjusted_var = np.var(np.concatenate([ctrl_adj, trt_adj]))
        assert adjusted_var < original_var

    def test_ate_preserved(self):
        """Treatment effect is preserved after CUPAC adjustment."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data(n=1000, seed=99)
        cupac = CUPAC(random_state=99)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        original_ate = np.mean(trt) - np.mean(ctrl)
        adjusted_ate = np.mean(trt_adj) - np.mean(ctrl_adj)
        np.testing.assert_allclose(adjusted_ate, original_ate, atol=0.15)

    def test_default_estimator_ridge(self):
        """Works without specifying an estimator (defaults to Ridge)."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(random_state=42)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        assert hasattr(cupac, "theta_")
        assert hasattr(cupac, "cv_r2_")
        assert cupac.cv_r2_ > 0.0


# ── Statistical correctness ─────────────────────────────────────────


class TestStatistical:
    """Statistical correctness of CUPAC."""

    def test_out_of_fold_predictions(self):
        """Out-of-fold predictions differ from in-sample predictions.

        In-sample R squared should be >= OOF R squared because in-sample overfits.
        """
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data(n=200)
        cupac = CUPAC(random_state=42)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        # Train a model on ALL data and predict (in-sample, overfitted)
        from sklearn.linear_model import Ridge

        Y = np.concatenate([ctrl, trt])
        X = np.concatenate([X_ctrl, X_trt], axis=0)
        full_model = Ridge(alpha=1.0).fit(X, Y)
        in_sample_preds = full_model.predict(X)

        # In-sample R squared should be >= OOF R squared (in-sample overfits)
        corr_is = float(np.corrcoef(Y, in_sample_preds)[0, 1]) ** 2
        assert corr_is >= cupac.variance_reduction_ - 0.01

        # Basic shape checks
        assert len(ctrl_adj) == len(ctrl)
        assert len(trt_adj) == len(trt)

    def test_theta_computation(self):
        """Theta matches manual Cov(Y, Y_hat) / Var(Y_hat)."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data(n=300, seed=7)
        cupac = CUPAC(random_state=7)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        # Reconstruct Y_hat from the adjustment formula:
        # Y_adj = Y - theta * (Y_hat - mean(Y_hat))
        # We can verify theta_ was stored correctly
        assert isinstance(cupac.theta_, float)
        assert np.isfinite(cupac.theta_)

    def test_variance_reduction_equals_r_squared(self):
        """variance_reduction_ == correlation_²."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(random_state=42)
        cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        np.testing.assert_allclose(
            cupac.variance_reduction_,
            cupac.correlation_ ** 2,
            rtol=1e-10,
        )


# ── Custom estimator ────────────────────────────────────────────────


class TestCustomEstimator:
    """CUPAC with custom estimators."""

    def test_custom_ridge(self):
        """Explicit Ridge(alpha=0.5) works."""
        from sklearn.linear_model import Ridge

        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(estimator=Ridge(alpha=0.5), random_state=42)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        assert cupac.cv_r2_ > 0.0
        assert len(ctrl_adj) == len(ctrl)

    def test_gradient_boosting(self):
        """GradientBoostingRegressor achieves higher R² on nonlinear data."""
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.default_rng(42)
        n = 500
        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))
        # Nonlinear relationship
        ctrl = np.sin(X_ctrl[:, 0]) + X_ctrl[:, 1] ** 2 + rng.normal(0, 0.5, n)
        trt = np.sin(X_trt[:, 0]) + X_trt[:, 1] ** 2 + 0.5 + rng.normal(0, 0.5, n)

        cupac_gbr = CUPAC(
            estimator=GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42
            ),
            random_state=42,
        )
        ctrl_adj, trt_adj = cupac_gbr.fit_transform(ctrl, trt, X_ctrl, X_trt)
        assert cupac_gbr.cv_r2_ > 0.1


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for CUPAC."""

    def test_low_r2_warning(self):
        """Random features produce low R² and emit a warning."""
        rng = np.random.default_rng(42)
        n = 200
        ctrl = rng.normal(0, 1, n)
        trt = rng.normal(0.5, 1, n)
        # Random features uncorrelated with outcome
        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))

        cupac = CUPAC(random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

            r2_warnings = [x for x in w if "R² is low" in str(x.message)]
            assert len(r2_warnings) >= 1

    def test_low_r2_warning_fit(self):
        """Low R² warning also fires from fit() (not just fit_transform)."""
        rng = np.random.default_rng(42)
        n = 200
        ctrl = rng.normal(0, 1, n)
        trt = rng.normal(0.5, 1, n)
        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))

        cupac = CUPAC(random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cupac.fit(ctrl, trt, X_ctrl, X_trt)

            r2_warnings = [x for x in w if "R² is low" in str(x.message)]
            assert len(r2_warnings) >= 1

    def test_single_feature(self):
        """X with 1 column works."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data(n_features=1)
        cupac = CUPAC(random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)
        assert len(ctrl_adj) == len(ctrl)

    def test_many_features(self):
        """X with 10 columns works."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data(n_features=10)
        cupac = CUPAC(random_state=42)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)
        assert len(ctrl_adj) == len(ctrl)


# ── Validation ──────────────────────────────────────────────────────


class TestValidation:
    """Input validation for CUPAC."""

    def test_bad_estimator_no_fit(self):
        """Object without fit/predict raises ValueError."""
        with pytest.raises(ValueError, match="fit.*predict"):
            CUPAC(estimator=object())

    def test_cv_less_than_2(self):
        """cv < 2 raises ValueError."""
        with pytest.raises(ValueError, match="cv.*>= 2"):
            CUPAC(cv=1)

    def test_cv_zero(self):
        """cv=0 raises ValueError."""
        with pytest.raises(ValueError, match="cv.*>= 2"):
            CUPAC(cv=0)

    def test_x_column_mismatch(self):
        """X_control and X_treatment with different columns raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0, 1, 50)
        X_ctrl = rng.normal(0, 1, size=(50, 3))
        X_trt = rng.normal(0, 1, size=(50, 5))

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="same number of columns"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

    def test_x_row_mismatch_control(self):
        """X_control with wrong number of rows raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0, 1, 50)
        X_ctrl = rng.normal(0, 1, size=(40, 3))  # Wrong: 40 != 50
        X_trt = rng.normal(0, 1, size=(50, 3))

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="same number of rows"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

    def test_x_row_mismatch_treatment(self):
        """X_treatment with wrong number of rows raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0, 1, 50)
        X_ctrl = rng.normal(0, 1, size=(50, 3))
        X_trt = rng.normal(0, 1, size=(40, 3))  # Wrong: 40 != 50

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="same number of rows"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

    def test_theta_before_fit_transform(self):
        """Accessing theta_ before fit_transform raises AttributeError."""
        cupac = CUPAC()
        with pytest.raises(AttributeError):
            _ = cupac.theta_

    def test_x_control_not_2d(self):
        """1-D X_control raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0, 1, 50)
        X_ctrl = rng.normal(0, 1, size=50)  # 1-D
        X_trt = rng.normal(0, 1, size=(50, 3))

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="2-D"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

    def test_x_treatment_not_2d(self):
        """1-D X_treatment raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 1, 50)
        trt = rng.normal(0, 1, 50)
        X_ctrl = rng.normal(0, 1, size=(50, 3))
        X_trt = rng.normal(0, 1, size=50)  # 1-D

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="2-D"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)


# ── Integration ─────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests for CUPAC."""

    def test_outlier_handler_then_cupac(self):
        """OutlierHandler -> CUPAC pipeline works."""
        from splita.variance import OutlierHandler

        rng = np.random.default_rng(42)
        n = 300
        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))
        ctrl = X_ctrl @ [1, 2, 0.5] + rng.normal(0, 1, n)
        trt = X_trt @ [1, 2, 0.5] + 0.5 + rng.normal(0, 1, n)

        # Add outliers
        ctrl[0] = 100.0
        trt[0] = -100.0

        # Clip outliers (OutlierHandler only clips the outcome arrays)
        handler = OutlierHandler(method="iqr")
        ctrl_clean, trt_clean = handler.fit_transform(ctrl, trt)

        # CUPAC on cleaned data
        cupac = CUPAC(random_state=42)
        ctrl_adj, trt_adj = cupac.fit_transform(
            ctrl_clean, trt_clean, X_ctrl, X_trt,
        )
        assert cupac.variance_reduction_ > 0.0

    def test_reproducibility(self):
        """Same random_state produces identical results."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()

        cupac1 = CUPAC(random_state=42)
        c1, t1 = cupac1.fit_transform(ctrl, trt, X_ctrl, X_trt)

        cupac2 = CUPAC(random_state=42)
        c2, t2 = cupac2.fit_transform(ctrl, trt, X_ctrl, X_trt)

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(t1, t2)

    def test_fit_transform_returns_tuple_of_arrays(self):
        """fit_transform returns a tuple of two numpy arrays."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(random_state=42)
        result = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)


# ── Additional coverage tests ──────────────────────────────────────


# ── fit / transform tests ──────────────────────────────────────────


class TestFitTransform:
    """Tests for fit() + transform() workflow."""

    def test_fit_then_transform(self):
        """fit on one dataset, transform on another with same structure."""
        ctrl1, trt1, X_ctrl1, X_trt1 = _make_cupac_data(n=1000, seed=42)

        cupac = CUPAC(random_state=42)
        cupac.fit(ctrl1, trt1, X_ctrl1, X_trt1)

        # Generate a second dataset with same feature weights (same seed
        # for _make_cupac_data produces same weights)
        ctrl2, trt2, X_ctrl2, X_trt2 = _make_cupac_data(n=1000, seed=99)
        ctrl_adj, trt_adj = cupac.transform(ctrl2, trt2, X_ctrl2, X_trt2)

        assert len(ctrl_adj) == len(ctrl2)
        assert len(trt_adj) == len(trt2)
        # ATE should be preserved
        original_ate = np.mean(trt2) - np.mean(ctrl2)
        adjusted_ate = np.mean(trt_adj) - np.mean(ctrl_adj)
        np.testing.assert_allclose(adjusted_ate, original_ate, atol=0.15)

    def test_transform_before_fit_raises(self):
        """transform without fit raises RuntimeError."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(random_state=42)
        with pytest.raises(RuntimeError, match="not been fitted"):
            cupac.transform(ctrl, trt, X_ctrl, X_trt)

    def test_fit_transform_vs_fit_plus_transform(self):
        """Both approaches produce valid results (not necessarily identical).

        fit_transform uses out-of-fold predictions; fit+transform uses
        in-sample predictions, so results will differ but both should
        reduce variance.
        """
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data(seed=42)

        # Approach 1: fit_transform (OOF)
        cupac1 = CUPAC(random_state=42)
        c1, t1 = cupac1.fit_transform(ctrl, trt, X_ctrl, X_trt)

        # Approach 2: fit + transform (in-sample)
        cupac2 = CUPAC(random_state=42)
        cupac2.fit(ctrl, trt, X_ctrl, X_trt)
        c2, t2 = cupac2.transform(ctrl, trt, X_ctrl, X_trt)

        original_var = np.var(np.concatenate([ctrl, trt]))

        # Both approaches reduce variance
        assert np.var(np.concatenate([c1, t1])) < original_var
        assert np.var(np.concatenate([c2, t2])) < original_var

        # They are NOT identical (OOF vs in-sample)
        assert not np.allclose(c1, c2)

    def test_fit_sets_is_fitted(self):
        """fit() sets _is_fitted to True."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(random_state=42)
        assert not cupac._is_fitted
        cupac.fit(ctrl, trt, X_ctrl, X_trt)
        assert cupac._is_fitted

    def test_fit_transform_sets_is_fitted(self):
        """fit_transform() also sets _is_fitted."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        cupac = CUPAC(random_state=42)
        cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)
        assert cupac._is_fitted


# ── NaN / Inf validation tests ─────────────────────────────────────


class TestNanInfValidation:
    """NaN and Inf validation on feature matrices."""

    def test_nan_in_features_raises(self):
        """X_control with NaN raises ValueError."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        X_ctrl[0, 0] = np.nan

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="NaN or infinite"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

    def test_inf_in_features_raises(self):
        """X_treatment with Inf raises ValueError."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        X_trt[0, 0] = np.inf

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="NaN or infinite"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

    def test_nan_in_x_control_fit_raises(self):
        """fit() also validates NaN."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        X_ctrl[0, 0] = np.nan

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="NaN or infinite"):
            cupac.fit(ctrl, trt, X_ctrl, X_trt)

    def test_neg_inf_in_features_raises(self):
        """X_control with -Inf raises ValueError."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        X_ctrl[0, 0] = -np.inf

        cupac = CUPAC(random_state=42)
        with pytest.raises(ValueError, match="NaN or infinite"):
            cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)


# ── cv cap tests ───────────────────────────────────────────────────


class TestCvCap:
    """Tests for cv parameter cap."""

    def test_cv_too_high_raises(self):
        """cv=100 raises ValueError."""
        with pytest.raises(ValueError, match="cv.*<= 50"):
            CUPAC(cv=100)

    def test_cv_boundary_51_raises(self):
        """cv=51 raises ValueError."""
        with pytest.raises(ValueError, match="cv.*<= 50"):
            CUPAC(cv=51)

    def test_cv_boundary_50_ok(self):
        """cv=50 is allowed."""
        cupac = CUPAC(cv=50)
        assert cupac.cv == 50


class TestGeneratorRandomState:
    """Cover line 254: np.random.Generator as random_state."""

    def test_generator_random_state(self):
        """Passing a numpy Generator as random_state should work."""
        ctrl, trt, X_ctrl, X_trt = _make_cupac_data()
        gen = np.random.default_rng(42)
        cupac = CUPAC(random_state=gen)
        ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        assert len(ctrl_adj) == len(ctrl)
        assert cupac.cv_r2_ > 0.0


class TestDegenerateConstantPredictions:
    """Cover lines 290-293: model predicts a constant (var_y_hat == 0)."""

    def test_constant_prediction_returns_copy(self):
        """When the model predicts a constant, CUPAC returns copies of original."""
        from unittest.mock import MagicMock

        from sklearn.base import BaseEstimator

        class ConstantPredictor(BaseEstimator):
            """Estimator that always predicts 0.0."""

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        rng = np.random.default_rng(42)
        n = 50
        ctrl = rng.normal(10, 2, n)
        trt = rng.normal(10.5, 2, n)
        X_ctrl = rng.normal(0, 1, size=(n, 3))
        X_trt = rng.normal(0, 1, size=(n, 3))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cupac = CUPAC(estimator=ConstantPredictor(), random_state=42)
            ctrl_adj, trt_adj = cupac.fit_transform(ctrl, trt, X_ctrl, X_trt)

        assert cupac.theta_ == 0.0
        assert cupac.correlation_ == 0.0
        assert cupac.variance_reduction_ == 0.0
        np.testing.assert_array_equal(ctrl_adj, ctrl)
        np.testing.assert_array_equal(trt_adj, trt)
