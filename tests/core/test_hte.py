"""Tests for HTEEstimator — heterogeneous treatment effect estimation (M12)."""

from __future__ import annotations

import numpy as np
import pytest

from splita import HTEEstimator, HTEResult


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture()
def heterogeneous_data():
    """Data with clear heterogeneity: CATE depends on X[:, 0]."""
    rng = np.random.default_rng(42)
    n = 300
    X_ctrl = rng.normal(size=(n, 3))
    X_trt = rng.normal(size=(n, 3))
    # Control: Y = X0 * 0.5
    y_ctrl = X_ctrl[:, 0] * 0.5 + rng.normal(0, 0.1, n)
    # Treatment: Y = X0 * 2.0  (heterogeneous effect depends on X0)
    y_trt = X_trt[:, 0] * 2.0 + rng.normal(0, 0.1, n)
    return y_ctrl, y_trt, X_ctrl, X_trt


@pytest.fixture()
def uniform_data():
    """Data with no heterogeneity: constant CATE."""
    rng = np.random.default_rng(99)
    n = 300
    X_ctrl = rng.normal(size=(n, 3))
    X_trt = rng.normal(size=(n, 3))
    y_ctrl = rng.normal(5, 1, n)
    y_trt = rng.normal(5, 1, n)  # no effect
    return y_ctrl, y_trt, X_ctrl, X_trt


# ─── T-learner tests ─────────────────────────────────────────────


class TestTLearner:
    def test_detects_heterogeneity(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator(method="t_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = hte.result()
        assert isinstance(result, HTEResult)
        assert result.method == "t_learner"
        # CATE should have nonzero std (heterogeneity)
        assert result.cate_std > 0.1

    def test_returns_correct_length(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator(method="t_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = hte.result()
        # CATE estimates for all data (control + treatment)
        assert len(result.cate_estimates) == len(y_ctrl) + len(y_trt)

    def test_predict_new_data(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator(method="t_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
        X_new = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        preds = hte.predict(X_new)
        assert preds.shape == (2,)
        # X0=1 should have higher CATE than X0=-1
        assert preds[0] > preds[1]

    def test_no_heterogeneity_flat_cate(self, uniform_data):
        y_ctrl, y_trt, X_ctrl, X_trt = uniform_data
        hte = HTEEstimator(method="t_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = hte.result()
        # std should be low when there's no real heterogeneity
        assert result.cate_std < 1.0

    def test_top_features_present(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator(method="t_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = hte.result()
        # Ridge has coef_, so top_features should be populated
        assert result.top_features is not None
        assert len(result.top_features) > 0


# ─── S-learner tests ─────────────────────────────────────────────


class TestSLearner:
    def test_works(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator(method="s_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = hte.result()
        assert result.method == "s_learner"
        assert len(result.cate_estimates) == len(y_ctrl) + len(y_trt)

    def test_predict_new_data(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator(method="s_learner").fit(y_ctrl, y_trt, X_ctrl, X_trt)
        X_new = np.array([[1.0, 0.0, 0.0]])
        preds = hte.predict(X_new)
        assert preds.shape == (1,)


# ─── Validation tests ────────────────────────────────────────────


class TestHTEValidation:
    def test_invalid_method(self):
        with pytest.raises(ValueError, match="must be one of"):
            HTEEstimator(method="invalid")

    def test_mismatched_control_lengths(self):
        hte = HTEEstimator()
        with pytest.raises(ValueError, match="same number of rows"):
            hte.fit(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
                np.array([[1.0], [2.0]]),  # 2 rows, but 3 outcomes
                np.array([[1.0], [2.0]]),
            )

    def test_mismatched_treatment_lengths(self):
        hte = HTEEstimator()
        with pytest.raises(ValueError, match="same number of rows"):
            hte.fit(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([[1.0], [2.0]]),
                np.array([[1.0], [2.0]]),  # 2 rows, but 3 outcomes
            )

    def test_mismatched_feature_counts(self):
        hte = HTEEstimator()
        with pytest.raises(ValueError, match="same number of features"):
            hte.fit(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 features
                np.array([[1.0], [2.0]]),  # 1 feature
            )

    def test_predict_before_fit(self):
        hte = HTEEstimator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            hte.predict(np.array([[1.0]]))

    def test_result_before_fit(self):
        hte = HTEEstimator()
        with pytest.raises(RuntimeError, match="must be fitted"):
            hte.result()

    def test_control_too_short(self):
        hte = HTEEstimator()
        with pytest.raises(ValueError, match="at least"):
            hte.fit(
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([[1.0]]),
                np.array([[1.0], [2.0]]),
            )


# ─── Serialisation & repr ────────────────────────────────────────


class TestHTESerialisation:
    def test_to_dict(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator().fit(y_ctrl, y_trt, X_ctrl, X_trt)
        d = hte.result().to_dict()
        assert isinstance(d, dict)
        assert "mean_cate" in d
        assert "cate_estimates" in d

    def test_repr(self, heterogeneous_data):
        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator().fit(y_ctrl, y_trt, X_ctrl, X_trt)
        r = repr(hte.result())
        assert "HTEResult" in r
        assert "t_learner" in r


# ─── Custom estimator test ───────────────────────────────────────


class TestCustomEstimator:
    def test_with_decision_tree(self, heterogeneous_data):
        from sklearn.tree import DecisionTreeRegressor

        y_ctrl, y_trt, X_ctrl, X_trt = heterogeneous_data
        hte = HTEEstimator(
            estimator=DecisionTreeRegressor(max_depth=3, random_state=0),
        ).fit(y_ctrl, y_trt, X_ctrl, X_trt)
        result = hte.result()
        # DecisionTree has feature_importances_
        assert result.top_features is not None
