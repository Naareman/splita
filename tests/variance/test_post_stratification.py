"""Tests for PostStratification."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import PostStratResult
from splita.variance.post_stratification import PostStratification


# ── helpers ──────────────────────────────────────────────────────────


def _make_stratified_data(
    n_per_stratum: int = 200,
    effect: float = 0.5,
    strata_means: list[float] | None = None,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    if strata_means is None:
        strata_means = [5.0, 20.0]

    ctrl_list, trt_list = [], []
    strata_c, strata_t = [], []

    for i, mu in enumerate(strata_means):
        ctrl_list.append(rng.normal(mu, 1.0, n_per_stratum))
        trt_list.append(rng.normal(mu + effect, 1.0, n_per_stratum))
        strata_c.extend([i] * n_per_stratum)
        strata_t.extend([i] * n_per_stratum)

    return (
        np.concatenate(ctrl_list),
        np.concatenate(trt_list),
        np.array(strata_c),
        np.array(strata_t),
    )


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic PostStratification behaviour."""

    def test_detects_known_effect(self):
        """Detects a known stratified effect."""
        ctrl, trt, sc, st = _make_stratified_data(effect=0.5)
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        assert isinstance(result, PostStratResult)
        assert result.significant
        np.testing.assert_allclose(result.ate, 0.5, atol=0.15)

    def test_no_effect(self):
        """Zero effect is not significant."""
        ctrl, trt, sc, st = _make_stratified_data(effect=0.0)
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        assert not result.significant

    def test_result_frozen(self):
        """Result is frozen."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        with pytest.raises(AttributeError):
            result.ate = 999.0  # type: ignore[misc]

    def test_to_dict(self):
        """Result serializes to dict."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "variance_reduction" in d

    def test_repr(self):
        """Result has string representation."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        s = repr(result)
        assert "PostStratResult" in s


# ── Variance reduction ───────────────────────────────────────────────


class TestVarianceReduction:
    """Variance reduction properties."""

    def test_reduces_variance(self):
        """Post-stratification reduces variance compared to naive."""
        ctrl, trt, sc, st = _make_stratified_data(strata_means=[5.0, 50.0])
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        assert result.variance_reduction > 0.0

    def test_ci_covers_true_effect(self):
        """CI covers the true effect."""
        ctrl, trt, sc, st = _make_stratified_data(effect=0.5, seed=123)
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        assert result.ci_lower < 0.5 < result.ci_upper

    def test_variance_reduction_nonneg(self):
        """Variance reduction is non-negative."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        assert result.variance_reduction >= 0.0


# ── Strata ───────────────────────────────────────────────────────────


class TestStrata:
    """Stratum handling."""

    def test_n_strata(self):
        """Correct number of strata reported."""
        ctrl, trt, sc, st = _make_stratified_data(strata_means=[1.0, 5.0, 10.0])
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        assert result.n_strata == 3

    def test_single_stratum(self):
        """Single stratum works."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(10.5, 1, 200)
        sc = np.array([0] * 200)
        st = np.array([0] * 200)
        result = PostStratification().fit_transform(ctrl, trt, sc, st)
        assert result.n_strata == 1


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_alpha_out_of_range(self):
        """alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            PostStratification(alpha=1.5)

    def test_no_overlapping_strata(self):
        """No common strata raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="share at least one label"):
            PostStratification().fit_transform(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                np.array(["A"] * 50),
                np.array(["B"] * 50),
            )

    def test_mismatched_strata_length(self):
        """Strata length must match data length."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="same length"):
            PostStratification().fit_transform(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                np.array(["A"] * 30),
                np.array(["A"] * 50),
            )

    def test_strata_not_array_like(self):
        """Non-array strata raises TypeError."""
        rng = np.random.default_rng(42)
        with pytest.raises(TypeError, match="array-like"):
            PostStratification().fit_transform(
                rng.normal(0, 1, 50),
                rng.normal(0, 1, 50),
                42,
                np.array(["A"] * 50),
            )

    def test_too_few_in_stratum(self):
        """Stratum with <2 obs per group raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = np.concatenate([rng.normal(0, 1, 1), rng.normal(0, 1, 50)])
        trt = np.concatenate([rng.normal(0, 1, 50), rng.normal(0, 1, 50)])
        sc = np.array(["A"] * 1 + ["B"] * 50)
        st = np.array(["A"] * 50 + ["B"] * 50)
        with pytest.raises(ValueError, match="at least 2"):
            PostStratification().fit_transform(ctrl, trt, sc, st)

    def test_control_too_short(self):
        """Control with <2 elements raises ValueError."""
        with pytest.raises(ValueError, match="at least"):
            PostStratification().fit_transform([1.0], [1.0, 2.0], ["A"], ["A", "A"])


# ── E2E scenario ─────────────────────────────────────────────────────


class TestE2E:
    """End-to-end scenario."""

    def test_multistratum_experiment(self):
        """Full post-stratification workflow with 4 strata."""
        ctrl, trt, sc, st = _make_stratified_data(
            n_per_stratum=300,
            effect=0.3,
            strata_means=[2.0, 8.0, 15.0, 30.0],
            seed=99,
        )
        result = PostStratification(alpha=0.05).fit_transform(ctrl, trt, sc, st)
        assert result.n_strata == 4
        assert result.significant
        assert result.variance_reduction > 0
        np.testing.assert_allclose(result.ate, 0.3, atol=0.15)
