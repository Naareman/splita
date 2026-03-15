"""Tests for StratifiedExperiment."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import StratifiedResult
from splita.core.stratified import StratifiedExperiment


# ── helpers ──────────────────────────────────────────────────────────


def _make_stratified_data(
    n_per_stratum: int = 200,
    effect: float = 0.5,
    strata_means: list[float] | None = None,
    seed: int = 42,
):
    """Generate stratified data with known effect."""
    rng = np.random.default_rng(seed)
    if strata_means is None:
        strata_means = [5.0, 10.0]

    ctrl_list, trt_list = [], []
    strata_c, strata_t = [], []

    for i, mu in enumerate(strata_means):
        label = f"stratum_{i}"
        ctrl_list.append(rng.normal(mu, 1.0, n_per_stratum))
        trt_list.append(rng.normal(mu + effect, 1.0, n_per_stratum))
        strata_c.extend([label] * n_per_stratum)
        strata_t.extend([label] * n_per_stratum)

    ctrl = np.concatenate(ctrl_list)
    trt = np.concatenate(trt_list)
    return ctrl, trt, np.array(strata_c), np.array(strata_t)


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic StratifiedExperiment behaviour."""

    def test_known_effect(self):
        """Detects a known stratified effect."""
        ctrl, trt, sc, st = _make_stratified_data(effect=0.5)
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert isinstance(result, StratifiedResult)
        assert result.significant
        np.testing.assert_allclose(result.ate, 0.5, atol=0.15)

    def test_no_effect(self):
        """Zero effect is not significant."""
        ctrl, trt, sc, st = _make_stratified_data(effect=0.0)
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert not result.significant
        np.testing.assert_allclose(result.ate, 0.0, atol=0.15)

    def test_result_is_frozen(self):
        """Result is a frozen dataclass."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        with pytest.raises(AttributeError):
            result.ate = 999.0  # type: ignore[misc]

    def test_to_dict(self):
        """Result serializes to a dictionary."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ate" in d
        assert "stratum_effects" in d

    def test_repr(self):
        """Result has a string representation."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        s = repr(result)
        assert "StratifiedResult" in s
        assert "ate" in s


# ── Variance reduction ───────────────────────────────────────────────


class TestVarianceReduction:
    """Stratification reduces variance compared to unstratified test."""

    def test_reduces_variance_vs_unstratified(self):
        """SE from stratified test is lower than naive pooled SE."""
        rng = np.random.default_rng(42)
        n = 200
        # Two strata with very different means -> stratification helps
        ctrl = np.concatenate([rng.normal(5, 1, n), rng.normal(50, 1, n)])
        trt = np.concatenate([rng.normal(5.5, 1, n), rng.normal(50.5, 1, n)])
        sc = np.array(["low"] * n + ["high"] * n)
        st = np.array(["low"] * n + ["high"] * n)

        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()

        # Unstratified SE (pooled)
        se_pooled = np.sqrt(
            np.var(ctrl, ddof=1) / len(ctrl) + np.var(trt, ddof=1) / len(trt)
        )
        assert result.se < se_pooled

    def test_ci_covers_true_effect(self):
        """CI from stratified test covers the true effect."""
        ctrl, trt, sc, st = _make_stratified_data(effect=0.5, seed=123)
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.ci_lower < 0.5 < result.ci_upper


# ── Multiple strata ──────────────────────────────────────────────────


class TestMultipleStrata:
    """Tests with varying numbers of strata."""

    def test_single_stratum(self):
        """Single stratum degenerates to unstratified analysis."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(10.5, 1, 200)
        sc = np.array(["all"] * 200)
        st = np.array(["all"] * 200)

        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.n_strata == 1
        np.testing.assert_allclose(result.ate, 0.5, atol=0.25)

    def test_three_strata(self):
        """Three strata work correctly."""
        ctrl, trt, sc, st = _make_stratified_data(
            strata_means=[5.0, 10.0, 20.0]
        )
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.n_strata == 3
        assert len(result.stratum_effects) == 3

    def test_five_strata(self):
        """Five strata work correctly."""
        ctrl, trt, sc, st = _make_stratified_data(
            strata_means=[1.0, 5.0, 10.0, 20.0, 50.0]
        )
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.n_strata == 5

    def test_integer_strata_labels(self):
        """Integer stratum labels work correctly."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 200)
        trt = rng.normal(10.5, 1, 200)
        sc = np.array([0] * 100 + [1] * 100)
        st = np.array([0] * 100 + [1] * 100)

        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.n_strata == 2


# ── Imbalanced strata ───────────────────────────────────────────────


class TestImbalanced:
    """Tests with imbalanced strata."""

    def test_unequal_strata_sizes(self):
        """Handles strata with different sizes."""
        rng = np.random.default_rng(42)
        # Stratum A: 50 ctrl, 50 trt; Stratum B: 200 ctrl, 200 trt
        ctrl = np.concatenate([
            rng.normal(5, 1, 50),
            rng.normal(10, 1, 200),
        ])
        trt = np.concatenate([
            rng.normal(5.5, 1, 50),
            rng.normal(10.5, 1, 200),
        ])
        sc = np.array(["A"] * 50 + ["B"] * 200)
        st = np.array(["A"] * 50 + ["B"] * 200)

        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.n_strata == 2
        # Weight of B should be larger
        effects = {e["stratum"]: e for e in result.stratum_effects}
        assert effects["B"]["weight"] > effects["A"]["weight"]

    def test_unequal_ctrl_trt_within_stratum(self):
        """Handles unequal control/treatment sizes within a stratum."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 100)
        trt = rng.normal(10.5, 1, 300)
        sc = np.array(["A"] * 100)
        st = np.array(["A"] * 300)

        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.n_strata == 1

    def test_partial_overlap_strata(self):
        """Only common strata are used."""
        rng = np.random.default_rng(42)
        ctrl = np.concatenate([rng.normal(5, 1, 50), rng.normal(10, 1, 50)])
        trt = np.concatenate([rng.normal(5.5, 1, 50), rng.normal(15, 1, 50)])
        sc = np.array(["A"] * 50 + ["B"] * 50)
        st = np.array(["A"] * 50 + ["C"] * 50)  # C is unique to treatment

        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        assert result.n_strata == 1  # only A is common


# ── Stratum effects ──────────────────────────────────────────────────


class TestStratumEffects:
    """Tests for per-stratum effect details."""

    def test_stratum_effects_structure(self):
        """Each stratum effect dict has the expected keys."""
        ctrl, trt, sc, st = _make_stratified_data()
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        for effect in result.stratum_effects:
            assert "stratum" in effect
            assert "ate" in effect
            assert "se" in effect
            assert "n_control" in effect
            assert "n_treatment" in effect
            assert "weight" in effect

    def test_weights_sum_to_one(self):
        """Stratum weights sum to 1.0."""
        ctrl, trt, sc, st = _make_stratified_data(
            strata_means=[5.0, 10.0, 20.0]
        )
        result = StratifiedExperiment(
            ctrl, trt, control_strata=sc, treatment_strata=st
        ).run()
        weights = [e["weight"] for e in result.stratum_effects]
        np.testing.assert_allclose(sum(weights), 1.0, atol=1e-10)


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_alpha_out_of_range(self):
        """alpha outside (0, 1) raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 50)
        trt = rng.normal(10.5, 1, 50)
        sc = np.array(["A"] * 50)
        st = np.array(["A"] * 50)
        with pytest.raises(ValueError, match="alpha"):
            StratifiedExperiment(
                ctrl, trt, control_strata=sc, treatment_strata=st, alpha=1.5
            )

    def test_mismatched_strata_length(self):
        """Strata array length must match data length."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 50)
        trt = rng.normal(10.5, 1, 50)
        with pytest.raises(ValueError, match="same length"):
            StratifiedExperiment(
                ctrl,
                trt,
                control_strata=np.array(["A"] * 30),
                treatment_strata=np.array(["A"] * 50),
            )

    def test_no_overlapping_strata(self):
        """No common strata raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 50)
        trt = rng.normal(10.5, 1, 50)
        with pytest.raises(ValueError, match="share at least one label"):
            StratifiedExperiment(
                ctrl,
                trt,
                control_strata=np.array(["A"] * 50),
                treatment_strata=np.array(["B"] * 50),
            )

    def test_stratum_too_few_control(self):
        """Stratum with <2 control obs raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = np.concatenate([rng.normal(10, 1, 1), rng.normal(5, 1, 50)])
        trt = rng.normal(10.5, 1, 51)
        sc = np.array(["A"] * 1 + ["B"] * 50)
        st = np.array(["A"] * 1 + ["B"] * 50)
        with pytest.raises(ValueError, match="fewer than 2 control"):
            StratifiedExperiment(
                ctrl, trt, control_strata=sc, treatment_strata=st
            ).run()

    def test_strata_not_array_like(self):
        """Non-array strata raises TypeError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 50)
        trt = rng.normal(10.5, 1, 50)
        with pytest.raises(TypeError, match="array-like"):
            StratifiedExperiment(
                ctrl, trt, control_strata=42, treatment_strata=np.array(["A"] * 50)
            )

    def test_control_too_short(self):
        """Control array with < 2 elements raises ValueError."""
        with pytest.raises(ValueError, match="at least"):
            StratifiedExperiment(
                [1.0],
                [1.0, 2.0],
                control_strata=["A"],
                treatment_strata=["A", "A"],
            )

    def test_strata_not_1d(self):
        """Multi-dimensional strata array raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10, 1, 50)
        trt = rng.normal(10.5, 1, 50)
        with pytest.raises(ValueError, match="1-D"):
            StratifiedExperiment(
                ctrl,
                trt,
                control_strata=np.array([["A"] * 50]),
                treatment_strata=np.array(["A"] * 50),
            )

    def test_stratum_too_few_treatment(self):
        """Stratum with <2 treatment obs raises ValueError."""
        rng = np.random.default_rng(42)
        ctrl = np.concatenate([rng.normal(10, 1, 50), rng.normal(5, 1, 50)])
        trt = np.concatenate([rng.normal(10, 1, 1), rng.normal(5, 1, 50)])
        sc = np.array(["A"] * 50 + ["B"] * 50)
        st = np.array(["A"] * 1 + ["B"] * 50)
        with pytest.raises(ValueError, match="fewer than 2 treatment"):
            StratifiedExperiment(
                ctrl, trt, control_strata=sc, treatment_strata=st
            ).run()
