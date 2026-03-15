"""Tests for RandomizationValidator."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import RandomizationResult
from splita.diagnostics.randomization import RandomizationValidator


# ── helpers ──────────────────────────────────────────────────────────


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── Basic behaviour ──────────────────────────────────────────────────


class TestBasic:
    """Basic RandomizationValidator behaviour."""

    def test_balanced_groups(self):
        """Properly randomised groups are balanced."""
        rng = _rng()
        ctrl = rng.normal(0, 1, (500, 3))
        trt = rng.normal(0, 1, (500, 3))
        result = RandomizationValidator().validate(ctrl, trt)
        assert isinstance(result, RandomizationResult)
        assert result.balanced

    def test_imbalanced_groups(self):
        """Groups with different means are flagged as imbalanced."""
        rng = _rng()
        ctrl = rng.normal(0, 1, (500, 2))
        trt = rng.normal(0, 1, (500, 2))
        # Make one covariate very different
        trt[:, 0] += 2.0
        result = RandomizationValidator().validate(ctrl, trt, ["age", "income"])
        assert not result.balanced
        assert "age" in result.imbalanced_covariates

    def test_result_frozen(self):
        """Result is frozen."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (50, 2)), rng.normal(0, 1, (50, 2))
        )
        with pytest.raises(AttributeError):
            result.balanced = False  # type: ignore[misc]

    def test_to_dict(self):
        """Result serializes to dict."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (50, 2)), rng.normal(0, 1, (50, 2))
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "balanced" in d
        assert "smd_per_covariate" in d

    def test_repr(self):
        """Result has string representation."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (50, 2)), rng.normal(0, 1, (50, 2))
        )
        s = repr(result)
        assert "RandomizationResult" in s


# ── SMD properties ───────────────────────────────────────────────────


class TestSMD:
    """Standardised mean difference properties."""

    def test_smd_count_matches_covariates(self):
        """Number of SMD entries matches number of covariates."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (100, 5)),
            rng.normal(0, 1, (100, 5)),
        )
        assert len(result.smd_per_covariate) == 5

    def test_smd_nonneg(self):
        """All SMDs are non-negative."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (100, 3)),
            rng.normal(0, 1, (100, 3)),
        )
        for d in result.smd_per_covariate:
            assert d["smd"] >= 0

    def test_max_smd(self):
        """max_smd is the maximum of per-covariate SMDs."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (100, 3)),
            rng.normal(0, 1, (100, 3)),
        )
        smds = [d["smd"] for d in result.smd_per_covariate]
        np.testing.assert_allclose(result.max_smd, max(smds))


# ── Covariate names ──────────────────────────────────────────────────


class TestNames:
    """Covariate naming."""

    def test_default_names(self):
        """Default names are x_0, x_1, ..."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (50, 3)),
            rng.normal(0, 1, (50, 3)),
        )
        names = [d["name"] for d in result.smd_per_covariate]
        assert names == ["x_0", "x_1", "x_2"]

    def test_custom_names(self):
        """Custom names are used."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (50, 2)),
            rng.normal(0, 1, (50, 2)),
            covariate_names=["age", "income"],
        )
        names = [d["name"] for d in result.smd_per_covariate]
        assert names == ["age", "income"]


# ── 1-D input ────────────────────────────────────────────────────────


class TestOneDimensional:
    """Single covariate (1-D input)."""

    def test_1d_arrays(self):
        """1-D arrays are handled correctly."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, 100),
            rng.normal(0, 1, 100),
        )
        assert len(result.smd_per_covariate) == 1


# ── Omnibus test ─────────────────────────────────────────────────────


class TestOmnibus:
    """Omnibus chi-squared test."""

    def test_omnibus_pvalue_range(self):
        """Omnibus p-value is between 0 and 1."""
        rng = _rng()
        result = RandomizationValidator().validate(
            rng.normal(0, 1, (200, 3)),
            rng.normal(0, 1, (200, 3)),
        )
        assert 0.0 <= result.omnibus_pvalue <= 1.0


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_alpha_out_of_range(self):
        """alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            RandomizationValidator(alpha=1.5)

    def test_different_covariate_counts(self):
        """Different number of covariates raises ValueError."""
        rng = _rng()
        with pytest.raises(ValueError, match="same number"):
            RandomizationValidator().validate(
                rng.normal(0, 1, (50, 3)),
                rng.normal(0, 1, (50, 2)),
            )

    def test_too_few_control(self):
        """< 2 control observations raises ValueError."""
        rng = _rng()
        with pytest.raises(ValueError, match="at least 2"):
            RandomizationValidator().validate(
                rng.normal(0, 1, (1, 2)),
                rng.normal(0, 1, (50, 2)),
            )

    def test_wrong_name_count(self):
        """Wrong number of names raises ValueError."""
        rng = _rng()
        with pytest.raises(ValueError, match="covariate_names"):
            RandomizationValidator().validate(
                rng.normal(0, 1, (50, 3)),
                rng.normal(0, 1, (50, 3)),
                covariate_names=["a", "b"],
            )


# ── E2E scenario ─────────────────────────────────────────────────────


class TestE2E:
    """End-to-end scenario."""

    def test_ab_test_randomization_check(self):
        """Pre-experiment randomisation check for an A/B test."""
        rng = _rng(99)
        n = 5000
        # Well-randomised
        age = rng.normal(35, 10, n)
        income = rng.normal(50000, 15000, n)
        visits = rng.poisson(5, n).astype(float)

        ctrl_idx = rng.choice(n, n // 2, replace=False)
        trt_idx = np.setdiff1d(np.arange(n), ctrl_idx)

        covs = np.column_stack([age, income, visits])
        result = RandomizationValidator().validate(
            covs[ctrl_idx],
            covs[trt_idx],
            covariate_names=["age", "income", "visits"],
        )
        assert result.balanced
        assert result.max_smd < 0.1
        assert len(result.imbalanced_covariates) == 0
