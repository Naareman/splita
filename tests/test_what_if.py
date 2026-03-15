"""Tests for splita.what_if."""

from __future__ import annotations

import numpy as np
import pytest

from splita._types import ExperimentResult
from splita.what_if import what_if


def _make_result(
    *,
    lift: float = 0.02,
    pvalue: float = 0.12,
    significant: bool = False,
    control_n: int = 1000,
    treatment_n: int = 1000,
    control_mean: float = 0.10,
    treatment_mean: float = 0.12,
    alpha: float = 0.05,
) -> ExperimentResult:
    return ExperimentResult(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        lift=lift,
        relative_lift=lift / control_mean if control_mean else 0,
        pvalue=pvalue,
        statistic=1.5,
        ci_lower=-0.005,
        ci_upper=0.045,
        significant=significant,
        alpha=alpha,
        method="ztest",
        metric="conversion",
        control_n=control_n,
        treatment_n=treatment_n,
        power=0.45,
        effect_size=0.06,
    )


class TestWhatIfBasic:
    """Basic what_if functionality."""

    def test_returns_what_if_result(self) -> None:
        r = _make_result()
        w = what_if(r, n=10000)
        assert hasattr(w, "projected_n")
        assert hasattr(w, "projected_pvalue")

    def test_projected_n_matches_input(self) -> None:
        r = _make_result()
        w = what_if(r, n=10000)
        assert w.projected_n == 10000

    def test_original_n_preserved(self) -> None:
        r = _make_result(control_n=1000, treatment_n=1000)
        w = what_if(r, n=10000)
        assert w.original_n == 2000

    def test_larger_n_reduces_pvalue(self) -> None:
        r = _make_result(pvalue=0.12)
        w = what_if(r, n=20000)
        assert w.projected_pvalue < w.original_pvalue

    def test_smaller_n_increases_pvalue(self) -> None:
        r = _make_result(pvalue=0.04, significant=True)
        w = what_if(r, n=500)
        assert w.projected_pvalue > w.original_pvalue


class TestWhatIfSignificance:
    """Significance projections."""

    def test_becomes_significant_with_more_users(self) -> None:
        r = _make_result(pvalue=0.12, significant=False)
        w = what_if(r, n=50000)
        assert w.projected_significant is True
        assert w.original_significant is False

    def test_stays_significant(self) -> None:
        r = _make_result(pvalue=0.01, significant=True)
        w = what_if(r, n=20000)
        assert w.projected_significant is True

    def test_loses_significance_with_fewer_users(self) -> None:
        r = _make_result(pvalue=0.04, significant=True)
        w = what_if(r, n=200)
        assert w.projected_significant is False


class TestWhatIfAlpha:
    """Changing alpha."""

    def test_stricter_alpha_may_lose_significance(self) -> None:
        r = _make_result(pvalue=0.04, significant=True, alpha=0.05)
        w = what_if(r, alpha=0.01)
        # p=0.04 > 0.01 so should not be significant
        assert w.projected_significant is False


class TestWhatIfMessage:
    """Message content."""

    def test_message_contains_n(self) -> None:
        r = _make_result()
        w = what_if(r, n=10000)
        assert "10,000" in w.message

    def test_message_indicates_would_be_significant(self) -> None:
        r = _make_result(pvalue=0.12, significant=False)
        w = what_if(r, n=50000)
        assert "WOULD be" in w.message

    def test_message_no_change(self) -> None:
        r = _make_result()
        w = what_if(r)
        assert "same conditions" in w.message


class TestWhatIfPower:
    """Power projections."""

    def test_power_between_0_and_1(self) -> None:
        r = _make_result()
        w = what_if(r, n=10000)
        assert 0.0 <= w.projected_power <= 1.0

    def test_power_increases_with_n(self) -> None:
        r = _make_result()
        w1 = what_if(r, n=2000)
        w2 = what_if(r, n=20000)
        assert w2.projected_power >= w1.projected_power


class TestWhatIfValidation:
    """Input validation."""

    def test_missing_attributes_raises(self) -> None:
        class FakeResult:
            pass
        with pytest.raises(ValueError, match="missing required"):
            what_if(FakeResult(), n=1000)


class TestWhatIfSerialization:
    """Serialization."""

    def test_to_dict(self) -> None:
        r = _make_result()
        w = what_if(r, n=10000)
        d = w.to_dict()
        assert isinstance(d, dict)
        assert "projected_n" in d

    def test_to_json(self) -> None:
        r = _make_result()
        w = what_if(r, n=10000)
        j = w.to_json()
        assert isinstance(j, str)
        assert "projected_n" in j
