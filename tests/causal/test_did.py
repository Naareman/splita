"""Tests for DifferenceInDifferences (M17: Causal Inference — DiD)."""

from __future__ import annotations

import numpy as np
import pytest

from splita.causal.did import DifferenceInDifferences


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestDiDBasic:
    """Basic functionality tests."""

    def test_known_effect(self, rng):
        """DiD should recover a known treatment effect."""
        n = 200
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(13, 1, n)  # +3 effect

        did = DifferenceInDifferences()
        did.fit(pre_ctrl, pre_trt, post_ctrl, post_trt)
        r = did.result()

        assert abs(r.att - 3.0) < 1.0
        assert r.significant is True
        assert r.pvalue < 0.05

    def test_no_effect(self, rng):
        """No treatment effect should yield non-significant ATT."""
        n = 200
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(10, 1, n)

        r = DifferenceInDifferences().fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        assert abs(r.att) < 1.0
        assert r.pvalue > 0.01

    def test_parallel_trends_pass(self, rng):
        """When pre-period means are similar, parallel trends test passes."""
        n = 200
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(12, 1, n)

        r = DifferenceInDifferences().fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        assert r.parallel_trends_pvalue > 0.05

    def test_parallel_trends_fail(self, rng):
        """When pre-period means differ, parallel trends test should fail."""
        n = 200
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(15, 1, n)  # very different pre-period
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(17, 1, n)

        r = DifferenceInDifferences().fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        assert r.parallel_trends_pvalue < 0.05
        assert abs(r.pre_trend_diff) > 1.0

    def test_negative_effect(self, rng):
        """Negative treatment effect should yield negative ATT."""
        n = 200
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(7, 1, n)  # -3 effect

        r = DifferenceInDifferences().fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        assert r.att < 0
        assert r.significant is True

    def test_ci_contains_att(self, rng):
        """CI should contain the point estimate."""
        n = 200
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(12, 1, n)

        r = DifferenceInDifferences().fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        assert r.ci_lower <= r.att <= r.ci_upper

    def test_ci_contains_true_effect(self, rng):
        """CI should contain the true effect."""
        true_effect = 2.0
        n = 500
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(10 + true_effect, 1, n)

        r = DifferenceInDifferences().fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        assert r.ci_lower <= true_effect <= r.ci_upper

    def test_se_positive(self, rng):
        """Standard error should be positive."""
        n = 100
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(12, 1, n)

        r = DifferenceInDifferences().fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        assert r.se > 0

    def test_fit_returns_self(self, rng):
        did = DifferenceInDifferences()
        n = 50
        result = did.fit(
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
        )
        assert result is did

    def test_alpha_parameter(self, rng):
        n = 200
        pre_ctrl = rng.normal(10, 1, n)
        pre_trt = rng.normal(10, 1, n)
        post_ctrl = rng.normal(10, 1, n)
        post_trt = rng.normal(10.5, 1, n)

        r_05 = DifferenceInDifferences(alpha=0.05).fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()
        r_001 = DifferenceInDifferences(alpha=0.001).fit(
            pre_ctrl, pre_trt, post_ctrl, post_trt
        ).result()

        # p-values should be the same
        assert abs(r_05.pvalue - r_001.pvalue) < 0.001


class TestDiDValidation:
    """Tests for input validation."""

    def test_result_before_fit(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            DifferenceInDifferences().result()

    def test_too_few_samples(self, rng):
        with pytest.raises(ValueError, match="at least"):
            DifferenceInDifferences().fit(
                [1.0], [1.0], [1.0], [1.0]
            )

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            DifferenceInDifferences(alpha=1.5)

    def test_non_array_input(self):
        with pytest.raises(TypeError, match="array-like"):
            DifferenceInDifferences().fit(
                "not_an_array", [1, 2, 3], [1, 2, 3], [1, 2, 3]
            )

    def test_different_group_sizes(self, rng):
        """Different group sizes should work fine."""
        r = DifferenceInDifferences().fit(
            rng.normal(10, 1, 50),
            rng.normal(10, 1, 100),
            rng.normal(10, 1, 75),
            rng.normal(12, 1, 150),
        ).result()
        assert r.att > 0


class TestDiDResult:
    """Tests for result properties."""

    def test_to_dict(self, rng):
        n = 100
        r = DifferenceInDifferences().fit(
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(12, 1, n),
        ).result()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "att" in d
        assert "parallel_trends_pvalue" in d

    def test_repr(self, rng):
        n = 100
        r = DifferenceInDifferences().fit(
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(10, 1, n),
            rng.normal(12, 1, n),
        ).result()
        assert "DiDResult" in repr(r)
