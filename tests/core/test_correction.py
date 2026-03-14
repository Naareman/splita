"""Tests for MultipleCorrection."""

from __future__ import annotations

import pytest

from splita.core.correction import MultipleCorrection

# ─── Basic tests ────────────────────────────────────────────────────


class TestBHDefault:
    """BH default: [0.005, 0.04, 0.20, 0.35, 0.60] → first 2 rejected."""

    def test_bh_rejections(self):
        # 0.005*5/1=0.025, 0.04*5/2=0.10 → but backward: min(0.10,...)=0.10
        # adjusted: [0.025, 0.10, 0.333, 0.4375, 0.60]
        # Only first is < 0.05
        result = MultipleCorrection([0.005, 0.009, 0.20, 0.35, 0.60]).run()
        # 0.005*5/1=0.025 < 0.05, 0.009*5/2=0.0225 → backward enforced to 0.025 < 0.05
        assert result.rejected == [True, True, False, False, False]
        assert result.n_rejected == 2

    def test_bh_method_label(self):
        result = MultipleCorrection([0.01, 0.04, 0.20, 0.35, 0.60]).run()
        assert result.method == "Benjamini-Hochberg"


class TestBonferroni:
    """Bonferroni: [0.01, 0.04, 0.20] → only first rejected."""

    def test_bonferroni_rejections(self):
        result = MultipleCorrection([0.01, 0.04, 0.20], method="bonferroni").run()
        assert result.rejected == [True, False, False]
        assert result.n_rejected == 1

    def test_bonferroni_adjusted_values(self):
        result = MultipleCorrection([0.01, 0.04, 0.20], method="bonferroni").run()
        assert result.adjusted_pvalues[0] == pytest.approx(0.03)
        assert result.adjusted_pvalues[1] == pytest.approx(0.12)
        assert result.adjusted_pvalues[2] == pytest.approx(0.60)


class TestHolm:
    """Holm: [0.01, 0.04, 0.20] → first rejected, second not (0.04*2=0.08)."""

    def test_holm_rejections(self):
        result = MultipleCorrection([0.01, 0.04, 0.20], method="holm").run()
        assert result.rejected == [True, False, False]

    def test_holm_adjusted_values(self):
        result = MultipleCorrection([0.01, 0.04, 0.20], method="holm").run()
        # 0.01*3=0.03, max(0.04*2, 0.03)=0.08, max(0.20*1, 0.08)=0.20
        assert result.adjusted_pvalues[0] == pytest.approx(0.03)
        assert result.adjusted_pvalues[1] == pytest.approx(0.08)
        assert result.adjusted_pvalues[2] == pytest.approx(0.20)


class TestBY:
    """BY is more conservative than BH — fewer rejections on same data."""

    def test_by_more_conservative(self):
        pvals = [0.005, 0.009, 0.20, 0.35, 0.60]
        bh = MultipleCorrection(pvals, method="bh").run()
        by = MultipleCorrection(pvals, method="by").run()
        assert by.n_rejected <= bh.n_rejected


# ─── Statistical correctness ────────────────────────────────────────


class TestBonferroniCorrectness:
    """Verify adjusted = p * n for Bonferroni."""

    def test_adjusted_equals_p_times_n(self):
        pvals = [0.005, 0.03, 0.10, 0.50]
        result = MultipleCorrection(pvals, method="bonferroni").run()
        n = len(pvals)
        for i, p in enumerate(pvals):
            expected = min(p * n, 1.0)
            assert result.adjusted_pvalues[i] == pytest.approx(expected)


class TestHolmMonotonicity:
    """Holm adjusted p-values are non-decreasing."""

    def test_monotonicity(self):
        pvals = [0.03, 0.001, 0.15, 0.04, 0.50]
        result = MultipleCorrection(pvals, method="holm").run()
        # Sort by original p-value to check monotonicity in sorted order
        paired = sorted(zip(result.pvalues, result.adjusted_pvalues, strict=False))
        sorted_adj = [a for _, a in paired]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1]


class TestBHMonotonicity:
    """BH adjusted p-values are non-decreasing when sorted by original."""

    def test_monotonicity(self):
        pvals = [0.03, 0.001, 0.15, 0.04, 0.50]
        result = MultipleCorrection(pvals, method="bh").run()
        paired = sorted(zip(result.pvalues, result.adjusted_pvalues, strict=False))
        sorted_adj = [a for _, a in paired]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1]


class TestAdjustedCappedAtOne:
    """No adjusted p-value exceeds 1.0."""

    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm", "by"])
    def test_all_adjusted_le_one(self, method):
        pvals = [0.50, 0.60, 0.70, 0.80, 0.90]
        result = MultipleCorrection(pvals, method=method).run()
        for adj in result.adjusted_pvalues:
            assert adj <= 1.0


class TestBYMoreConservativeThanBH:
    """BY rejects <= BH on same data."""

    def test_by_le_bh_rejections(self):
        pvals = [0.001, 0.01, 0.03, 0.08, 0.20]
        bh = MultipleCorrection(pvals, method="bh").run()
        by = MultipleCorrection(pvals, method="by").run()
        assert by.n_rejected <= bh.n_rejected
        # BY adjusted should be >= BH adjusted
        for bh_adj, by_adj in zip(
            bh.adjusted_pvalues, by.adjusted_pvalues, strict=False
        ):
            assert by_adj >= bh_adj - 1e-10


class TestKnownBHAdjustments:
    """Manually computed BH adjustments."""

    def test_known_values(self):
        # pvalues sorted: 0.01, 0.04, 0.20; n=3
        # rank 1: 0.01 * 3/1 = 0.03
        # rank 2: 0.04 * 3/2 = 0.06
        # rank 3: 0.20 * 3/3 = 0.20
        # Backward enforcement: min(0.20, 0.20)=0.20,
        # min(0.06, 0.20)=0.06, min(0.03, 0.06)=0.03
        result = MultipleCorrection([0.01, 0.04, 0.20], method="bh").run()
        assert result.adjusted_pvalues[0] == pytest.approx(0.03)
        assert result.adjusted_pvalues[1] == pytest.approx(0.06)
        assert result.adjusted_pvalues[2] == pytest.approx(0.20)


# ─── Edge cases ─────────────────────────────────────────────────────


class TestSinglePValue:
    """Single p-value: all methods return the same adjusted value."""

    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm", "by"])
    def test_single_pvalue(self, method):
        result = MultipleCorrection([0.03], method=method).run()
        assert result.adjusted_pvalues[0] == pytest.approx(0.03)
        assert result.n_tests == 1


class TestAllSignificant:
    """All very small p-values → all rejected."""

    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm", "by"])
    def test_all_rejected(self, method):
        result = MultipleCorrection([0.001, 0.002, 0.003], method=method).run()
        assert all(result.rejected)
        assert result.n_rejected == 3


class TestNoneSignificant:
    """All large p-values → none rejected."""

    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm", "by"])
    def test_none_rejected(self, method):
        result = MultipleCorrection([0.50, 0.60, 0.70], method=method).run()
        assert not any(result.rejected)
        assert result.n_rejected == 0


class TestPValueZero:
    """P-value = 0 → adjusted should be 0 or very small."""

    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm", "by"])
    def test_zero_pvalue(self, method):
        result = MultipleCorrection([0.0, 0.5], method=method).run()
        assert result.adjusted_pvalues[0] == pytest.approx(0.0, abs=1e-15)


class TestPValueOne:
    """P-value = 1.0 → adjusted should be 1.0."""

    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm", "by"])
    def test_one_pvalue(self, method):
        result = MultipleCorrection([0.01, 1.0], method=method).run()
        assert result.adjusted_pvalues[1] == pytest.approx(1.0)


class TestIdenticalPValues:
    """Identical p-values → all adjusted the same."""

    @pytest.mark.parametrize("method", ["bh", "bonferroni", "holm", "by"])
    def test_identical(self, method):
        result = MultipleCorrection([0.03, 0.03, 0.03], method=method).run()
        adj = result.adjusted_pvalues
        assert adj[0] == pytest.approx(adj[1])
        assert adj[1] == pytest.approx(adj[2])


# ─── Validation ─────────────────────────────────────────────────────


class TestValidation:
    """Validation errors."""

    def test_empty_pvalues(self):
        with pytest.raises(ValueError, match="can't be empty"):
            MultipleCorrection([])

    def test_pvalue_above_one(self):
        with pytest.raises(ValueError, match="must all be in"):
            MultipleCorrection([0.5, 1.5])

    def test_pvalue_below_zero(self):
        with pytest.raises(ValueError, match="must all be in"):
            MultipleCorrection([-0.1, 0.5])

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="must be one of"):
            MultipleCorrection([0.05], method="fdr")

    def test_alpha_zero(self):
        with pytest.raises(ValueError, match="alpha"):
            MultipleCorrection([0.05], alpha=0.0)

    def test_alpha_one(self):
        with pytest.raises(ValueError, match="alpha"):
            MultipleCorrection([0.05], alpha=1.0)

    def test_labels_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            MultipleCorrection([0.01, 0.05], labels=["a"])


# ─── Properties ─────────────────────────────────────────────────────


class TestProperties:
    """Result properties are correct."""

    def test_labels_in_result(self):
        labels = ["revenue", "cvr", "aov", "session", "bounce"]
        result = MultipleCorrection([0.01, 0.04, 0.20, 0.35, 0.60], labels=labels).run()
        assert result.labels == labels

    def test_n_rejected_correct(self):
        result = MultipleCorrection([0.01, 0.04, 0.20, 0.35, 0.60]).run()
        assert result.n_rejected == sum(result.rejected)

    def test_n_tests_correct(self):
        pvals = [0.01, 0.04, 0.20, 0.35, 0.60]
        result = MultipleCorrection(pvals).run()
        assert result.n_tests == len(pvals)

    def test_to_dict(self):
        result = MultipleCorrection([0.01, 0.04, 0.20]).run()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "pvalues" in d
        assert "adjusted_pvalues" in d
        assert "rejected" in d
        assert "method" in d

    def test_idempotent(self):
        mc = MultipleCorrection([0.01, 0.04, 0.20, 0.35, 0.60])
        r1 = mc.run()
        r2 = mc.run()
        assert r1.adjusted_pvalues == r2.adjusted_pvalues
        assert r1.rejected == r2.rejected


# ─── Labels ─────────────────────────────────────────────────────────


class TestLabelsPreserved:
    """Labels are preserved in result."""

    def test_labels_preserved(self):
        labels = ["revenue", "cvr", "aov", "session", "bounce"]
        result = MultipleCorrection([0.01, 0.04, 0.20, 0.35, 0.60], labels=labels).run()
        assert result.labels == labels

    def test_no_labels(self):
        result = MultipleCorrection([0.01, 0.04]).run()
        assert result.labels is None


# ─── Milestone 3 review fixes ────────────────────────────────────


class TestMilestone3ReviewFixes:
    """Tests added to address expert review items."""

    def test_2d_pvalues_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            MultipleCorrection([[0.01, 0.02], [0.03, 0.04]])

    def test_holm_less_conservative_than_bonferroni(self):
        pvals = [0.01, 0.04, 0.20]
        holm = MultipleCorrection(pvals, method="holm").run()
        bonf = MultipleCorrection(pvals, method="bonferroni").run()
        for h_adj, b_adj in zip(
            holm.adjusted_pvalues, bonf.adjusted_pvalues, strict=False
        ):
            assert h_adj <= b_adj + 1e-10

    def test_unsorted_input_ordering_preserved(self):
        pvals = [0.60, 0.01, 0.04, 0.20]
        result = MultipleCorrection(pvals, method="bh").run()
        # Original order preserved in result.pvalues
        assert result.pvalues == [0.60, 0.01, 0.04, 0.20]
        # The smallest raw p-value (index 1) should have the smallest adjusted
        adj = result.adjusted_pvalues
        assert adj[1] <= adj[0]
        assert adj[1] <= adj[2]
        assert adj[1] <= adj[3]

    def test_tied_pvalues_mixed(self):
        result = MultipleCorrection([0.01, 0.01, 0.05], method="bh").run()
        # Both 0.01 values should get the same adjusted p-value
        assert result.adjusted_pvalues[0] == pytest.approx(
            result.adjusted_pvalues[1], abs=1e-10
        )
        # All adjusted should be <= 1.0
        for adj in result.adjusted_pvalues:
            assert adj <= 1.0

    def test_labels_all_methods(self):
        pvals = [0.01, 0.04, 0.20]
        labels = ["metric_a", "metric_b", "metric_c"]
        for method in ["bonferroni", "holm", "bh", "by"]:
            result = MultipleCorrection(pvals, method=method, labels=labels).run()
            assert result.labels == labels
