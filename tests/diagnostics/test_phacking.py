"""Tests for PHackingDetector."""

from __future__ import annotations

import numpy as np
import pytest

from splita.diagnostics.phacking import PHackingDetector


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestPHackingBasic:
    """Basic functionality tests."""

    def test_honest_pvalues_not_suspicious(self, rng):
        """Right-skewed p-values (true effects) should not be suspicious."""
        # Under true effects, significant p-values are right-skewed
        pvals = (rng.beta(1, 10, 50) * 0.05).tolist()
        r = PHackingDetector().detect(pvals)
        assert r.suspicious is False

    def test_bunched_pvalues_suspicious(self):
        """P-values bunched near 0.05 should trigger bunching detection."""
        # Lots of values just below 0.05
        pvals = [0.048, 0.049, 0.047, 0.046, 0.045, 0.044, 0.043,
                 0.042, 0.041, 0.049, 0.048, 0.047, 0.046, 0.045,
                 0.01, 0.02, 0.03]
        r = PHackingDetector().detect(pvals)
        assert r.bunching_near_05 is True

    def test_uniform_pvalues(self, rng):
        """Uniform p-values indicate no true effect (null)."""
        pvals = rng.uniform(0, 0.05, 30).tolist()
        r = PHackingDetector().detect(pvals)
        # Should not be marked as right-skewed
        assert isinstance(r.suspicious, bool)

    def test_insufficient_significant(self, rng):
        """Fewer than 3 significant p-values should return early."""
        pvals = [0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.02]
        r = PHackingDetector().detect(pvals)
        assert "Insufficient" in r.message or r.p_curve_test_pvalue == 1.0

    def test_n_experiments(self, rng):
        pvals = rng.uniform(0, 0.05, 20).tolist()
        r = PHackingDetector().detect(pvals)
        assert r.n_experiments == 20

    def test_message_is_string(self, rng):
        pvals = rng.uniform(0, 0.05, 10).tolist()
        r = PHackingDetector().detect(pvals)
        assert isinstance(r.message, str)
        assert len(r.message) > 10

    def test_all_very_small(self, rng):
        """Very small p-values (strong effects) should not be suspicious."""
        pvals = [0.001, 0.0001, 0.002, 0.003, 0.0005, 0.001,
                 0.0002, 0.004, 0.001, 0.003]
        r = PHackingDetector().detect(pvals)
        assert r.suspicious is False

    def test_custom_threshold(self, rng):
        """Custom significance threshold should be respected."""
        pvals = (rng.beta(1, 10, 30) * 0.10).tolist()
        r = PHackingDetector(significance_threshold=0.10).detect(pvals)
        assert isinstance(r.suspicious, bool)

    def test_to_dict(self, rng):
        pvals = rng.uniform(0, 0.05, 10).tolist()
        r = PHackingDetector().detect(pvals)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "suspicious" in d

    def test_repr(self, rng):
        pvals = rng.uniform(0, 0.05, 10).tolist()
        r = PHackingDetector().detect(pvals)
        assert "PHackingResult" in repr(r)


class TestPHackingValidation:
    """Validation and error tests."""

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="significance_threshold"):
            PHackingDetector(significance_threshold=0.0)

    def test_too_few_pvalues(self):
        with pytest.raises(ValueError, match="at least 3"):
            PHackingDetector().detect([0.01, 0.02])

    def test_pvalues_out_of_range(self):
        with pytest.raises(ValueError, match="[0, 1]"):
            PHackingDetector().detect([0.01, 0.02, -0.5, 0.03])

    def test_pvalues_above_one(self):
        with pytest.raises(ValueError, match="[0, 1]"):
            PHackingDetector().detect([0.01, 0.02, 1.5, 0.03])

    def test_non_list_input(self):
        with pytest.raises(TypeError, match="list"):
            PHackingDetector().detect("not_a_list")

    def test_nan_pvalues(self):
        with pytest.raises(ValueError, match="NaN"):
            PHackingDetector().detect([0.01, 0.02, float("nan"), 0.03])
