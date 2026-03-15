"""Tests for splita.power_report."""

from __future__ import annotations

import pytest

from splita.power_report import (
    _mde_for_n,
    _n_for_power,
    _power_for_n,
    power_report,
)


class TestPowerReportText:
    """Text format report."""

    def test_contains_header(self) -> None:
        txt = power_report(0.10, format="text")
        assert "Power Analysis Report" in txt

    def test_contains_baseline(self) -> None:
        txt = power_report(0.10, format="text")
        assert "0.1000" in txt

    def test_contains_table_headers(self) -> None:
        txt = power_report(0.10, format="text")
        assert "Table 1" in txt
        assert "Table 2" in txt
        assert "Table 3" in txt

    def test_contains_recommendation(self) -> None:
        txt = power_report(0.10, format="text")
        assert "Recommendation" in txt

    def test_custom_mde_range(self) -> None:
        txt = power_report(0.10, mde_range=[0.01, 0.05], format="text")
        assert "MDE=0.0100" in txt
        assert "MDE=0.0500" in txt

    def test_custom_n_range(self) -> None:
        txt = power_report(0.10, n_range=[100, 200], format="text")
        assert "100" in txt
        assert "200" in txt


class TestPowerReportHTML:
    """HTML format report."""

    def test_html_contains_tags(self) -> None:
        html = power_report(0.10, format="html")
        assert "<h2>" in html
        assert "<table" in html

    def test_html_contains_baseline(self) -> None:
        html = power_report(0.10, format="html")
        assert "0.1000" in html

    def test_html_color_coding(self) -> None:
        html = power_report(0.10, n_range=[50000], mde_range=[0.05],
                            format="html")
        assert "#28a745" in html  # green for high power


class TestPowerHelpers:
    """Internal helper functions."""

    def test_power_increases_with_n(self) -> None:
        p1 = _power_for_n(0.10, 0.02, 1000, 0.05, "conversion")
        p2 = _power_for_n(0.10, 0.02, 10000, 0.05, "conversion")
        assert p2 > p1

    def test_power_increases_with_mde(self) -> None:
        p1 = _power_for_n(0.10, 0.01, 5000, 0.05, "conversion")
        p2 = _power_for_n(0.10, 0.05, 5000, 0.05, "conversion")
        assert p2 > p1

    def test_n_for_power_returns_positive(self) -> None:
        n = _n_for_power(0.10, 0.02, 0.05, 0.80, "conversion")
        assert n > 0

    def test_n_for_power_decreases_with_larger_mde(self) -> None:
        n1 = _n_for_power(0.10, 0.01, 0.05, 0.80, "conversion")
        n2 = _n_for_power(0.10, 0.05, 0.05, 0.80, "conversion")
        assert n2 < n1

    def test_mde_for_n_decreases_with_larger_n(self) -> None:
        mde1 = _mde_for_n(0.10, 1000, 0.05, 0.80, "conversion")
        mde2 = _mde_for_n(0.10, 10000, 0.05, 0.80, "conversion")
        assert mde2 < mde1

    def test_continuous_metric(self) -> None:
        txt = power_report(5.0, metric="continuous", format="text")
        assert "continuous" in txt

    def test_power_near_80_at_required_n(self) -> None:
        n = _n_for_power(0.10, 0.02, 0.05, 0.80, "conversion")
        p = _power_for_n(0.10, 0.02, n, 0.05, "conversion")
        assert 0.75 < p < 0.90  # Should be close to 80%
