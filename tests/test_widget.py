"""Tests for splita.widget — sample size planning widget."""

from __future__ import annotations

import pytest


class TestWidgetImport:
    def test_import_error_without_ipywidgets(self):
        """Widget should raise ImportError with helpful message if ipywidgets missing."""
        try:
            import ipywidgets  # noqa: F401

            pytest.skip("ipywidgets is installed; cannot test ImportError path")
        except ImportError:
            pass

        from splita.widget import sample_size_widget

        with pytest.raises(ImportError, match="ipywidgets"):
            sample_size_widget()

    def test_widget_function_exists(self):
        """Verify the function is importable."""
        from splita.widget import sample_size_widget

        assert callable(sample_size_widget)
