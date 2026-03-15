"""Tests for splita.playground — interactive Streamlit playground."""

from __future__ import annotations

import pytest


class TestPlaygroundImport:
    def test_import_error_without_streamlit(self):
        """playground() should raise ImportError with helpful message if streamlit missing."""
        try:
            import streamlit  # noqa: F401

            pytest.skip("streamlit is installed; cannot test ImportError path")
        except ImportError:
            pass

        from splita.playground import playground

        with pytest.raises(ImportError, match="streamlit"):
            playground()

    def test_playground_function_exists(self):
        """Verify the function is importable."""
        from splita.playground import playground

        assert callable(playground)

    def test_playground_exported_from_package(self):
        """Verify playground is accessible from the top-level package."""
        import splita

        assert hasattr(splita, "playground")
        assert callable(splita.playground)

    def test_playground_in_all(self):
        """Verify playground is listed in __all__."""
        import splita

        assert "playground" in splita.__all__

    def test_playground_app_module_exists(self):
        """Verify the internal Streamlit app file exists."""
        import os

        import splita

        app_path = os.path.join(os.path.dirname(splita.__file__), "_playground_app.py")
        assert os.path.isfile(app_path)
