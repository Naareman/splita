"""Tests for EValue placeholder class."""

from __future__ import annotations

import pytest

from splita.sequential.evalue import EValue


class TestEValuePlaceholder:
    """EValue is planned for v0.2.0 and should raise NotImplementedError."""

    def test_raises_not_implemented(self):
        """Instantiating EValue should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match=r"planned for splita v0.2.0"):
            EValue()

    def test_raises_with_args(self):
        """Instantiating EValue with arguments should also raise."""
        with pytest.raises(NotImplementedError, match=r"planned for splita v0.2.0"):
            EValue(alpha=0.05, metric="conversion")
