"""Tests for LinUCB stub (planned for v0.2.0)."""

from __future__ import annotations

import pytest

from splita.bandits.linucb import LinUCB


class TestLinUCBStub:
    """LinUCB is a planned stub — verify it raises NotImplementedError."""

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="planned for splita v0.2.0"):
            LinUCB()

    def test_raises_with_args(self):
        with pytest.raises(NotImplementedError, match="LinUCB"):
            LinUCB(3, n_features=5)
