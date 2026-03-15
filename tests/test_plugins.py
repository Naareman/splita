"""Tests for the plugin system."""

from __future__ import annotations

import pytest

from splita.plugins import (
    clear_methods,
    get_method,
    list_methods,
    register_method,
    unregister_method,
)


class _DummyMethod:
    """Minimal custom method with run()."""

    def __init__(self, control, treatment):
        self.control = control
        self.treatment = treatment

    def run(self):
        return {"diff": sum(self.treatment) - sum(self.control)}


class _NoRunMethod:
    """Class without a run() method."""

    pass


class TestPluginSystem:
    """Tests for register_method / get_method / list_methods."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        clear_methods()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        clear_methods()

    def test_register_and_get(self) -> None:
        """Registered method can be retrieved."""
        register_method("my_test", _DummyMethod)
        assert get_method("my_test") is _DummyMethod

    def test_get_nonexistent(self) -> None:
        """get_method returns None for unregistered name."""
        assert get_method("does_not_exist") is None

    def test_list_methods(self) -> None:
        """list_methods returns sorted names."""
        register_method("zzz", _DummyMethod)
        register_method("aaa", _DummyMethod)
        assert list_methods() == ["aaa", "zzz"]

    def test_list_methods_empty(self) -> None:
        """list_methods returns empty list when nothing registered."""
        assert list_methods() == []

    def test_register_no_run_method(self) -> None:
        """ValueError raised when class has no run() method."""
        with pytest.raises(ValueError, match="must have a run"):
            register_method("bad", _NoRunMethod)

    def test_register_empty_name(self) -> None:
        """ValueError raised for empty name."""
        with pytest.raises(ValueError, match="non-empty"):
            register_method("", _DummyMethod)

    def test_register_non_string_name(self) -> None:
        """TypeError raised for non-string name."""
        with pytest.raises(TypeError, match="must be a string"):
            register_method(42, _DummyMethod)  # type: ignore[arg-type]

    def test_register_non_class(self) -> None:
        """TypeError raised when passing an instance instead of a class."""
        with pytest.raises(TypeError, match="must be a class"):
            register_method("bad", _DummyMethod([], []))  # type: ignore[arg-type]

    def test_overwrite_registration(self) -> None:
        """Re-registering a name overwrites the previous entry."""
        register_method("my_test", _DummyMethod)

        class AnotherMethod:
            def run(self):
                pass

        register_method("my_test", AnotherMethod)
        assert get_method("my_test") is AnotherMethod

    def test_unregister_method(self) -> None:
        """Unregistering a method removes it from the registry."""
        register_method("temp", _DummyMethod)
        assert unregister_method("temp") is True
        assert get_method("temp") is None

    def test_unregister_nonexistent(self) -> None:
        """Unregistering a nonexistent method returns False."""
        assert unregister_method("nope") is False

    def test_clear_methods(self) -> None:
        """clear_methods empties the registry."""
        register_method("a", _DummyMethod)
        register_method("b", _DummyMethod)
        clear_methods()
        assert list_methods() == []

    def test_registered_method_is_callable(self) -> None:
        """Retrieved method class can be instantiated and run."""
        register_method("my_test", _DummyMethod)
        cls = get_method("my_test")
        assert cls is not None
        instance = cls([1, 2, 3], [4, 5, 6])
        result = instance.run()
        assert result == {"diff": 9}
