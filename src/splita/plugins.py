"""Plugin system for registering custom test methods.

Allows users to extend splita with their own statistical test
implementations::

    from splita.plugins import register_method, get_method, list_methods

    class MyCustomTest:
        def __init__(self, control, treatment, **kwargs):
            self.control = control
            self.treatment = treatment

        def run(self):
            # ... custom analysis logic ...
            return result

    register_method("my_test", MyCustomTest)
    cls = get_method("my_test")
"""

from __future__ import annotations

_CUSTOM_METHODS: dict[str, type] = {}


def register_method(name: str, cls: type) -> None:
    """Register a custom test method.

    Parameters
    ----------
    name : str
        Name to register the method under. Must be a non-empty string.
    cls : type
        A class that has a ``run()`` method.

    Raises
    ------
    ValueError
        If *name* is empty or *cls* does not have a ``run()`` method.
    TypeError
        If *name* is not a string or *cls* is not a class.
    """
    if not isinstance(name, str):
        raise TypeError(f"`name` must be a string, got {type(name).__name__}.")
    if not name:
        raise ValueError("`name` must be a non-empty string.")
    if not isinstance(cls, type):
        raise TypeError(f"`cls` must be a class, got {type(cls).__name__}.")
    if not hasattr(cls, "run"):
        raise ValueError(f"Custom method {cls.__name__!r} must have a run() method.")
    _CUSTOM_METHODS[name] = cls


def get_method(name: str) -> type | None:
    """Get a registered custom method by name.

    Parameters
    ----------
    name : str
        The registered name.

    Returns
    -------
    type or None
        The registered class, or None if not found.
    """
    return _CUSTOM_METHODS.get(name)


def list_methods() -> list[str]:
    """List all registered custom method names.

    Returns
    -------
    list of str
        Sorted list of registered method names.
    """
    return sorted(_CUSTOM_METHODS.keys())


def unregister_method(name: str) -> bool:
    """Remove a registered custom method.

    Parameters
    ----------
    name : str
        The method name to unregister.

    Returns
    -------
    bool
        True if the method was found and removed, False otherwise.
    """
    if name in _CUSTOM_METHODS:
        del _CUSTOM_METHODS[name]
        return True
    return False


def clear_methods() -> None:
    """Remove all registered custom methods."""
    _CUSTOM_METHODS.clear()
