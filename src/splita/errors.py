"""Structured exception hierarchy for splita.

All public exceptions inherit from ``SplitaError`` so users can catch the
entire family with a single ``except SplitaError`` clause.
"""

from __future__ import annotations


class SplitaError(Exception):
    """Base exception for splita."""


class ValidationError(SplitaError, ValueError):
    """Raised when input validation fails.

    Also inherits from ``ValueError`` for backward compatibility — existing
    ``except ValueError`` handlers will still catch these.

    Attributes
    ----------
    parameter : str or None
        The name of the parameter that failed validation, if known.
    """

    def __init__(self, message: str, parameter: str | None = None) -> None:
        self.parameter = parameter
        super().__init__(message)


class NotFittedError(SplitaError, RuntimeError):
    """Raised when transform is called before fit.

    Also inherits from ``RuntimeError`` for backward compatibility — existing
    ``except RuntimeError`` handlers will still catch these.
    """


class InsufficientDataError(SplitaError, ValueError):
    """Raised when there isn't enough data for analysis.

    Also inherits from ``ValueError`` for backward compatibility.
    """
