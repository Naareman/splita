"""Structured exception hierarchy for splita.

All public exceptions inherit from ``SplitaError`` so users can catch the
entire family with a single ``except SplitaError`` clause.
"""

from __future__ import annotations


class SplitaError(Exception):
    """Base exception for splita."""


class ValidationError(SplitaError):
    """Raised when input validation fails.

    Attributes:
        parameter: The name of the parameter that failed validation, if known.
    """

    def __init__(self, message: str, parameter: str | None = None) -> None:
        self.parameter = parameter
        super().__init__(message)


class NotFittedError(SplitaError):
    """Raised when transform is called before fit."""


class InsufficientDataError(SplitaError):
    """Raised when there isn't enough data for analysis."""
