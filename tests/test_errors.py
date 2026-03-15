"""Tests for splita error hierarchy."""

from splita.errors import InsufficientDataError, NotFittedError, SplitaError, ValidationError


def test_validation_error_is_value_error():
    assert issubclass(ValidationError, ValueError)
    assert issubclass(ValidationError, SplitaError)


def test_not_fitted_error_is_runtime_error():
    assert issubclass(NotFittedError, RuntimeError)
    assert issubclass(NotFittedError, SplitaError)


def test_insufficient_data_error_is_value_error():
    assert issubclass(InsufficientDataError, ValueError)


def test_validation_error_parameter():
    err = ValidationError("bad", parameter="alpha")
    assert err.parameter == "alpha"
    assert str(err) == "bad"
