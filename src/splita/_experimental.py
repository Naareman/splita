import functools
import warnings


def experimental(func):
    """Mark a class or function as experimental."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__qualname__} is experimental and may change without notice.",
            FutureWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper
