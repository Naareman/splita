"""Verbose context manager for splita operations.

Enables detailed informational output about what splita is doing
under the hood — which tests it selects, why, and what it finds.
"""

from __future__ import annotations

import contextlib
from typing import Generator


@contextlib.contextmanager
def verbose() -> Generator[None, None, None]:
    """Enable verbose explanations for all splita operations.

    Example
    -------
    >>> import splita
    >>> import numpy as np
    >>> ctrl = np.random.default_rng(0).normal(10, 2, 100)
    >>> trt = np.random.default_rng(1).normal(10.5, 2, 100)
    >>> with splita.verbose():
    ...     result = splita.Experiment(ctrl, trt).run()
    """
    import splita._advisory as adv

    old_verbose = getattr(adv, "_VERBOSE", False)
    adv._VERBOSE = True
    try:
        yield
    finally:
        adv._VERBOSE = old_verbose
