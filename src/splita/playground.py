"""Interactive Streamlit-based A/B test playground.

Launch via Python::

    from splita import playground
    playground()  # opens browser at localhost:8501

Or from CLI::

    python -m splita.playground

Requires ``streamlit >= 1.30``.  Install with::

    pip install splita[playground]

Examples
--------
>>> from splita.playground import playground  # doctest: +SKIP
>>> playground(port=8502)  # doctest: +SKIP
"""

from __future__ import annotations

import os
import subprocess
import sys


def playground(port: int = 8501) -> None:
    """Launch interactive A/B testing playground in browser.

    Starts a Streamlit server serving the splita playground app and
    opens the default browser at ``http://localhost:<port>``.

    Parameters
    ----------
    port : int, default 8501
        Port number for the Streamlit server.

    Raises
    ------
    ImportError
        If ``streamlit`` is not installed.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise ImportError(
            "splita.playground requires streamlit. Install with: pip install splita[playground]"
        ) from None

    app_path = os.path.join(os.path.dirname(__file__), "_playground_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)],
    )
