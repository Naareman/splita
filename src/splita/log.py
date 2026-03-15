"""Experiment logging to local JSON-lines storage.

Provides simple append-only logging of experiment results::

    from splita.log import log, load_log

    result = Experiment(ctrl, trt).run()
    log(result, "checkout_button_color")

    history = load_log()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log(
    result: Any,
    experiment_name: str,
    *,
    storage: str = "json",
    path: str = "experiments.json",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an experiment result to local storage.

    Writes one JSON object per line (JSON-lines format).

    Parameters
    ----------
    result : object
        A splita result object with a ``to_dict()`` method.
    experiment_name : str
        A human-readable name for this experiment.
    storage : str, default "json"
        Storage format. Currently only ``"json"`` (JSON-lines) is supported.
    path : str, default "experiments.json"
        File path for the log file.
    metadata : dict or None, default None
        Optional extra metadata to include in the log entry.

    Returns
    -------
    dict
        The logged entry.

    Raises
    ------
    ValueError
        If the result does not have a ``to_dict()`` method, or if the
        storage format is not supported.
    TypeError
        If *experiment_name* is not a string.
    """
    if storage != "json":
        raise ValueError(
            f"Unsupported storage format {storage!r}.\n"
            f"  Detail: only 'json' (JSON-lines) is currently supported.\n"
            f"  Hint: use storage='json'."
        )

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise TypeError("`experiment_name` must be a non-empty string.")

    if not hasattr(result, "to_dict"):
        raise ValueError(
            f"Result object of type {type(result).__name__!r} does not have a "
            f"to_dict() method.\n"
            f"  Hint: pass a splita result object (e.g. ExperimentResult)."
        )

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "result": result.to_dict(),
    }

    if metadata:
        entry["metadata"] = metadata

    file_path = Path(path)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return entry


def load_log(path: str = "experiments.json") -> list[dict[str, Any]]:
    """Load all logged experiments from a JSON-lines file.

    Parameters
    ----------
    path : str, default "experiments.json"
        File path for the log file.

    Returns
    -------
    list of dict
        All logged experiment entries, in chronological order.
        Returns an empty list if the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path!r}.\n  Detail: {exc}"
                ) from exc

    return entries
