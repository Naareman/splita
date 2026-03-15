"""Slack webhook notification for splita results.

Posts experiment results to Slack via incoming webhooks using only
``urllib`` (no ``requests`` dependency).

Examples
--------
>>> from splita._types import ExperimentResult
>>> r = ExperimentResult(
...     control_mean=0.10, treatment_mean=0.12,
...     lift=0.02, relative_lift=0.2, pvalue=0.003,
...     statistic=2.97, ci_lower=0.007, ci_upper=0.033,
...     significant=True, alpha=0.05, method="ztest",
...     metric="conversion", control_n=5000,
...     treatment_n=5000, power=0.82, effect_size=0.15,
... )
>>> # notify(r, webhook_url="https://hooks.slack.com/services/...")
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import fields
from typing import Any

from splita.explain import explain


def _format_fields(result: Any) -> str:
    """Format key result fields as a compact summary."""
    lines: list[str] = []
    for f in fields(result):
        val = getattr(result, f.name)
        if isinstance(val, float):
            if abs(val) < 0.0001 and val != 0.0:
                lines.append(f"*{f.name}*: `{val:.2e}`")
            else:
                lines.append(f"*{f.name}*: `{val:.4f}`")
        elif isinstance(val, bool):
            emoji = ":white_check_mark:" if val else ":x:"
            lines.append(f"*{f.name}*: {val} {emoji}")
        elif isinstance(val, (list, tuple)) and len(val) > 10:
            lines.append(f"*{f.name}*: [{len(val)} items]")
        else:
            lines.append(f"*{f.name}*: `{val}`")
    return "\n".join(lines)


def _build_blocks(
    result: Any,
    title: str,
) -> list[dict[str, Any]]:
    """Build Slack Block Kit payload."""
    type_name = type(result).__name__

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": title,
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Result type*: `{type_name}`",
            },
        },
    ]

    # Add explain() text if supported
    try:
        explanation = explain(result)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": explanation,
                },
            }
        )
    except TypeError:
        pass

    # Add key fields as context
    field_text = _format_fields(result)
    # Slack blocks have a 3000 char limit per text field
    if len(field_text) > 2900:
        field_text = field_text[:2900] + "\n..."

    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": field_text,
            },
        }
    )

    return blocks


def notify(
    result: Any,
    webhook_url: str,
    *,
    channel: str | None = None,
    title: str = "Experiment Result",
) -> bool:
    """Post an experiment result to Slack via incoming webhook.

    Uses ``urllib`` so there is no dependency on ``requests``.

    Parameters
    ----------
    result : dataclass
        Any splita result object.
    webhook_url : str
        Slack incoming webhook URL.
    channel : str or None, default None
        Override the webhook's default channel (e.g. ``"#experiments"``).
    title : str, default "Experiment Result"
        Header text for the Slack message.

    Returns
    -------
    bool
        ``True`` if the webhook returned HTTP 200, ``False`` otherwise.

    Examples
    --------
    >>> from splita._types import ExperimentResult
    >>> r = ExperimentResult(
    ...     control_mean=0.10, treatment_mean=0.12,
    ...     lift=0.02, relative_lift=0.2, pvalue=0.003,
    ...     statistic=2.97, ci_lower=0.007, ci_upper=0.033,
    ...     significant=True, alpha=0.05, method="ztest",
    ...     metric="conversion", control_n=5000,
    ...     treatment_n=5000, power=0.82, effect_size=0.15,
    ... )
    >>> # notify(r, "https://hooks.slack.com/services/T.../B.../xxx")
    """
    payload: dict[str, Any] = {
        "blocks": _build_blocks(result, title),
    }
    if channel is not None:
        payload["channel"] = channel

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False
