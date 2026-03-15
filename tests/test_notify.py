"""Tests for splita.integrations.notify — Slack webhook notification."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import pytest

from splita._types import ExperimentResult, SRMResult
from splita.integrations.notify import _build_blocks, _format_fields, notify


@pytest.fixture
def experiment_result() -> ExperimentResult:
    return ExperimentResult(
        control_mean=0.10,
        treatment_mean=0.12,
        lift=0.02,
        relative_lift=0.20,
        pvalue=0.003,
        statistic=2.97,
        ci_lower=0.007,
        ci_upper=0.033,
        significant=True,
        alpha=0.05,
        method="ztest",
        metric="conversion",
        control_n=5000,
        treatment_n=5000,
        power=0.82,
        effect_size=0.15,
    )


@pytest.fixture
def srm_result() -> SRMResult:
    return SRMResult(
        observed=[4500, 5500],
        expected_counts=[5000.0, 5000.0],
        chi2_statistic=100.0,
        pvalue=0.001,
        passed=False,
        alpha=0.01,
        deviations_pct=[-10.0, 10.0],
        worst_variant=1,
        message="SRM detected.",
    )


class TestFormatFields:
    def test_formats_float_fields(self, experiment_result):
        text = _format_fields(experiment_result)
        assert "*control_mean*" in text
        assert "`0.1000`" in text

    def test_formats_bool_fields(self, experiment_result):
        text = _format_fields(experiment_result)
        assert ":white_check_mark:" in text  # significant=True

    def test_formats_string_fields(self, experiment_result):
        text = _format_fields(experiment_result)
        assert "`ztest`" in text


class TestBuildBlocks:
    def test_has_header_block(self, experiment_result):
        blocks = _build_blocks(experiment_result, "Test Title")
        header = blocks[0]
        assert header["type"] == "header"
        assert header["text"]["text"] == "Test Title"

    def test_has_type_name_block(self, experiment_result):
        blocks = _build_blocks(experiment_result, "Test")
        type_block = blocks[1]
        assert "ExperimentResult" in type_block["text"]["text"]

    def test_includes_explain_text(self, experiment_result):
        blocks = _build_blocks(experiment_result, "Test")
        texts = [b.get("text", {}).get("text", "") for b in blocks]
        combined = " ".join(texts)
        assert "significant" in combined.lower()

    def test_includes_divider(self, experiment_result):
        blocks = _build_blocks(experiment_result, "Test")
        types = [b["type"] for b in blocks]
        assert "divider" in types


class TestNotify:
    def test_returns_false_on_bad_url(self, experiment_result):
        result = notify(experiment_result, "http://localhost:1/nonexistent")
        assert result is False

    def test_returns_true_on_success(self, experiment_result):
        """Test with a local HTTP server that returns 200."""
        received: list[bytes] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(self.rfile.read(length))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args: Any) -> None:
                pass  # Suppress server logs in tests

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request)
        thread.start()

        try:
            result = notify(
                experiment_result,
                f"http://127.0.0.1:{port}/webhook",
                title="CI Test",
            )
            thread.join(timeout=5)
            assert result is True
            assert len(received) == 1
            payload = json.loads(received[0])
            assert "blocks" in payload
        finally:
            server.server_close()

    def test_channel_override(self, experiment_result):
        """Test that channel parameter is included in the payload."""
        received: list[bytes] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(self.rfile.read(length))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args: Any) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request)
        thread.start()

        try:
            notify(
                experiment_result,
                f"http://127.0.0.1:{port}/webhook",
                channel="#testing",
            )
            thread.join(timeout=5)
            payload = json.loads(received[0])
            assert payload["channel"] == "#testing"
        finally:
            server.server_close()
