"""Tests for the experiment log module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from splita import Experiment
from splita.log import load_log, log


class _FakeResult:
    """Minimal result-like object with to_dict()."""

    def to_dict(self) -> dict:
        return {"pvalue": 0.05, "lift": 1.23}


class _NoDict:
    """Object without to_dict()."""

    pass


class TestLog:
    """Tests for log() function."""

    def test_log_creates_file(self, tmp_path: Path) -> None:
        """log() creates a new file and writes an entry."""
        path = str(tmp_path / "test.json")
        result = _FakeResult()
        entry = log(result, "my_experiment", path=path)
        assert entry["experiment_name"] == "my_experiment"
        assert "timestamp" in entry
        assert entry["result"] == {"pvalue": 0.05, "lift": 1.23}

        content = Path(path).read_text()
        assert content.endswith("\n")
        parsed = json.loads(content.strip())
        assert parsed["experiment_name"] == "my_experiment"

    def test_log_appends(self, tmp_path: Path) -> None:
        """Multiple log() calls append to the same file."""
        path = str(tmp_path / "test.json")
        log(_FakeResult(), "exp_1", path=path)
        log(_FakeResult(), "exp_2", path=path)

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["experiment_name"] == "exp_1"
        assert json.loads(lines[1])["experiment_name"] == "exp_2"

    def test_log_with_metadata(self, tmp_path: Path) -> None:
        """Optional metadata is included in the log entry."""
        path = str(tmp_path / "test.json")
        entry = log(_FakeResult(), "exp", path=path, metadata={"team": "growth"})
        assert entry["metadata"] == {"team": "growth"}

    def test_log_with_real_experiment(self, tmp_path: Path) -> None:
        """log() works with a real ExperimentResult."""
        path = str(tmp_path / "test.json")
        rng = np.random.default_rng(42)
        result = Experiment(rng.normal(10, 2, 50), rng.normal(10.5, 2, 50)).run()
        entry = log(result, "real_experiment", path=path)
        assert entry["result"]["method"] == "ttest"

    def test_log_rejects_no_to_dict(self, tmp_path: Path) -> None:
        """ValueError raised for objects without to_dict()."""
        path = str(tmp_path / "test.json")
        with pytest.raises(ValueError, match="to_dict"):
            log(_NoDict(), "bad", path=path)

    def test_log_rejects_empty_name(self, tmp_path: Path) -> None:
        """TypeError raised for empty experiment name."""
        path = str(tmp_path / "test.json")
        with pytest.raises(TypeError, match="non-empty"):
            log(_FakeResult(), "", path=path)

    def test_log_rejects_unsupported_storage(self, tmp_path: Path) -> None:
        """ValueError raised for unsupported storage format."""
        path = str(tmp_path / "test.json")
        with pytest.raises(ValueError, match="Unsupported storage"):
            log(_FakeResult(), "exp", path=path, storage="csv")


class TestLoadLog:
    """Tests for load_log() function."""

    def test_load_log_reads_entries(self, tmp_path: Path) -> None:
        """load_log reads all JSON-lines entries."""
        path = str(tmp_path / "test.json")
        log(_FakeResult(), "exp_1", path=path)
        log(_FakeResult(), "exp_2", path=path)

        entries = load_log(path)
        assert len(entries) == 2
        assert entries[0]["experiment_name"] == "exp_1"
        assert entries[1]["experiment_name"] == "exp_2"

    def test_load_log_empty_file(self, tmp_path: Path) -> None:
        """load_log returns empty list for nonexistent file."""
        path = str(tmp_path / "nonexistent.json")
        assert load_log(path) == []

    def test_load_log_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines in the file are skipped."""
        path = tmp_path / "test.json"
        path.write_text('{"experiment_name": "a"}\n\n{"experiment_name": "b"}\n')
        entries = load_log(str(path))
        assert len(entries) == 2

    def test_load_log_bad_json(self, tmp_path: Path) -> None:
        """ValueError raised for malformed JSON."""
        path = tmp_path / "test.json"
        path.write_text("not json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_log(str(path))
