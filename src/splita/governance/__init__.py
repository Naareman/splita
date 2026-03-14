"""Experiment governance — registry and conflict detection."""

from splita.governance.conflict import ConflictDetector, ConflictResult
from splita.governance.registry import ExperimentRegistry

__all__ = ["ConflictDetector", "ConflictResult", "ExperimentRegistry"]
