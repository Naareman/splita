# Contributing to splita

Thank you for your interest in contributing to splita! This guide will help you get started.

## Development setup

```bash
# Clone the repo
git clone https://github.com/Naareman/splita.git
cd splita

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,ml]"

# Install pre-commit hooks
pre-commit install
```

## Running checks

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/splita/

# Tests
pytest

# Tests with coverage
pytest --cov=splita --cov-report=term-missing
```

## Pull request guidelines

1. **Fork and branch.** Create a feature branch from `main` (e.g. `feat/my-feature` or `fix/issue-42`).
2. **Write tests first.** Every public method needs tests covering the happy path, edge cases, and validation errors.
3. **Follow existing patterns.** Look at similar classes in the codebase for API conventions.
4. **Keep PRs focused.** One feature or fix per PR. Small PRs are reviewed faster.
5. **Write a clear description.** Explain what the PR does and why, not just what files changed.

## Code style

- **Python 3.10+** type hints: use `X | Y`, not `Optional[X]` or `Union[X, Y]`.
- **NumPy docstrings** (numpydoc format) on all public APIs.
- **snake_case** everywhere, no exceptions.
- **Frozen dataclasses** for all result types, with `.to_dict()`.
- **Keyword-only** config args (after positional data args).
- **Error messages** follow the 3-part structure: Problem, Detail, Hint.

```python
raise ValueError(
    "`alpha` must be in (0, 1), got 1.5.\n"
    "  Detail: alpha=1.5 means a 150% false positive rate, which is not meaningful.\n"
    "  Hint: typical values are 0.05, 0.01, or 0.10."
)
```

## Test conventions

- Mirror the `src/` structure in `tests/` (e.g. `src/splita/core/` -> `tests/core/`).
- Test file names: `test_<module>.py`.
- Use `pytest` fixtures for shared setup.
- Statistical correctness tests: compare against `scipy.stats` or known analytic solutions.
- Every `ValueError` / `TypeError` validation path must have a test.

## Dependencies

- **Required:** numpy, scipy. Nothing else.
- **Optional:** scikit-learn (for CUPAC/ML features only).
- Do not add new dependencies without discussion in an issue first.
