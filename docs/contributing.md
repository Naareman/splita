# Contributing

We welcome contributions to splita. For full details on development setup, code style, testing conventions, and pull request guidelines, see the main contributing guide:

[**CONTRIBUTING.md on GitHub**](https://github.com/Naareman/splita/blob/main/CONTRIBUTING.md)

## Quick start

```bash
git clone https://github.com/Naareman/splita.git
cd splita
pip install -e ".[dev,ml]"
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

## Code style highlights

- **Python 3.10+** type hints: `X | Y`, not `Optional[X]`
- **NumPy docstrings** (numpydoc format)
- **snake_case** everywhere
- **Frozen dataclasses** for all result types with `.to_dict()`
- **3-part error messages**: Problem, Detail, Hint

## Dependencies

- **Required:** numpy, scipy. Nothing else.
- **Optional:** scikit-learn (for ML features only).
- Do not add new dependencies without discussion.
