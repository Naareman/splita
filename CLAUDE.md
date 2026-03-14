# splita — Development Guide

## What is this?
Python A/B testing library. Correct by default, informative by design, composable by construction.

## Project Structure
```
src/splita/          # Source code (src layout)
tests/               # Mirrors src structure
PRD.md               # Product requirements
splita_api_spec.docx # Detailed API spec
```

## Commands
```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=splita --cov-report=term-missing

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/splita/
```

## Conventions

### Code Style
- Python 3.10+ type hints: `X | Y` not `Optional[X]` or `Union[X, Y]`
- NumPy docstrings (numpydoc format) on all public APIs
- `snake_case` everywhere, no exceptions
- Frozen dataclasses for all results
- Keyword-only config args (after positional data args)

### Error Messages (3-part structure)
```python
raise ValueError(
    "`alpha` must be in (0, 1), got 1.5.\n"
    "  Detail: alpha=1.5 means a 150% false positive rate, which is not meaningful.\n"
    "  Hint: typical values are 0.05, 0.01, or 0.10."
)
```
- Line 1: Problem — use "must" or "can't"
- Line 2: Detail — show actual bad value, explain why it's wrong
- Line 3: Hint (optional) — suggested fix

### Naming
- Related functions share common prefix for autocomplete discovery
- String enums over booleans: `alternative="two-sided"` not `two_sided=True`
- Names are concise but self-documenting

### API Patterns
- Stateful transformers: `fit()` / `transform()` / `fit_transform()` (sklearn)
- Online/streaming: `update()` / `recommend()` / `result()`
- All public methods return frozen dataclasses with `.to_dict()`

### Testing
- Every public method has tests
- Every validation path (ValueError/TypeError) is tested
- Statistical correctness: compare against scipy.stats / known solutions
- Scenario tests: real-world A/B test workflows

### Dependencies
- Required: numpy>=1.24, scipy>=1.10
- Optional: scikit-learn (only for CUPAC)
- No other dependencies. Ever.
