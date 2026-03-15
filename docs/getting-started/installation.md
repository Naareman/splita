# Installation

## Basic install

```bash
pip install splita
```

This installs splita with its only required dependencies: **numpy** and **scipy**.

## Optional extras

splita ships optional extras for additional functionality. Install them as needed:

=== "ML (CUPAC)"

    ```bash
    pip install splita[ml]
    ```

    Adds **scikit-learn** for ML-powered variance reduction (CUPAC, CausalForest, HTEEstimator).

=== "Visualization"

    ```bash
    pip install splita[viz]
    ```

    Adds **matplotlib** for built-in plotting of results, confidence intervals, and power curves.

=== "Jupyter Widgets"

    ```bash
    pip install splita[widget]
    ```

    Adds **ipywidgets** for interactive experiment dashboards in Jupyter notebooks.

=== "API Server"

    ```bash
    pip install splita[api]
    ```

    Adds **FastAPI** and **uvicorn** for serving splita as a REST API via `splita.serve()`.

=== "Everything"

    ```bash
    pip install splita[ml,viz,widget,api]
    ```

## Development install

For contributing to splita:

```bash
git clone https://github.com/Naareman/splita.git
cd splita
pip install -e ".[dev,ml]"
```

This installs splita in editable mode with dev tools: pytest, ruff, mypy, and pytest-cov.

## Requirements

- Python 3.10 or later
- numpy >= 1.24
- scipy >= 1.10

## Verify installation

```python
import splita
print(splita.__version__)
```

```python
from splita import Experiment
result = Experiment([0, 1, 0], [1, 1, 1]).run()
print(result.method)  # 'ztest'
```
