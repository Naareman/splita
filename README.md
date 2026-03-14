# splita

A/B test analysis that is correct by default, informative by design, and composable by construction.

## Installation

```bash
pip install splita
```

## Quick Start

```python
from splita.core import Experiment

result = Experiment(control_data, treatment_data).run()
print(result)
```
