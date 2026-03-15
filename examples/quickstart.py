# %% [markdown]
# # splita Quickstart
#
# This notebook walks through the core splita workflow:
# planning, analysis, interpretation, and visualization.

# %% Import splita
import numpy as np

from splita import (
    Experiment,
    SRMCheck,
    SampleSize,
    explain,
    report,
)
from splita.datasets import load_ecommerce

# %% [markdown]
# ## 1. Load sample data
#
# splita ships with deterministic sample datasets for tutorials.

# %% Load e-commerce dataset
data = load_ecommerce()
print(f"Control users: {len(data['control']):,}")
print(f"Treatment users: {len(data['treatment']):,}")
print(f"Description: {data['description']}")

# %% [markdown]
# ## 2. Plan your experiment with SampleSize
#
# Before running an experiment, calculate how many users you need.

# %% Sample size planning
plan = SampleSize.for_proportion(baseline=0.10, mde=0.02, power=0.80)
print(f"Users per variant: {plan.n_per_variant:,}")
print(f"Total users: {plan.n_total:,}")

# Add duration estimate
plan_with_days = plan.duration(daily_users=5000)
print(f"Estimated days: {plan_with_days.days_needed}")

# %% [markdown]
# ## 3. Check data quality with SRM
#
# Always check for Sample Ratio Mismatch before analyzing results.

# %% SRM check
srm = SRMCheck([len(data["control"]), len(data["treatment"])]).run()
print(f"SRM passed: {srm.passed}")
print(f"SRM p-value: {srm.pvalue:.4f}")

# %% [markdown]
# ## 4. Run the experiment
#
# Pass your data and splita auto-detects the metric type.

# %% Run experiment
result = Experiment(data["control"], data["treatment"]).run()
print(result)

# %% [markdown]
# ## 5. Explain the result in plain English
#
# `explain()` converts any result into a human-readable paragraph.

# %% Explain
print(explain(result))

# %% Explain in other languages
print("\n--- Arabic ---")
print(explain(result, lang="ar"))

print("\n--- Spanish ---")
print(explain(result, lang="es"))

print("\n--- Chinese ---")
print(explain(result, lang="zh"))

# %% [markdown]
# ## 6. Visualize (requires matplotlib)
#
# splita has optional visualization support.

# %% Visualization
try:
    from splita.viz import plot_experiment

    fig = plot_experiment(result)
    # fig.savefig("experiment_result.png", dpi=150, bbox_inches="tight")
    print("Plot created successfully.")
except ImportError:
    print("Install matplotlib for visualization: pip install splita[viz]")

# %% [markdown]
# ## 7. Generate a full report
#
# `report()` creates a self-contained HTML report with all sections.

# %% Generate report
html = report(result, srm, title="E-commerce Checkout Test")
print(f"Report generated: {len(html):,} characters of HTML")

# Save to file
# with open("report.html", "w") as f:
#     f.write(html)

# %% [markdown]
# ## 8. Export for academic papers
#
# Use `to_latex()` for LaTeX tables.

# %% LaTeX export
print(result.to_latex())

# %% [markdown]
# ## 9. Serialization
#
# All results are frozen dataclasses with `.to_dict()` and `.to_json()`.

# %% Serialization
d = result.to_dict()
print(f"Dict keys: {list(d.keys())}")

json_str = result.to_json()
print(f"JSON length: {len(json_str)} chars")

# Roundtrip
from splita import ExperimentResult

restored = ExperimentResult.from_dict(d)
assert restored.pvalue == result.pvalue
print("Roundtrip successful!")
