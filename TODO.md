# splita — TODO

## User Experience Improvements

### 1. `splita.explain()` — Plain-English Result Interpretation
- Takes any result dataclass, returns human-readable interpretation
- "Your treatment increased conversion by 2.0pp (95% CI: 0.7pp to 3.3pp)..."
- Warns about underpowered tests, suggests next steps
- No competitor does this well — potential viral feature

### 2. Jupyter HTML `_repr_html_()` on All Result Dataclasses
- Colored tables, significance indicators
- Every data scientist lives in notebooks
- Add `_repr_html_()` to `_DictMixin` base class

### 3. Warnings That Teach
- When underpowered: "With n=200, you can only detect effects >5.2pp"
- When SRM fails: structured `next_steps` field
- When high skewness: recommend specific alternative method

## Real-World Data Integration

### 4. Visualization Module (`splita.viz`)
- Forest plots for multiple metrics
- Effect over time charts
- Power curves
- Funnel charts
- Optional matplotlib dependency

### 5. Report Generation
- `splita.report(results, format="html")` — self-contained experiment report
- All checks, results, recommendations in one document

## Developer Experience

### 6. Serialization
- `result.to_json()` / `Result.from_dict(d)` roundtrip
- `result.to_parquet()` for bulk storage

### 7. Cookbook with Real Datasets
- 3-5 anonymized A/B test datasets shipped with package
- Worked examples: e-commerce, marketplace, subscription, mobile app
- Problem-oriented: "I'm a marketplace — which methods do I use?"

## Future (v0.6+)

### Experimentation Accelerator (AI-Powered)
- AI-powered experiment prioritization using content embeddings
- Based on arxiv:2602.13852 (2026)
- Requires LLM/embedding infrastructure — not in scope for pure statistics library
