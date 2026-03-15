# splita — TODO

## COMPLETED

- [x] 1. `splita.explain()` — plain-English result interpretation
- [x] 2. Jupyter HTML `_repr_html_()` on all result dataclasses
- [x] 3. Warnings that teach (integrated into explain)
- [x] 4. `splita.viz` — 5 matplotlib plots
- [x] 5. `splita.report()` — HTML/text report generation
- [x] 6. `to_json()` / `from_dict()` serialization
- [x] 7. Cookbook datasets (4 generators)
- [x] 8. GitHub repo created and pushed

---

## TODAY'S TODO

### Infrastructure & CI/CD

- [ ] 9. **CI/CD** — GitHub Actions for test/lint/type-check on Python 3.10-3.13
- [ ] 10. **Pre-commit hooks** — ruff + mypy on every commit
- [ ] 11. **mypy strict** — run mypy --strict and fix type issues
- [ ] 12. **Update TRACKER.md** — currently stale (shows 14 DONE, should show 88)
- [ ] 13. **mkdocs documentation site** — auto-generated API docs, deploy to GitHub Pages
- [ ] 14. **GitHub Actions for docs** — auto-deploy on push to main
- [ ] 15. **Changelog automation** — tag a version -> auto-publish release notes
- [ ] 16. **Working badges** — wire up codecov, CI status (currently placeholders)
- [ ] 17. **GitHub topics** — ab-testing, experimentation, statistics, causal-inference, python

### Community / Open Source

- [ ] 18. **GitHub issue templates** — bug_report.md, feature_request.md
- [ ] 19. **CONTRIBUTING.md** — dev setup, PR guidelines, code style
- [ ] 20. **SECURITY.md** — security policy
- [ ] 21. **CODE_OF_CONDUCT.md** — standard for open source
- [ ] 22. **API stability markers** — `@experimental` decorator for less battle-tested classes

### Killer Features

- [ ] 23. **`splita.check()`** — pre-analysis health report (SRM + covariate balance + flicker + outliers + sensitivity in one call)
- [ ] 24. **`splita.auto()`** — zero-config complete analysis (detect metric, check SRM, handle outliers, apply CUPED, run test, correct for multiple metrics)
- [ ] 25. **`splita.simulate()`** — simulate an A/B test before running it (synthetic data matching your params, show what to expect)
- [ ] 26. **`splita.compare(result_a, result_b)`** — test whether two treatment effects are significantly different
- [ ] 27. **`splita.diagnose(result)`** — structured actionable checklist for any result
- [ ] 28. **`splita.monitor()`** — real-time experiment dashboard data (SRM, effect trajectory, guardrails, predicted end date)
- [ ] 29. **`splita.power_report()`** — visual power analysis dashboard (curves, tables, duration estimates)
- [ ] 30. **`splita.meta_analysis()`** — combine results across experiments (fixed/random effects)
- [ ] 31. **`splita.what_if(result, n=10000)`** — counterfactual projections ("what if we had more users?")
- [ ] 32. **`splita.audit_trail()`** — immutable experiment record with data hash for research integrity

### Ecosystem Integration

- [ ] 33. **Pandas accessor** — `df.splita.experiment(ctrl_col, trt_col)` via register_dataframe_accessor
- [ ] 34. **Polars support** — accept polars Series/DataFrame alongside numpy/pandas
- [ ] 35. **Plugin system** — `splita.register_method("my_test", MyTestClass)` for custom methods
- [ ] 36. **`splita.migrate_from(platform="growthbook")`** — import results from other platforms
- [ ] 37. **Experiment log** — `splita.log(result, name, storage="json")` append to local storage
- [ ] 38. **REST API wrapper** — `splita.serve(port=8080)` one-line FastAPI server

### Export & Communication

- [ ] 39. **LaTeX export** — `result.to_latex()` for academic papers
- [ ] 40. **Slack/email notify** — `splita.notify(result, webhook_url)` auto-post results
- [ ] 41. **Multilingual explain()** — `explain(result, lang="ar")` for global teams
- [ ] 42. **Interactive widget** — ipywidgets for sample size planning in Jupyter

### Documentation & Marketing

- [ ] 43. **Comparison benchmark** — `BENCHMARKS.md` splita vs scipy vs statsmodels
- [ ] 44. **Quickstart notebook** — `examples/quickstart.ipynb` showing all features
- [ ] 45. **Comparison page** — "splita vs GrowthBook vs Eppo" feature matrix

---

## Future (v0.6+)

### Experimentation Accelerator (AI-Powered)
- AI-powered experiment prioritization using content embeddings
- Based on arxiv:2602.13852 (2026)
- Requires LLM/embedding infrastructure — not in scope for pure statistics library

### Video tutorials / blog posts
- Marketing, not code
