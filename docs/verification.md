# Verification record

Verification date: 2026-10-06

## Automated checks

- `pytest`: 14 tests passed, including four tiny NUTS runs covering both models
  under both prior profiles.
- `ruff check .`: passed.
- Raw-data validation: 380 matches, 20 teams, no duplicate ordered fixtures,
  19 home and 19 away matches per team.
- Raw-data SHA-256: `1b405d19e804f5221d65472ffc3fa0ca9044ba957c9a83775ca41bb47dab2dc0`.

## End-to-end smoke runs

Three complete pipelines were executed locally:

1. modernized basic model on all 380 matches;
2. modernized mixture model on all 380 matches;
3. modernized basic model on 320 training matches with 60 held-out matches.

Each run successfully produced a resolved configuration, posterior archive,
ArviZ diagnostics, and posterior predictions. Both full-season smoke fits had
zero divergences. The held-out smoke run completed with outcome accuracy 0.617
and a three-class Brier score of 0.536.

These values are functional checks only. The smoke configurations use one short
chain, so R-hat is undefined and the reported accuracy is not used for inference.
Four-chain runs are assessed under `docs/reproducibility.md`.

## Automated reporting

Reporting utilities and unit tests cover cross-model metrics, posterior summaries,
convergence status, and selected trace/rank visualizations. Generated results are
intentionally excluded from Git: a fresh clone produces them only after the user
runs the documented four-chain fits and `scripts/summarize_results.py`. Smoke
directories are excluded from formal summaries by default.
