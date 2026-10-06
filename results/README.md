# Generated results

Model runs create a dedicated subdirectory containing:

- `config_resolved.yaml`: exact data, model, sampler, and seed settings;
- `posterior_samples.npz`: posterior draws;
- `diagnostics.csv` and `sampler_diagnostics.json`;
- `posterior_predictive_table.csv` for full-season fits;
- held-out match predictions and metric summaries for forecast runs.

Generated artifacts are ignored by Git because they can be large and are
recreated from the tracked data, code, and configuration.

## Summary report and figures

After completing at least one full-season fit, run:

```bash
uv run python scripts/summarize_results.py
```

By default, this excludes all `quick_smoke` directories and creates:

- `summary.md`: bilingual result index and model comparison;
- `tables/model_comparison.csv`: compact predictive and sampler metrics;
- `tables/posterior_parameter_summary.csv`: scalar and team-effect summaries;
- `diagnostics/convergence_summary.csv`: interpretable-parameter R-hat, ESS, and status;
- `figures/`: comparison, prediction, team-effect, trace, and rank plots.

`figures/convergence_overview.png` is a compact four-chain trace figure suitable
for reviewing the run. The per-model `trace_*.png` and `rank_*.png` files provide
the detailed convergence views.

A fresh clone contains only this README and `.gitkeep` placeholders. Posterior
archives, summary tables, and figures are ignored by Git and are generated locally
from the tracked data, code, and configurations.
