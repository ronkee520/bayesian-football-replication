# Generated results

Model runs create a dedicated subdirectory containing:

- `config_resolved.yaml`: exact data, model, sampler, and seed settings;
- `posterior_samples.npz`: posterior draws;
- `diagnostics.csv` and `sampler_diagnostics.json`;
- `posterior_predictive_table.csv` for full-season fits;
- held-out match predictions and metric summaries for forecast runs.

Generated artifacts are ignored by Git because they can be large and are
recreated from the tracked data, code, and configuration.
