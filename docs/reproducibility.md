# Reproducibility protocol

## Environment

Use Python 3.10–3.12 and recreate the locked environment with:

```bash
uv sync --extra dev --frozen
```

Dependency ranges are declared in `pyproject.toml`; exact resolved versions are
stored in `uv.lock`. Each result directory records the model configuration and
random seed.

## Data controls

1. Confirm the raw CSV SHA-256 matches `data/README.md`.
2. Run `python scripts/validate_data.py`.
3. Confirm 380 matches, 20 teams, 38 matches per team, and no duplicate ordered
   fixtures.
4. Treat generated files in `data/processed/` as disposable derivatives.

## Model runs

Every run is controlled by a tracked YAML file. The workflow copies the resolved
configuration into the result directory, including the random seed. Changes to
results are made through the model, configuration, or reporting code and then
regenerated.

The quick smoke configuration is a functional test and is not used for inference.
Reported estimates use the full four-chain settings.

## Acceptance checks for a final run

- no divergent transitions;
- rank-normalized R-hat close to 1, investigated when above 1.01;
- adequate bulk and tail ESS for every reported parameter and quantity;
- BFMI and energy plots checked by chain;
- trace plots and posterior predictive checks visually reviewed;
- conclusions stable across seeds or explained when mixture modes differ;
- separate reporting for full-season model checks and held-out forecasts.

## Release checklist

1. Run `uv run pytest` and `uv run ruff check .`.
2. Run the end-to-end smoke configurations.
3. Generate full basic, mixture, and holdout results.
4. Add final figures and tables to a versioned release or archive rather than
   committing large posterior arrays to Git.
5. Retain `uv.lock` and record the JAX backend with the release notes.
6. Review all paths and commands from a fresh clone.
