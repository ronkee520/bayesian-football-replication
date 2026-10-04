# Reproducibility protocol

## Environment

Use Python 3.10 or 3.11 in a fresh virtual environment and install the project
with `python -m pip install -e ".[dev]"`. The supported dependency ranges are in
`pyproject.toml`. Before the final public release, create a lock file on the
target operating system and record the JAX backend in each result bundle.

## Data controls

1. Confirm the raw CSV SHA-256 matches `data/README.md`.
2. Run `python scripts/validate_data.py`.
3. Confirm 380 matches, 20 teams, 38 matches per team, and no duplicate ordered
   fixtures.
4. Treat generated files in `data/processed/` as disposable derivatives.

## Model runs

Every run is controlled by a tracked YAML file. The workflow copies the resolved
configuration into the result directory, including the random seed. Do not edit
a result table by hand. Change the model, configuration, or reporting code and
rerun the pipeline.

The quick smoke configuration is only a functional test. It is too short for
substantive inference. Final conclusions require the full four-chain settings.

## Acceptance checks for a final run

- no divergent transitions;
- rank-normalized R-hat close to 1, investigated when above 1.01;
- adequate bulk and tail ESS for every reported parameter and quantity;
- BFMI and energy plots checked by chain;
- trace plots and posterior predictive checks visually reviewed;
- conclusions stable across seeds or explained when mixture modes differ;
- full-season model checking clearly separated from held-out forecasting.

## Release checklist

1. Run `pytest -m "not slow"` and `ruff check .`.
2. Run the two optional MCMC smoke tests.
3. Generate full basic, mixture, and holdout results.
4. Add final figures and tables to a versioned release or archive rather than
   committing large posterior arrays to Git.
5. Record package versions with `python -m pip freeze`.
6. Review all paths and commands from a fresh clone.
