# Bayesian football model replication

This repository reproduces and extends the hierarchical football models in
Baio and Blangiardo (2010) on the 2023/24 English Premier League season.
It is designed as a self-contained Bayesian workflow: the raw match data,
model code, configurations, diagnostics, tests, and reproducibility notes live
in this repository.

## Research questions

1. Can a hierarchical Poisson model recover interpretable attack, defence, and
   home-advantage effects?
2. Does a heavy-tailed three-component mixture improve posterior predictive fit
   for unusually strong or weak teams?
3. Do apparent in-sample improvements persist when the final rounds are held out?

## Models

For match \(g\), home and away goals are conditionally independent Poisson variables:

\[
y_{g,h}\sim\operatorname{Poisson}(\theta_{g,h}),\qquad
y_{g,a}\sim\operatorname{Poisson}(\theta_{g,a}).
\]

The basic model uses

\[
\log\theta_{g,h}=\text{home}+\text{attack}_{h(g)}+\text{defence}_{a(g)},
\]

\[
\log\theta_{g,a}=\text{attack}_{a(g)}+\text{defence}_{h(g)}.
\]

The mixture model replaces the single population distribution for team effects
with weak, middle, and strong Student-t components. Discrete component labels
are marginalized so NUTS can sample the continuous posterior directly.

Two prior profiles are kept deliberately separate:

- `paper_replication`: follows the paper/BUGS formulation as closely as practical.
- `modernized`: uses non-centred effects and regularising scale priors.

The first profile answers a replication question. The second is the recommended
statistical model for new analysis.

## Quick start

The reproducible path uses the tracked `uv.lock` file:

```bash
uv sync --extra dev
```

Alternatively, create a standard virtual environment and install the package:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Validate the included data and reproduce the observed on-field league table:

```bash
python scripts/validate_data.py
```

Run a short basic-model smoke fit:

```bash
python scripts/run_model.py --config configs/quick_smoke.yaml
```

Run the full modernized basic and mixture models:

```bash
python scripts/run_model.py --config configs/basic_modernized.yaml
python scripts/run_model.py --config configs/mixture_modernized.yaml
```

Run the held-out final-six-round experiment:

```bash
python scripts/run_holdout.py --config configs/holdout_modernized.yaml
```

Generated files are written under `results/` and are intentionally excluded
from version control. Every command records the resolved configuration and seed.

## Evaluation

The project separates two questions that should not be conflated:

- **Replication / model checking:** fit the full season and perform posterior
  predictive checks on the same matches.
- **Forecasting:** fit only the earlier matches and evaluate log predictive
  density, outcome Brier score, score MAE, and calibration on held-out matches.

League points in this project are **on-field points before administrative
deductions**. This is appropriate for a score model but differs from the official
2023/24 table for Everton and Nottingham Forest.

## Repository guide

- `data/raw/`: immutable input data and provenance.
- `src/bayes_football/`: independent model and workflow implementation.
- `configs/`: reproducible prior and sampler settings.
- `scripts/`: command-line entry points.
- `tests/`: data, likelihood, prediction, and optional MCMC checks.
- `docs/methods_zh.md`: detailed Chinese explanation of the Bayesian methods.
- `docs/replication_notes.md`: paper-to-code choices and known discrepancies.
- `docs/reproducibility.md`: exact reproduction protocol.
- `docs/verification.md`: tests and end-to-end smoke-run evidence.

## Reference

Baio, G. and Blangiardo, M. (2010). Bayesian hierarchical model for the
prediction of football results. *Journal of Applied Statistics*, 37(2),
253–264. <https://doi.org/10.1080/02664760802684177>
