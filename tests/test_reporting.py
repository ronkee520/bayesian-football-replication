from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayes_football.reporting import (
    FittedRun,
    convergence_table,
    model_comparison_table,
    parameter_summary_table,
    prediction_metrics,
)


def _example_run() -> FittedRun:
    predictions = pd.DataFrame(
        {
            "team": ["Alpha", "Beta", "Gamma"],
            "observed_on_field_points": [10, 20, 30],
            "predicted_points_mean": [12.0, 18.0, 30.0],
            "predicted_points_q05": [8.0, 15.0, 25.0],
            "predicted_points_q95": [15.0, 22.0, 35.0],
        }
    )
    samples = {
        "home": np.arange(12, dtype=float).reshape(2, 6) / 100,
        "tau_attack": np.ones((2, 6)),
        "attack": np.arange(36, dtype=float).reshape(2, 6, 3) / 100,
        "defence": -np.arange(36, dtype=float).reshape(2, 6, 3) / 100,
    }
    diagnostics = pd.DataFrame(
        {
            "mcse_mean": [0.001, 0.0],
            "mcse_sd": [0.001, np.nan],
            "ess_bulk": [500.0, 12.0],
            "ess_tail": [450.0, 12.0],
            "r_hat": [1.0, np.nan],
        },
        index=["home", "fixed_component"],
    )
    return FittedRun(
        name="example",
        path=Path("results/example"),
        config={"model": "basic", "profile": "paper_replication"},
        predictions=predictions,
        diagnostics=diagnostics,
        sampler={
            "divergences": 0,
            "mean_accept_probability": 0.91,
            "mean_leapfrog_steps": 20.0,
            "max_rank_normalized_rhat": 1.0,
            "min_bulk_ess": 500.0,
            "min_tail_ess": 450.0,
            "bfmi_by_chain": [0.9, 1.0],
        },
        samples=samples,
        teams=("Alpha", "Beta", "Gamma"),
    )


def test_prediction_and_comparison_metrics_are_aggregated() -> None:
    run = _example_run()
    metrics = prediction_metrics(run.predictions)
    assert metrics["points_mae"] == 4 / 3
    assert metrics["points_bias"] == 0.0
    assert metrics["predictive_interval_90_coverage"] == 1.0

    comparison = model_comparison_table([run]).iloc[0]
    assert comparison["posterior_draws"] == 12
    assert comparison["divergences"] == 0
    assert comparison["min_bfmi"] == 0.9

    convergence = convergence_table([run]).set_index("parameter")
    assert convergence.loc["home", "status"] == "ok"
    assert convergence.loc["fixed_component", "status"] == "fixed"


def test_parameter_summary_has_scalar_and_named_team_effects() -> None:
    summary = parameter_summary_table(_example_run())
    assert set(summary.loc[summary["parameter"] == "attack", "team"]) == {
        "Alpha",
        "Beta",
        "Gamma",
    }
    assert set(summary.loc[summary["parameter"] == "defence", "team"]) == {
        "Alpha",
        "Beta",
        "Gamma",
    }
    assert {"home", "tau_attack"}.issubset(set(summary["parameter"]))
