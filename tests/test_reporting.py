from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayes_football.data import season_from_frame
from bayes_football.reporting import (
    FittedRun,
    _cumulative_points_from_scores,
    convergence_table,
    mixture_component_probabilities,
    model_comparison_table,
    paper_point_comparison_table,
    parameter_summary_table,
    prediction_metrics,
)


def _example_run() -> FittedRun:
    season = season_from_frame(
        pd.DataFrame(
            {
                "home_team": ["Alpha", "Beta", "Gamma"],
                "away_team": ["Beta", "Gamma", "Alpha"],
                "home_goals": [1, 0, 2],
                "away_goals": [0, 0, 1],
            }
        )
    )
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
        season=season,
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


def test_paper_point_comparison_preserves_observed_values_and_errors() -> None:
    comparison = paper_point_comparison_table([_example_run()]).set_index("team")
    assert comparison.loc["Alpha", "observed_points"] == 10
    assert comparison.loc["Alpha", "example_predicted_mean"] == 12
    assert comparison.loc["Alpha", "example_error"] == 2


def test_cumulative_points_follow_each_teams_match_order() -> None:
    run = _example_run()
    cumulative = _cumulative_points_from_scores(
        run.season,
        run.season.home_goals[None, :],
        run.season.away_goals[None, :],
    )
    assert cumulative.shape == (1, 3, 2)
    assert cumulative[0, run.teams.index("Alpha")].tolist() == [3, 3]
    assert cumulative[0, run.teams.index("Beta")].tolist() == [0, 1]
    assert cumulative[0, run.teams.index("Gamma")].tolist() == [1, 4]


def test_marginalized_component_probabilities_are_interpretable() -> None:
    run = _example_run()
    run.config = {"model": "mixture", "profile": "paper_replication"}
    draws = 4
    run.samples.update(
        {
            "attack": np.tile(np.array([[[-1.0, 0.0, 1.0]]]), (1, draws, 1)),
            "defence": np.tile(np.array([[[1.0, 0.0, -1.0]]]), (1, draws, 1)),
            "attack_component_means": np.tile(
                np.array([[[-1.0, 0.0, 1.0]]]), (1, draws, 1)
            ),
            "defence_component_means": np.tile(
                np.array([[[-1.0, 0.0, 1.0]]]), (1, draws, 1)
            ),
            "attack_precision": np.full((1, draws, 3), 100.0),
            "defence_precision": np.full((1, draws, 3), 100.0),
            "attack_weights": np.full((1, draws, 3, 3), 1 / 3),
            "defence_weights": np.full((1, draws, 3, 3), 1 / 3),
        }
    )
    attack = mixture_component_probabilities(run, "attack").set_index("team")
    defence = mixture_component_probabilities(run, "defence").set_index("team")
    assert np.allclose(attack.sum(axis=1), 1)
    assert np.allclose(defence.sum(axis=1), 1)
    assert attack.loc["Alpha", "Bottom"] > 0.99
    assert attack.loc["Gamma", "Top"] > 0.99
    assert defence.loc["Alpha", "Bottom"] > 0.99
    assert defence.loc["Gamma", "Top"] > 0.99
