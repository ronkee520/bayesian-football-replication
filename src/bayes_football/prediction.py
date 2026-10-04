"""Posterior prediction, league tables, and held-out evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .data import SeasonData, on_field_table


def _flatten_draws(values: np.ndarray, trailing_dimensions: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < trailing_dimensions + 1:
        raise ValueError("Posterior array has too few dimensions")
    leading = int(np.prod(array.shape[: array.ndim - trailing_dimensions]))
    return array.reshape((leading,) + array.shape[array.ndim - trailing_dimensions :])


def posterior_rates(
    samples: Mapping[str, np.ndarray], data: SeasonData
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected goals for every posterior draw and match."""
    home = _flatten_draws(np.asarray(samples["home"]), 0).reshape(-1, 1)
    attack = _flatten_draws(np.asarray(samples["attack"]), 1)
    defence = _flatten_draws(np.asarray(samples["defence"]), 1)
    if attack.shape != defence.shape or attack.shape[1] != data.n_teams:
        raise ValueError("Team-effect posterior shapes do not match the data")
    if home.shape[0] != attack.shape[0]:
        raise ValueError("Posterior draw counts are inconsistent")

    log_home = home + attack[:, data.home_id] + defence[:, data.away_id]
    log_away = attack[:, data.away_id] + defence[:, data.home_id]
    return np.exp(log_home), np.exp(log_away)


def simulate_scores(
    samples: Mapping[str, np.ndarray],
    data: SeasonData,
    *,
    seed: int,
    max_draws: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate match scores from posterior expected-goal draws."""
    home_rate, away_rate = posterior_rates(samples, data)
    if max_draws is not None and max_draws < len(home_rate):
        selection = np.linspace(0, len(home_rate) - 1, max_draws, dtype=int)
        home_rate = home_rate[selection]
        away_rate = away_rate[selection]
    rng = np.random.default_rng(seed)
    return rng.poisson(home_rate), rng.poisson(away_rate)


def _points_from_scores(
    data: SeasonData, home_scores: np.ndarray, away_scores: np.ndarray
) -> np.ndarray:
    n_draws = home_scores.shape[0]
    points = np.zeros((n_draws, data.n_teams), dtype=np.int32)
    fixtures = zip(data.home_id, data.away_id, strict=True)
    for match_index, (home_team, away_team) in enumerate(fixtures):
        home_win = home_scores[:, match_index] > away_scores[:, match_index]
        away_win = home_scores[:, match_index] < away_scores[:, match_index]
        draw = ~(home_win | away_win)
        points[:, home_team] += 3 * home_win + draw
        points[:, away_team] += 3 * away_win + draw
    return points


def posterior_predictive_table(
    samples: Mapping[str, np.ndarray],
    data: SeasonData,
    *,
    seed: int,
    max_draws: int | None = None,
) -> pd.DataFrame:
    """Summarize simulated on-field points by team."""
    home_scores, away_scores = simulate_scores(samples, data, seed=seed, max_draws=max_draws)
    points = _points_from_scores(data, home_scores, away_scores)
    observed = on_field_table(data).set_index("team")["points"]
    result = pd.DataFrame(
        {
            "team": data.teams,
            "observed_on_field_points": [int(observed[team]) for team in data.teams],
            "predicted_points_mean": points.mean(axis=0),
            "predicted_points_sd": points.std(axis=0, ddof=1),
            "predicted_points_q05": np.quantile(points, 0.05, axis=0),
            "predicted_points_median": np.quantile(points, 0.50, axis=0),
            "predicted_points_q95": np.quantile(points, 0.95, axis=0),
        }
    )
    result["absolute_error"] = np.abs(
        result["predicted_points_mean"] - result["observed_on_field_points"]
    )
    return result.sort_values("predicted_points_mean", ascending=False).reset_index(drop=True)


def evaluate_heldout(
    samples: Mapping[str, np.ndarray],
    test_data: SeasonData,
    *,
    seed: int,
    max_draws: int = 2000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate score and three-way outcome predictions on held-out matches."""
    home_rate, away_rate = posterior_rates(samples, test_data)
    if max_draws < len(home_rate):
        selection = np.linspace(0, len(home_rate) - 1, max_draws, dtype=int)
        home_rate = home_rate[selection]
        away_rate = away_rate[selection]
    rng = np.random.default_rng(seed)
    sim_home = rng.poisson(home_rate)
    sim_away = rng.poisson(away_rate)

    home_win = sim_home > sim_away
    draw = sim_home == sim_away
    away_win = sim_home < sim_away
    probabilities = np.stack(
        [home_win.mean(axis=0), draw.mean(axis=0), away_win.mean(axis=0)], axis=1
    )

    actual_outcome = np.where(
        test_data.home_goals > test_data.away_goals,
        0,
        np.where(test_data.home_goals == test_data.away_goals, 1, 2),
    )
    one_hot = np.eye(3)[actual_outcome]
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    predicted_outcome = probabilities.argmax(axis=1)

    score_hits = (sim_home == test_data.home_goals) & (sim_away == test_data.away_goals)
    smoothed_score_probability = (score_hits.sum(axis=0) + 0.5) / (len(sim_home) + 1.0)
    mean_home = home_rate.mean(axis=0)
    mean_away = away_rate.mean(axis=0)

    matches = test_data.frame.copy()
    matches["predicted_home_goals"] = mean_home
    matches["predicted_away_goals"] = mean_away
    matches["p_home_win"] = probabilities[:, 0]
    matches["p_draw"] = probabilities[:, 1]
    matches["p_away_win"] = probabilities[:, 2]
    matches["actual_outcome"] = np.array(["H", "D", "A"])[actual_outcome]
    matches["predicted_outcome"] = np.array(["H", "D", "A"])[predicted_outcome]
    matches["score_log_predictive_density"] = np.log(smoothed_score_probability)

    metrics = {
        "matches": int(test_data.n_matches),
        "home_goal_mae": float(np.mean(np.abs(mean_home - test_data.home_goals))),
        "away_goal_mae": float(np.mean(np.abs(mean_away - test_data.away_goals))),
        "outcome_brier_score": float(brier),
        "outcome_accuracy": float(np.mean(predicted_outcome == actual_outcome)),
        "mean_score_log_predictive_density": float(np.mean(np.log(smoothed_score_probability))),
    }
    return matches, metrics
