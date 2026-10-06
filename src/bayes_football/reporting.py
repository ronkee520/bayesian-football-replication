"""Aggregate fitted runs into concise tables, diagnostics, and figures."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import SeasonData, load_season_csv
from .prediction import simulate_scores

REQUIRED_RUN_FILES = {
    "config_resolved.yaml",
    "diagnostics.csv",
    "posterior_predictive_table.csv",
    "posterior_samples.npz",
    "sampler_diagnostics.json",
}


@dataclass
class FittedRun:
    """Files and metadata from one completed full-season fit."""

    name: str
    path: Path
    config: dict[str, Any]
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    sampler: dict[str, Any]
    samples: dict[str, np.ndarray]
    season: SeasonData
    teams: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.config['model']} / {self.config['profile']}"


def discover_runs(results_dir: str | Path, *, include_smoke: bool = False) -> list[Path]:
    """Find completed full-season result directories."""
    root = Path(results_dir)
    if not root.exists():
        return []
    runs = []
    for path in sorted(candidate for candidate in root.iterdir() if candidate.is_dir()):
        if not include_smoke and "smoke" in path.name.lower():
            continue
        if REQUIRED_RUN_FILES.issubset({item.name for item in path.iterdir() if item.is_file()}):
            runs.append(path)
    return runs


def load_fitted_run(path: str | Path, project_root: str | Path) -> FittedRun:
    """Load one completed fit without relying on machine-specific paths."""
    run_path = Path(path)
    missing = sorted(name for name in REQUIRED_RUN_FILES if not (run_path / name).exists())
    if missing:
        raise ValueError(f"Incomplete result directory {run_path}: missing {missing}")

    config = yaml.safe_load((run_path / "config_resolved.yaml").read_text(encoding="utf-8"))
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = Path(project_root) / data_path
    season = load_season_csv(data_path, strict_season=True)
    with np.load(run_path / "posterior_samples.npz") as archive:
        samples = {name: np.asarray(archive[name]) for name in archive.files}

    return FittedRun(
        name=run_path.name,
        path=run_path,
        config=config,
        predictions=pd.read_csv(run_path / "posterior_predictive_table.csv"),
        diagnostics=pd.read_csv(run_path / "diagnostics.csv", index_col=0),
        sampler=json.loads((run_path / "sampler_diagnostics.json").read_text(encoding="utf-8")),
        samples=samples,
        season=season,
        teams=season.teams,
    )


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate compact in-sample posterior-predictive metrics."""
    observed = predictions["observed_on_field_points"].to_numpy(dtype=float)
    predicted = predictions["predicted_points_mean"].to_numpy(dtype=float)
    errors = predicted - observed
    covered = (observed >= predictions["predicted_points_q05"]) & (
        observed <= predictions["predicted_points_q95"]
    )
    return {
        "points_mae": float(np.mean(np.abs(errors))),
        "points_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "points_bias": float(np.mean(errors)),
        "points_correlation": float(np.corrcoef(observed, predicted)[0, 1]),
        "predictive_interval_90_coverage": float(np.mean(covered)),
    }


def model_comparison_table(runs: list[FittedRun]) -> pd.DataFrame:
    """Create one row of predictive and MCMC diagnostics per fitted run."""
    rows = []
    for run in runs:
        home = np.asarray(run.samples["home"])
        metrics = prediction_metrics(run.predictions)
        rows.append(
            {
                "run": run.name,
                "model": run.config["model"],
                "profile": run.config["profile"],
                "chains": int(home.shape[0]) if home.ndim >= 2 else 1,
                "posterior_draws": int(home.size),
                **metrics,
                "divergences": run.sampler.get("divergences"),
                "mean_accept_probability": run.sampler.get("mean_accept_probability"),
                "mean_leapfrog_steps": run.sampler.get("mean_leapfrog_steps"),
                "max_rank_normalized_rhat": run.sampler.get("max_rank_normalized_rhat"),
                "min_bulk_ess": run.sampler.get("min_bulk_ess"),
                "min_tail_ess": run.sampler.get("min_tail_ess"),
                "min_bfmi": min(run.sampler.get("bfmi_by_chain") or [np.nan]),
            }
        )
    return pd.DataFrame(rows)


def _posterior_summary(values: np.ndarray) -> dict[str, float]:
    draws = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=1)),
        "q05": float(np.quantile(draws, 0.05)),
        "median": float(np.quantile(draws, 0.50)),
        "q95": float(np.quantile(draws, 0.95)),
    }


def parameter_summary_table(run: FittedRun) -> pd.DataFrame:
    """Summarize interpretable scalar and team-effect posterior quantities."""
    rows: list[dict[str, Any]] = []
    excluded_prefixes = ("expected_home_goals", "expected_away_goals")
    for name, values in run.samples.items():
        array = np.asarray(values)
        if name in {"attack", "defence", "attack_free", "defence_free"}:
            continue
        if name.startswith(excluded_prefixes):
            continue
        trailing_size = int(np.prod(array.shape[2:])) if array.ndim > 2 else 1
        if trailing_size > 3:
            continue
        if trailing_size == 1:
            rows.append(
                {"run": run.name, "parameter": name, "team": "", **_posterior_summary(array)}
            )
        else:
            flattened = array.reshape(array.shape[0], array.shape[1], trailing_size)
            for index in range(trailing_size):
                rows.append(
                    {
                        "run": run.name,
                        "parameter": f"{name}[{index}]",
                        "team": "",
                        **_posterior_summary(flattened[:, :, index]),
                    }
                )

    for parameter in ("attack", "defence"):
        values = np.asarray(run.samples[parameter])
        for index, team in enumerate(run.teams):
            rows.append(
                {
                    "run": run.name,
                    "parameter": parameter,
                    "team": team,
                    **_posterior_summary(values[..., index]),
                }
            )
    return pd.DataFrame(rows)


def convergence_table(runs: list[FittedRun]) -> pd.DataFrame:
    """Combine interpretable parameter diagnostics without match-level clutter."""
    frames = []
    for run in runs:
        parameter_names = run.diagnostics.index.astype(str)
        excluded = parameter_names.str.startswith(
            ("expected_home_goals", "expected_away_goals", "attack_free", "defence_free")
        )
        frame = run.diagnostics.loc[~excluded].reset_index(names="parameter")
        fixed = frame["r_hat"].isna() & frame["mcse_mean"].fillna(0).eq(0)
        healthy = (
            (frame["r_hat"].fillna(np.inf) <= 1.01)
            & (frame["ess_bulk"].fillna(0) >= 400)
            & (frame["ess_tail"].fillna(0) >= 400)
        )
        frame["status"] = np.select([fixed, healthy], ["fixed", "ok"], default="check")
        frame.insert(0, "run", run.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    return plt


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    _pyplot().close(fig)


def _paper_comparison_runs(runs: list[FittedRun]) -> list[FittedRun]:
    """Select one basic and one mixture run for uncluttered paper-style figures."""
    selected = []
    profile_order = {"paper_replication": 0, "modernized": 1}
    for model in ("basic", "mixture"):
        candidates = [run for run in runs if run.config["model"] == model]
        if candidates:
            selected.append(
                min(candidates, key=lambda run: profile_order.get(run.config["profile"], 2))
            )
    return selected or runs[:2]


def paper_point_comparison_table(runs: list[FittedRun]) -> pd.DataFrame:
    """Create the paper-style observed-versus-predicted league table."""
    selected = _paper_comparison_runs(runs)
    first = selected[0].predictions.set_index("team")
    comparison = first[["observed_on_field_points"]].rename(
        columns={"observed_on_field_points": "observed_points"}
    )
    for run in selected:
        table = run.predictions.set_index("team").reindex(comparison.index)
        if table["predicted_points_mean"].isna().any():
            raise ValueError("Selected runs do not contain the same teams")
        prefix = _safe_name(run.name)
        comparison[f"{prefix}_predicted_mean"] = table["predicted_points_mean"]
        comparison[f"{prefix}_error"] = (
            table["predicted_points_mean"] - comparison["observed_points"]
        )
    return comparison.reset_index().sort_values("observed_points", ascending=False)


def _cumulative_points_from_scores(
    data: SeasonData, home_scores: np.ndarray, away_scores: np.ndarray
) -> np.ndarray:
    """Return cumulative points by posterior draw, team, and team match number."""
    home = np.asarray(home_scores)
    away = np.asarray(away_scores)
    if home.shape != away.shape or home.ndim != 2 or home.shape[1] != data.n_matches:
        raise ValueError("Score arrays must have shape (draws, matches)")

    matches_per_team = np.bincount(
        np.concatenate([data.home_id, data.away_id]), minlength=data.n_teams
    )
    if not np.all(matches_per_team == matches_per_team[0]):
        raise ValueError("Cumulative point plots require equal matches per team")

    cumulative = np.empty((home.shape[0], data.n_teams, matches_per_team[0]), dtype=float)
    totals = np.zeros((home.shape[0], data.n_teams), dtype=float)
    played = np.zeros(data.n_teams, dtype=int)
    for match, (home_team, away_team) in enumerate(
        zip(data.home_id, data.away_id, strict=True)
    ):
        home_win = home[:, match] > away[:, match]
        away_win = home[:, match] < away[:, match]
        draw = ~(home_win | away_win)
        totals[:, home_team] += 3 * home_win + draw
        totals[:, away_team] += 3 * away_win + draw
        cumulative[:, home_team, played[home_team]] = totals[:, home_team]
        cumulative[:, away_team, played[away_team]] = totals[:, away_team]
        played[home_team] += 1
        played[away_team] += 1
    return cumulative


def plot_cumulative_points(runs: list[FittedRun], destination: Path) -> None:
    """Recreate the paper's 20-team cumulative-points small multiples."""
    plt = _pyplot()
    selected = _paper_comparison_runs(runs)
    season = selected[0].season
    for run in selected[1:]:
        if not run.season.frame.equals(season.frame):
            raise ValueError("Cumulative point comparison requires identical match data")

    observed = _cumulative_points_from_scores(
        season, season.home_goals[None, :], season.away_goals[None, :]
    )[0]
    simulations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for run in selected:
        home_scores, away_scores = simulate_scores(
            run.samples,
            run.season,
            seed=int(run.config.get("seed", 0)) + 2718,
            max_draws=1000,
        )
        curves = _cumulative_points_from_scores(run.season, home_scores, away_scores)
        simulations[run.name] = (
            curves.mean(axis=0),
            np.quantile(curves, 0.05, axis=0),
            np.quantile(curves, 0.95, axis=0),
        )

    final_points = observed[:, -1]
    team_order = np.argsort(-final_points)
    colors = ["#2c7fb8", "#e34a33"]
    fig, axes = plt.subplots(4, 5, figsize=(18, 10), sharex=True, sharey=True)
    game_number = np.arange(1, observed.shape[1] + 1)
    for axis, team_index in zip(axes.ravel(), team_order, strict=True):
        axis.plot(
            game_number,
            observed[team_index],
            color="#202020",
            linewidth=1.6,
            label="Observed",
            zorder=4,
        )
        for color, run in zip(colors, selected, strict=False):
            mean, lower, upper = simulations[run.name]
            axis.fill_between(
                game_number,
                lower[team_index],
                upper[team_index],
                color=color,
                alpha=0.09,
                linewidth=0,
            )
            axis.plot(
                game_number,
                mean[team_index],
                color=color,
                linewidth=1.35,
                label=run.label,
                zorder=3,
            )
        axis.set_title(season.teams[team_index], fontsize=10)
        axis.set_xlim(1, observed.shape[1])
    for axis in axes[-1]:
        axis.set_xlabel("Team match number")
    for axis in axes[:, 0]:
        axis.set_ylabel("Cumulative points")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(labels),
        frameon=False,
    )
    fig.suptitle(
        "Posterior predictive cumulative points through the season",
        fontsize=15,
        y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(fig, destination)


def _spread_label_positions(
    values: np.ndarray, lower: float, upper: float, minimum_gap: float
) -> np.ndarray:
    """Spread one-dimensional label positions while preserving their order."""
    order = np.argsort(values)
    positioned = np.clip(np.asarray(values, dtype=float), lower, upper)[order]
    for index in range(1, len(positioned)):
        positioned[index] = max(positioned[index], positioned[index - 1] + minimum_gap)
    if len(positioned) and positioned[-1] > upper:
        positioned -= positioned[-1] - upper
        for index in range(len(positioned) - 2, -1, -1):
            positioned[index] = min(
                positioned[index], positioned[index + 1] - minimum_gap
            )
    if len(positioned) and positioned[0] < lower:
        positioned += lower - positioned[0]
    result = np.empty_like(positioned)
    result[order] = positioned
    return result


def plot_attack_defence(run: FittedRun, destination: Path) -> None:
    """Plot posterior mean attack against defence in the paper's Figure 3 style."""
    plt = _pyplot()
    attack = np.asarray(run.samples["attack"]).reshape(-1, len(run.teams)).mean(axis=0)
    defence = np.asarray(run.samples["defence"]).reshape(-1, len(run.teams)).mean(axis=0)
    fig, axis = plt.subplots(figsize=(10.5, 8.2))
    axis.scatter(defence, attack, s=58, color="#2166ac", edgecolor="white", linewidth=0.7)
    y_padding = max((attack.max() - attack.min()) * 0.06, 0.04)
    y_lower, y_upper = attack.min() - y_padding, attack.max() + y_padding
    x_padding = max((defence.max() - defence.min()) * 0.08, 0.05)
    left_label_x = defence.max() + x_padding
    right_label_x = defence.min() - x_padding
    groups = (
        (np.flatnonzero(defence >= 0), left_label_x, "left"),
        (np.flatnonzero(defence < 0), right_label_x, "right"),
    )
    minimum_gap = (y_upper - y_lower) / 24
    for indices, label_x, alignment in groups:
        label_y = _spread_label_positions(
            attack[indices], y_lower, y_upper, minimum_gap
        )
        for index, position_y in zip(indices, label_y, strict=True):
            axis.annotate(
                run.teams[index],
                (defence[index], attack[index]),
                xytext=(label_x, position_y),
                textcoords="data",
                ha=alignment,
                va="center",
                fontsize=8.2,
                arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.6},
            )
    axis.axhline(0, color="#777777", linestyle="--", linewidth=0.9)
    axis.axvline(0, color="#777777", linestyle="--", linewidth=0.9)
    axis.set_xlim(defence.min() - 2.6 * x_padding, defence.max() + 2.6 * x_padding)
    axis.set_ylim(y_lower, y_upper)
    axis.invert_xaxis()
    axis.set_xlabel("Defence effect (stronger to the right)")
    axis.set_ylabel("Attack effect (stronger upward)")
    axis.set_title(f"Posterior mean attack and defence effects: {run.label}")
    _save_figure(fig, destination)


def _student_t_logpdf(
    values: np.ndarray, means: np.ndarray, scales: np.ndarray, degrees_of_freedom: float
) -> np.ndarray:
    scale = np.clip(scales, 1e-12, None)
    standard = (values - means) / scale
    constant = (
        math.lgamma((degrees_of_freedom + 1.0) / 2.0)
        - math.lgamma(degrees_of_freedom / 2.0)
        - 0.5 * math.log(degrees_of_freedom * math.pi)
    )
    return (
        constant
        - np.log(scale)
        - ((degrees_of_freedom + 1.0) / 2.0)
        * np.log1p(np.square(standard) / degrees_of_freedom)
    )


def mixture_component_probabilities(run: FittedRun, parameter: str) -> pd.DataFrame:
    """Recover posterior group responsibilities from a marginalized mixture fit."""
    if run.config["model"] != "mixture" or parameter not in {"attack", "defence"}:
        raise ValueError("Component probabilities require a mixture run and team effect")

    effects = np.asarray(run.samples[parameter]).reshape(-1, len(run.teams))
    means = np.asarray(run.samples[f"{parameter}_component_means"]).reshape(-1, 3)
    if f"{parameter}_scale" in run.samples:
        scales = np.asarray(run.samples[f"{parameter}_scale"]).reshape(-1, 3)
    else:
        precision = np.asarray(run.samples[f"{parameter}_precision"]).reshape(-1, 3)
        scales = np.reciprocal(np.sqrt(np.clip(precision, 1e-12, None)))

    weights = np.asarray(run.samples[f"{parameter}_weights"])
    if weights.ndim == 3:
        weights = weights.reshape(-1, 1, 3)
        weights = np.broadcast_to(weights, (len(effects), len(run.teams), 3))
    else:
        weights = weights.reshape(-1, len(run.teams), 3)
    if not (len(effects) == len(means) == len(scales) == len(weights)):
        raise ValueError("Mixture posterior arrays have inconsistent draw counts")

    log_density = _student_t_logpdf(
        effects[:, :, None],
        means[:, None, :],
        scales[:, None, :],
        float(run.config.get("degrees_of_freedom", 4.0)),
    )
    log_responsibility = np.log(np.clip(weights, 1e-300, None)) + log_density
    log_responsibility -= log_responsibility.max(axis=-1, keepdims=True)
    responsibility = np.exp(log_responsibility)
    responsibility /= responsibility.sum(axis=-1, keepdims=True)
    probability = responsibility.mean(axis=0)

    # A lower defence effect means stronger defence, so component quality is
    # reversed relative to the numerical ordering of the defence means.
    quality_order = [0, 1, 2] if parameter == "attack" else [2, 1, 0]
    probability = probability[:, quality_order]
    return pd.DataFrame(
        {
            "team": run.teams,
            "Bottom": probability[:, 0],
            "Middle": probability[:, 1],
            "Top": probability[:, 2],
        }
    )


def mixture_membership_table(run: FittedRun) -> pd.DataFrame:
    """Combine attack and defence group probabilities into one export table."""
    attack = mixture_component_probabilities(run, "attack").set_index("team")
    defence = mixture_component_probabilities(run, "defence").set_index("team")
    attack = attack.add_prefix("attack_")
    defence = defence.add_prefix("defence_")
    return attack.join(defence).reset_index()


def plot_mixture_membership(run: FittedRun, destination: Path) -> None:
    """Plot posterior group probabilities in the paper's Figure 5 style."""
    plt = _pyplot()
    attack = mixture_component_probabilities(run, "attack").set_index("team")
    defence = mixture_component_probabilities(run, "defence").set_index("team")
    attack_mean = np.asarray(run.samples["attack"]).reshape(-1, len(run.teams)).mean(axis=0)
    order = np.asarray(run.teams)[np.argsort(-attack_mean)]
    colors = {"Bottom": "#d73027", "Middle": "#fdae61", "Top": "#1a9850"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for axis, table, title in zip(
        axes,
        (attack.loc[order], defence.loc[order]),
        ("(a) Attack effect", "(b) Defence effect"),
        strict=True,
    ):
        bottom = np.zeros(len(order))
        for group in ("Bottom", "Middle", "Top"):
            values = table[group].to_numpy()
            axis.bar(order, values, bottom=bottom, color=colors[group],
                     width=0.86, label=group)
            bottom += values
        axis.set_title(title, fontweight="bold")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=55, labelsize=8)
        axis.set_xlabel("Team (ordered by posterior mean attack)")
    axes[0].set_ylabel("Posterior component probability")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle(f"Posterior mixture-group probabilities: {run.label}", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, destination)


def plot_observed_vs_predicted(runs: list[FittedRun], destination: Path) -> None:
    """Plot actual points against posterior predictive means and intervals."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 5.4), squeeze=False)
    for axis, run in zip(axes[0], runs, strict=True):
        table = run.predictions
        observed = table["observed_on_field_points"].to_numpy()
        predicted = table["predicted_points_mean"].to_numpy()
        lower = predicted - table["predicted_points_q05"].to_numpy()
        upper = table["predicted_points_q95"].to_numpy() - predicted
        axis.errorbar(
            observed,
            predicted,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            color="#1f77b4",
            ecolor="#9ecae1",
            capsize=2,
            alpha=0.9,
        )
        limits = [min(observed.min(), table["predicted_points_q05"].min()) - 2,
                  max(observed.max(), table["predicted_points_q95"].max()) + 2]
        axis.plot(limits, limits, "--", color="#555555", linewidth=1)
        axis.set(xlim=limits, ylim=limits, xlabel="Observed on-field points",
                 ylabel="Posterior predictive points", title=run.label)
    fig.suptitle("Observed versus posterior predictive league points", fontsize=14)
    _save_figure(fig, destination)


def plot_prediction_errors(runs: list[FittedRun], destination: Path) -> None:
    """Compare team-level point prediction errors across fitted runs."""
    plt = _pyplot()
    teams = runs[0].predictions.sort_values("observed_on_field_points")["team"].tolist()
    y = np.arange(len(teams), dtype=float)
    height = 0.75 / len(runs)
    fig, axis = plt.subplots(figsize=(10, 8))
    for index, run in enumerate(runs):
        table = run.predictions.set_index("team").loc[teams]
        errors = table["predicted_points_mean"] - table["observed_on_field_points"]
        offset = (index - (len(runs) - 1) / 2) * height
        axis.barh(y + offset, errors, height=height, label=run.label, alpha=0.82)
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_yticks(y, teams)
    axis.set_xlabel("Prediction error (predicted minus observed points)")
    axis.set_title("Team-level posterior predictive errors")
    axis.legend()
    _save_figure(fig, destination)


def plot_team_effect(run: FittedRun, parameter: str, destination: Path) -> None:
    """Plot posterior means and 90% intervals for attack or defence effects."""
    plt = _pyplot()
    values = np.asarray(run.samples[parameter]).reshape(-1, len(run.teams))
    means = values.mean(axis=0)
    lower, upper = np.quantile(values, [0.05, 0.95], axis=0)
    order = np.argsort(means)
    y = np.arange(len(run.teams))
    fig, axis = plt.subplots(figsize=(9, 8))
    axis.errorbar(
        means[order],
        y,
        xerr=np.vstack([means[order] - lower[order], upper[order] - means[order]]),
        fmt="o",
        color="#1f77b4" if parameter == "attack" else "#d95f02",
        ecolor="#aaaaaa",
        capsize=3,
    )
    axis.axvline(0, linestyle="--", color="#444444", linewidth=1)
    axis.set_yticks(y, np.asarray(run.teams)[order])
    axis.set_xlabel(f"{parameter.title()} effect (posterior mean and 90% interval)")
    direction = "higher is stronger" if parameter == "attack" else "lower is stronger"
    axis.set_title(f"{parameter.title()} effects: {run.label}\n({direction})")
    _save_figure(fig, destination)


def _trace_series(run: FittedRun) -> list[tuple[str, np.ndarray]]:
    scalar_priority = (
        "home",
        "sigma_attack",
        "sigma_defence",
        "tau_attack",
        "tau_defence",
        "weak_attack",
        "strong_attack",
        "weak_defence",
        "strong_defence",
        "weak_attack_magnitude",
        "weak_defence_magnitude",
    )
    series = []
    for name in scalar_priority:
        if name in run.samples and np.asarray(run.samples[name]).ndim == 2:
            series.append((name, np.asarray(run.samples[name])))
    table = run.predictions.set_index("team")
    representative = [table["observed_on_field_points"].idxmax(),
                      table["observed_on_field_points"].idxmin()]
    for parameter in ("attack", "defence"):
        values = np.asarray(run.samples[parameter])
        for team in representative:
            index = run.teams.index(team)
            series.append((f"{parameter}: {team}", values[..., index]))
    return series[:8]


def plot_trace(run: FittedRun, destination: Path) -> None:
    """Plot chain traces for selected interpretable parameters."""
    plt = _pyplot()
    series = _trace_series(run)
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.1 * len(series)), squeeze=False)
    for axis, (label, values) in zip(axes[:, 0], series, strict=True):
        for chain in range(values.shape[0]):
            axis.plot(values[chain], linewidth=0.55, alpha=0.8, label=f"chain {chain + 1}")
        axis.set_ylabel(label)
    axes[0, 0].legend(ncol=min(4, series[0][1].shape[0]), fontsize=8)
    axes[-1, 0].set_xlabel("Draw")
    fig.suptitle(f"Selected MCMC traces: {run.label}", fontsize=14, y=1.0)
    fig.tight_layout()
    _save_figure(fig, destination)


def _ranks(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(flat))
    return ranks.reshape(values.shape)


def plot_rank(run: FittedRun, destination: Path) -> None:
    """Plot within-chain rank histograms for selected parameters."""
    plt = _pyplot()
    series = _trace_series(run)
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.1 * len(series)), squeeze=False)
    for axis, (label, values) in zip(axes[:, 0], series, strict=True):
        ranked = _ranks(values)
        bins = np.linspace(0, ranked.size, 21)
        for chain in range(values.shape[0]):
            axis.hist(ranked[chain], bins=bins, histtype="step", linewidth=1.2,
                      label=f"chain {chain + 1}")
        axis.set_ylabel(label)
    axes[0, 0].legend(ncol=min(4, series[0][1].shape[0]), fontsize=8)
    axes[-1, 0].set_xlabel("Rank across all chains")
    fig.suptitle(f"Selected chain rank plots: {run.label}", fontsize=14, y=1.0)
    fig.tight_layout()
    _save_figure(fig, destination)


def plot_convergence_overview(runs: list[FittedRun], destination: Path) -> None:
    """Create a compact, README-friendly view of multiple MCMC chains."""
    plt = _pyplot()
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 3.1 * len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        selected = _trace_series(run)[:2]
        for axis, (label, values) in zip(axes[row], selected, strict=True):
            for chain in range(values.shape[0]):
                axis.plot(
                    values[chain],
                    linewidth=0.55,
                    alpha=0.8,
                    label=f"chain {chain + 1}",
                )
            axis.set_title(f"{run.label}: {label}")
            axis.set_xlabel("Draw")
            axis.set_ylabel("Posterior value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("Four-chain MCMC convergence overview", fontsize=15, y=1.01)
    fig.tight_layout()
    _save_figure(fig, destination)


def plot_model_comparison(comparison: pd.DataFrame, destination: Path) -> None:
    """Plot predictive accuracy and sampler health in one dashboard."""
    plt = _pyplot()
    labels = comparison["model"] + "\n" + comparison["profile"]
    panels = [
        ("points_mae", "Points MAE", "lower is better"),
        ("points_rmse", "Points RMSE", "lower is better"),
        ("divergences", "Divergences", "zero is ideal"),
        ("min_bulk_ess", "Minimum bulk ESS", "higher is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    for axis, (column, title, subtitle) in zip(axes.ravel(), panels, strict=True):
        values = comparison[column].astype(float)
        colors = [palette[index % len(palette)] for index in range(len(values))]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_title(f"{title}\n{subtitle}")
    fig.suptitle("Model comparison dashboard", fontsize=15)
    fig.tight_layout()
    _save_figure(fig, destination)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = {
        "run": "Run",
        "points_mae": "MAE",
        "points_rmse": "RMSE",
        "points_correlation": "Correlation",
        "predictive_interval_90_coverage": "90% coverage",
        "divergences": "Divergences",
        "max_rank_normalized_rhat": "Max R-hat",
        "min_bulk_ess": "Min bulk ESS",
    }
    lines = ["| " + " | ".join(labels[column] for column in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in frame[columns].itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(comparison: pd.DataFrame, destination: Path) -> None:
    """Write a concise bilingual index for generated tables and figures."""
    columns = [
        "run",
        "points_mae",
        "points_rmse",
        "points_correlation",
        "predictive_interval_90_coverage",
        "divergences",
        "max_rank_normalized_rhat",
        "min_bulk_ess",
    ]
    table = _markdown_table(comparison, columns)
    text = f"""# Result summary / 结果摘要

This file is generated by `scripts/summarize_results.py`.
本文件由 `scripts/summarize_results.py` 自动生成。

## Model comparison / 模型比较

{table}

`MAE`, `RMSE`, correlation, and 90% coverage are in-sample posterior-predictive
checks, not held-out forecast scores. A useful model also needs satisfactory
R-hat, ESS, BFMI, and zero divergent transitions.

`MAE`、`RMSE`、相关系数和 90% 覆盖率属于样本内后验预测检查，不是留出预测成绩。
模型还应同时满足 R-hat、ESS、BFMI 和零发散等采样诊断要求。

## Generated files / 生成文件

- `tables/model_comparison.csv`: one-row-per-model overview / 模型总体比较；
- `tables/posterior_parameter_summary.csv`: posterior summaries / 参数后验摘要；
- `tables/paper_point_predictions.csv`: paper-style league point comparison / 论文式积分比较；
- `tables/mixture_membership_*.csv`: marginalized component probabilities / 边际化组分概率；
- `diagnostics/convergence_summary.csv`: parameter-level diagnostics / 参数级诊断；
- `figures/model_comparison.png`: compact comparison dashboard / 模型比较图；
- `figures/observed_vs_predicted_points.png`: predictive intervals / 积分预测区间；
- `figures/team_prediction_errors.png`: team-level errors / 球队预测误差；
- `figures/paper_cumulative_points.png`: 20-team cumulative points / 20 队累积积分；
- `figures/paper_attack_defence_*.png`: attack-defence plane / 攻防能力平面；
- `figures/paper_group_probabilities_*.png`: mixture responsibilities / 混合组分概率；
- `figures/convergence_overview.png`: compact four-chain traces / 四链轨迹总览；
- `figures/*_effects_*.png`: attack and defence effects / 进攻与防守效应；
- `figures/trace_*.png` and `figures/rank_*.png`: selected-chain diagnostics / 链诊断。

The three `paper_*` figures follow the visual structure of Figures 2/4, 3, and 5
in Baio and Blangiardo (2010), but all values are regenerated from the current
NumPyro fits. Because the mixture allocations are marginalized during NUTS,
the group-probability figure reports posterior component responsibilities rather
than sampled discrete labels. The cumulative-points figure uses the tracked CSV
row order as match order because the course data contain no date column.

三类 `paper_*` 图沿用 Baio 与 Blangiardo（2010）图 2/4、图 3 和图 5 的表达形式，
但数值全部由当前 NumPyro 拟合重新生成。混合模型在 NUTS 中已经边际化离散类别，
因此分组图展示的是后验组分责任概率，而不是离散标签的抽样频率。
由于课程数据没有日期列，累积积分图把仓库中 CSV 的行顺序视为比赛先后顺序。
"""
    destination.write_text(text, encoding="utf-8")


def generate_summary(
    run_paths: list[str | Path], *, project_root: str | Path, results_dir: str | Path
) -> list[Path]:
    """Generate all summary tables, figures, and the bilingual report."""
    project = Path(project_root)
    results = Path(results_dir)
    runs = [load_fitted_run(path, project) for path in run_paths]
    if not runs:
        raise ValueError("No completed full-season runs were found")

    tables_dir = results / "tables"
    diagnostics_dir = results / "diagnostics"
    figures_dir = results / "figures"
    for directory in (tables_dir, diagnostics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparison = model_comparison_table(runs)
    comparison_path = tables_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    parameter_path = tables_dir / "posterior_parameter_summary.csv"
    pd.concat([parameter_summary_table(run) for run in runs], ignore_index=True).to_csv(
        parameter_path, index=False
    )
    paper_points_path = tables_dir / "paper_point_predictions.csv"
    paper_point_comparison_table(runs).to_csv(paper_points_path, index=False)
    convergence_path = diagnostics_dir / "convergence_summary.csv"
    convergence_table(runs).to_csv(convergence_path, index=False)

    generated = [comparison_path, parameter_path, paper_points_path, convergence_path]
    common_figures = {
        "model_comparison.png": lambda path: plot_model_comparison(comparison, path),
        "observed_vs_predicted_points.png": lambda path: plot_observed_vs_predicted(runs, path),
        "team_prediction_errors.png": lambda path: plot_prediction_errors(runs, path),
        "convergence_overview.png": lambda path: plot_convergence_overview(runs, path),
        "paper_cumulative_points.png": lambda path: plot_cumulative_points(runs, path),
    }
    for filename, create in common_figures.items():
        path = figures_dir / filename
        create(path)
        generated.append(path)

    for run in runs:
        suffix = _safe_name(run.name)
        for parameter in ("attack", "defence"):
            path = figures_dir / f"{parameter}_effects_{suffix}.png"
            plot_team_effect(run, parameter, path)
            generated.append(path)
        trace_path = figures_dir / f"trace_{suffix}.png"
        rank_path = figures_dir / f"rank_{suffix}.png"
        plot_trace(run, trace_path)
        plot_rank(run, rank_path)
        generated.extend([trace_path, rank_path])

        if run.config["model"] == "mixture":
            membership_path = tables_dir / f"mixture_membership_{suffix}.csv"
            mixture_membership_table(run).to_csv(membership_path, index=False)
            attack_defence_path = figures_dir / f"paper_attack_defence_{suffix}.png"
            membership_figure_path = figures_dir / f"paper_group_probabilities_{suffix}.png"
            plot_attack_defence(run, attack_defence_path)
            plot_mixture_membership(run, membership_figure_path)
            generated.extend(
                [membership_path, attack_defence_path, membership_figure_path]
            )

    report_path = results / "summary.md"
    write_report(comparison, report_path)
    generated.append(report_path)
    return generated


def report_main(argv: list[str] | None = None) -> int:
    """Command-line entry point for result summaries and figures."""
    parser = argparse.ArgumentParser(description="Summarize fitted Bayesian football models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runs", nargs="*", help="Specific completed result directories")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    run_paths = [Path(path) for path in args.runs] if args.runs else discover_runs(
        results_dir, include_smoke=args.include_smoke
    )
    if not run_paths:
        parser.error(
            "No completed full-season runs found. Run the configured model fits before "
            "generating summaries."
        )
    run_paths = [path if path.is_absolute() else project_root / path for path in run_paths]
    generated = generate_summary(
        run_paths, project_root=project_root, results_dir=results_dir
    )
    print(f"Generated {len(generated)} summary files from {len(run_paths)} fitted runs.")
    print(f"Open {results_dir / 'summary.md'} for the bilingual index.")
    return 0
