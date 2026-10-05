"""Aggregate fitted runs into concise tables, diagnostics, and figures."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import load_season_csv

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
- `diagnostics/convergence_summary.csv`: parameter-level diagnostics / 参数级诊断；
- `figures/model_comparison.png`: compact comparison dashboard / 模型比较图；
- `figures/observed_vs_predicted_points.png`: predictive intervals / 积分预测区间；
- `figures/team_prediction_errors.png`: team-level errors / 球队预测误差；
- `figures/convergence_overview.png`: compact four-chain traces / 四链轨迹总览；
- `figures/*_effects_*.png`: attack and defence effects / 进攻与防守效应；
- `figures/trace_*.png` and `figures/rank_*.png`: selected-chain diagnostics / 链诊断。
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
    convergence_path = diagnostics_dir / "convergence_summary.csv"
    convergence_table(runs).to_csv(convergence_path, index=False)

    generated = [comparison_path, parameter_path, convergence_path]
    common_figures = {
        "model_comparison.png": lambda path: plot_model_comparison(comparison, path),
        "observed_vs_predicted_points.png": lambda path: plot_observed_vs_predicted(runs, path),
        "team_prediction_errors.png": lambda path: plot_prediction_errors(runs, path),
        "convergence_overview.png": lambda path: plot_convergence_overview(runs, path),
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
