"""Command-line workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, save_resolved_config
from .data import load_season_csv, on_field_table, split_last_rounds
from .diagnostics import save_diagnostics
from .inference import fit_nuts, save_posterior_samples
from .prediction import evaluate_heldout, posterior_predictive_table


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the EPL match data")
    parser.add_argument("data_path", nargs="?", default="data/raw/epl_2023_24.csv")
    parser.add_argument("--output", default="data/processed/observed_on_field_table.csv")
    args = parser.parse_args(argv)

    data = load_season_csv(args.data_path, strict_season=True)
    table = on_field_table(data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Validated {data.n_matches} matches and {data.n_teams} teams.")
    print(f"Observed on-field table written to {output}.")
    return 0


def _fit_from_config(config: dict, data):
    return fit_nuts(
        data,
        model_name=config["model"],
        profile=config["profile"],
        sampler=config["sampler"],
        seed=int(config["seed"]),
    )


def fit_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit a configured Bayesian football model")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    data = load_season_csv(config["data_path"], strict_season=True)
    save_resolved_config(config, output_dir)
    mcmc = _fit_from_config(config, data)
    save_posterior_samples(mcmc, output_dir)
    save_diagnostics(mcmc, output_dir)
    samples = mcmc.get_samples(group_by_chain=True)
    table = posterior_predictive_table(
        samples,
        data,
        seed=int(config["seed"]) + 1,
        max_draws=int(config.get("posterior_predictive_draws", 4000)),
    )
    table.to_csv(output_dir / "posterior_predictive_table.csv", index=False)
    print(f"Completed {config['model']} / {config['profile']} fit in {output_dir}")
    return 0


def holdout_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a final-round holdout")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    full_data = load_season_csv(config["data_path"], strict_season=True)
    train, test = split_last_rounds(full_data, int(config["holdout_rounds"]))
    config["training_matches"] = train.n_matches
    config["heldout_matches"] = test.n_matches
    config["holdout_assumption"] = "last n_teams/2 rows form each round"
    save_resolved_config(config, output_dir)

    mcmc = _fit_from_config(config, train)
    save_posterior_samples(mcmc, output_dir)
    save_diagnostics(mcmc, output_dir)
    matches, metrics = evaluate_heldout(
        mcmc.get_samples(group_by_chain=True),
        test,
        seed=int(config["seed"]) + 1,
        max_draws=int(config.get("posterior_predictive_draws", 2000)),
    )
    matches.to_csv(output_dir / "heldout_match_predictions.csv", index=False)
    (output_dir / "heldout_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Completed held-out evaluation in {output_dir}")
    return 0
