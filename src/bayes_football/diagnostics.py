"""Modern convergence and sampler diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_diagnostics(mcmc, output_dir: str | Path) -> tuple[Path, Path]:
    """Write rank-normalized R-hat, bulk/tail ESS, divergences, and BFMI."""
    import arviz as az

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    inference_data = az.from_numpyro(mcmc)
    summary = az.summary(inference_data, kind="diagnostics", round_to=None)
    summary_path = destination / "diagnostics.csv"
    summary.to_csv(summary_path)

    extra = mcmc.get_extra_fields(group_by_chain=True)
    divergences = np.asarray(extra.get("diverging", []), dtype=bool)
    accept_prob = np.asarray(extra.get("accept_prob", []), dtype=float)
    num_steps = np.asarray(extra.get("num_steps", []), dtype=float)
    try:
        bfmi = np.asarray(az.bfmi(inference_data), dtype=float).tolist()
    except (ValueError, TypeError):
        bfmi = []

    sampler_summary = {
        "divergences": int(divergences.sum()) if divergences.size else None,
        "draws_checked": int(divergences.size) if divergences.size else None,
        "mean_accept_probability": float(accept_prob.mean()) if accept_prob.size else None,
        "mean_leapfrog_steps": float(num_steps.mean()) if num_steps.size else None,
        "bfmi_by_chain": bfmi,
        "max_rank_normalized_rhat": _safe_extreme(summary, "r_hat", "max"),
        "min_bulk_ess": _safe_extreme(summary, "ess_bulk", "min"),
        "min_tail_ess": _safe_extreme(summary, "ess_tail", "min"),
    }
    sampler_path = destination / "sampler_diagnostics.json"
    sampler_path.write_text(
        json.dumps(sampler_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_path, sampler_path


def _safe_extreme(summary, column: str, operation: str) -> float | None:
    if column not in summary or summary[column].dropna().empty:
        return None
    values = summary[column].dropna()
    result = values.max() if operation == "max" else values.min()
    return float(result)
