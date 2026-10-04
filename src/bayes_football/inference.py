"""NUTS fitting and posterior persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .basic_model import basic_poisson_model
from .data import SeasonData
from .mixture_model import mixture_poisson_model

MODEL_REGISTRY = {
    "basic": basic_poisson_model,
    "mixture": mixture_poisson_model,
}


def fit_nuts(
    data: SeasonData,
    *,
    model_name: str,
    profile: str,
    sampler: dict[str, Any],
    seed: int,
):
    """Fit a configured model and return the NumPyro MCMC object."""
    import jax
    from numpyro.infer import MCMC, NUTS

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model = MODEL_REGISTRY[model_name]
    target_accept = float(sampler.get("target_accept", 0.9))
    kernel = NUTS(model, target_accept_prob=target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=int(sampler.get("warmup", 1000)),
        num_samples=int(sampler.get("samples", 1000)),
        num_chains=int(sampler.get("chains", 4)),
        chain_method=str(sampler.get("chain_method", "sequential")),
        progress_bar=bool(sampler.get("progress_bar", True)),
    )
    mcmc.run(
        jax.random.PRNGKey(int(seed)),
        data.home_id,
        data.away_id,
        data.home_goals,
        data.away_goals,
        n_teams=data.n_teams,
        profile=profile,
        extra_fields=("diverging", "energy", "num_steps", "accept_prob"),
    )
    return mcmc


def save_posterior_samples(mcmc, output_dir: str | Path) -> Path:
    """Save chain-grouped posterior samples in a portable NumPy archive."""
    destination = Path(output_dir) / "posterior_samples.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    samples = {
        name: np.asarray(values)
        for name, values in mcmc.get_samples(group_by_chain=True).items()
    }
    np.savez_compressed(destination, **samples)
    return destination
