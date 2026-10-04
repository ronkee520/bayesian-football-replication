"""Basic hierarchical Poisson model."""

from __future__ import annotations

from typing import Any

import numpy as np


def score_log_rates(
    home_id: np.ndarray,
    away_id: np.ndarray,
    home_advantage: float,
    attack: np.ndarray,
    defence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log expected goals using the paper's sign convention."""
    return (
        home_advantage + attack[home_id] + defence[away_id],
        attack[away_id] + defence[home_id],
    )


def basic_poisson_model(
    home_id: Any,
    away_id: Any,
    home_goals: Any | None = None,
    away_goals: Any | None = None,
    *,
    n_teams: int,
    profile: str = "modernized",
) -> None:
    """NumPyro implementation of the basic model.

    ``paper_replication`` keeps the broad precision priors and the asymmetric
    final-team sum-to-zero construction used in the BUGS appendix.
    ``modernized`` uses non-centred effects and regularising scale priors.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    home_id = jnp.asarray(home_id, dtype=jnp.int32)
    away_id = jnp.asarray(away_id, dtype=jnp.int32)

    if profile == "paper_replication":
        home = numpyro.sample("home", dist.Normal(0.0, 100.0))
        mu_attack = numpyro.sample("mu_attack", dist.Normal(0.0, 100.0))
        mu_defence = numpyro.sample("mu_defence", dist.Normal(0.0, 100.0))
        tau_attack = numpyro.sample("tau_attack", dist.Gamma(0.1, 0.1))
        tau_defence = numpyro.sample("tau_defence", dist.Gamma(0.1, 0.1))
        attack_free = numpyro.sample(
            "attack_free",
            dist.Normal(mu_attack, jnp.reciprocal(jnp.sqrt(tau_attack)))
            .expand([n_teams - 1])
            .to_event(1),
        )
        defence_free = numpyro.sample(
            "defence_free",
            dist.Normal(mu_defence, jnp.reciprocal(jnp.sqrt(tau_defence)))
            .expand([n_teams - 1])
            .to_event(1),
        )
        attack = jnp.concatenate([attack_free, -jnp.sum(attack_free, keepdims=True)])
        defence = jnp.concatenate([defence_free, -jnp.sum(defence_free, keepdims=True)])
    elif profile == "modernized":
        home = numpyro.sample("home", dist.Normal(0.0, 0.5))
        sigma_attack = numpyro.sample("sigma_attack", dist.HalfNormal(0.75))
        sigma_defence = numpyro.sample("sigma_defence", dist.HalfNormal(0.75))
        attack_raw = numpyro.sample(
            "attack_raw", dist.Normal(0.0, 1.0).expand([n_teams]).to_event(1)
        )
        defence_raw = numpyro.sample(
            "defence_raw", dist.Normal(0.0, 1.0).expand([n_teams]).to_event(1)
        )
        attack = sigma_attack * (attack_raw - jnp.mean(attack_raw))
        defence = sigma_defence * (defence_raw - jnp.mean(defence_raw))
    else:
        raise ValueError(f"Unknown prior profile: {profile}")

    numpyro.deterministic("attack", attack)
    numpyro.deterministic("defence", defence)
    log_home = home + attack[home_id] + defence[away_id]
    log_away = attack[away_id] + defence[home_id]
    numpyro.deterministic("expected_home_goals", jnp.exp(log_home))
    numpyro.deterministic("expected_away_goals", jnp.exp(log_away))

    with numpyro.plate("matches", home_id.shape[0]):
        numpyro.sample("home_goals", dist.Poisson(jnp.exp(log_home)), obs=home_goals)
        numpyro.sample("away_goals", dist.Poisson(jnp.exp(log_away)), obs=away_goals)
