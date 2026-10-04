"""Marginalized heavy-tailed mixture model for team effects."""

from __future__ import annotations

from typing import Any


def mixture_poisson_model(
    home_id: Any,
    away_id: Any,
    home_goals: Any | None = None,
    away_goals: Any | None = None,
    *,
    n_teams: int,
    profile: str = "modernized",
    degrees_of_freedom: float = 4.0,
) -> None:
    """Three-component Student-t model with marginalized allocations.

    Component order is anchored as weak, middle, strong. This avoids the exact
    exchangeability of an unconstrained mixture, although weak separation and
    posterior multimodality can still remain.
    """
    import jax.numpy as jnp
    import jax.scipy as jsp
    import numpyro
    import numpyro.distributions as dist

    home_id = jnp.asarray(home_id, dtype=jnp.int32)
    away_id = jnp.asarray(away_id, dtype=jnp.int32)

    if profile == "paper_replication":
        home = numpyro.sample("home", dist.Normal(0.0, 100.0))
        weak_attack = numpyro.sample(
            "weak_attack", dist.TruncatedNormal(-1.0, 1.0, low=-3.0, high=0.0)
        )
        strong_attack = numpyro.sample(
            "strong_attack", dist.TruncatedNormal(1.0, 1.0, low=0.0, high=3.0)
        )
        weak_defence = numpyro.sample(
            "weak_defence", dist.TruncatedNormal(-1.0, 1.0, low=-3.0, high=0.0)
        )
        strong_defence = numpyro.sample(
            "strong_defence", dist.TruncatedNormal(1.0, 1.0, low=0.0, high=3.0)
        )
        attack_precision = numpyro.sample(
            "attack_precision", dist.Gamma(0.01, 0.01).expand([3]).to_event(1)
        )
        defence_precision = numpyro.sample(
            "defence_precision", dist.Gamma(0.01, 0.01).expand([3]).to_event(1)
        )
        attack_scale = jnp.reciprocal(jnp.sqrt(attack_precision))
        defence_scale = jnp.reciprocal(jnp.sqrt(defence_precision))
        attack_weights = numpyro.sample(
            "attack_weights",
            dist.Dirichlet(jnp.ones(3)).expand([n_teams]).to_event(1),
        )
        defence_weights = numpyro.sample(
            "defence_weights",
            dist.Dirichlet(jnp.ones(3)).expand([n_teams]).to_event(1),
        )
    elif profile == "modernized":
        home = numpyro.sample("home", dist.Normal(0.0, 0.5))
        weak_attack = -numpyro.sample("weak_attack_magnitude", dist.HalfNormal(0.75))
        strong_attack = numpyro.sample("strong_attack", dist.HalfNormal(0.75))
        weak_defence = -numpyro.sample("weak_defence_magnitude", dist.HalfNormal(0.75))
        strong_defence = numpyro.sample("strong_defence", dist.HalfNormal(0.75))
        attack_scale = numpyro.sample(
            "attack_scale", dist.HalfNormal(0.75).expand([3]).to_event(1)
        )
        defence_scale = numpyro.sample(
            "defence_scale", dist.HalfNormal(0.75).expand([3]).to_event(1)
        )
        attack_weights = numpyro.sample("attack_weights", dist.Dirichlet(jnp.ones(3)))
        defence_weights = numpyro.sample("defence_weights", dist.Dirichlet(jnp.ones(3)))
    else:
        raise ValueError(f"Unknown prior profile: {profile}")

    attack_means = jnp.stack([weak_attack, jnp.array(0.0), strong_attack])
    defence_means = jnp.stack([weak_defence, jnp.array(0.0), strong_defence])

    proposal = dist.Normal(0.0, 2.0)
    attack_free = numpyro.sample(
        "attack_free", proposal.expand([n_teams - 1]).to_event(1)
    )
    defence_free = numpyro.sample(
        "defence_free", proposal.expand([n_teams - 1]).to_event(1)
    )
    attack = jnp.concatenate([attack_free, -jnp.sum(attack_free, keepdims=True)])
    defence = jnp.concatenate([defence_free, -jnp.sum(defence_free, keepdims=True)])

    attack_component_lp = dist.StudentT(
        degrees_of_freedom, attack_means, attack_scale
    ).log_prob(attack[:, None])
    defence_component_lp = dist.StudentT(
        degrees_of_freedom, defence_means, defence_scale
    ).log_prob(defence[:, None])

    attack_log_weights = jnp.log(attack_weights)
    defence_log_weights = jnp.log(defence_weights)
    if attack_log_weights.ndim == 1:
        attack_log_weights = jnp.broadcast_to(attack_log_weights, (n_teams, 3))
        defence_log_weights = jnp.broadcast_to(defence_log_weights, (n_teams, 3))

    attack_mixture_lp = jsp.special.logsumexp(
        attack_log_weights + attack_component_lp, axis=-1
    )
    defence_mixture_lp = jsp.special.logsumexp(
        defence_log_weights + defence_component_lp, axis=-1
    )

    # Correct the proposal density, leaving the intended mixture density on the
    # sum-to-zero subspace. The final team effect is deterministic.
    numpyro.factor(
        "attack_mixture_prior",
        jnp.sum(attack_mixture_lp) - jnp.sum(proposal.log_prob(attack_free)),
    )
    numpyro.factor(
        "defence_mixture_prior",
        jnp.sum(defence_mixture_lp) - jnp.sum(proposal.log_prob(defence_free)),
    )

    numpyro.deterministic("attack", attack)
    numpyro.deterministic("defence", defence)
    numpyro.deterministic("attack_component_means", attack_means)
    numpyro.deterministic("defence_component_means", defence_means)

    log_home = home + attack[home_id] + defence[away_id]
    log_away = attack[away_id] + defence[home_id]
    numpyro.deterministic("expected_home_goals", jnp.exp(log_home))
    numpyro.deterministic("expected_away_goals", jnp.exp(log_away))

    with numpyro.plate("matches", home_id.shape[0]):
        numpyro.sample("home_goals", dist.Poisson(jnp.exp(log_home)), obs=home_goals)
        numpyro.sample("away_goals", dist.Poisson(jnp.exp(log_away)), obs=away_goals)
