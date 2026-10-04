from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("model_name", "profile"),
    [
        ("basic", "modernized"),
        ("basic", "paper_replication"),
        ("mixture", "modernized"),
        ("mixture", "paper_replication"),
    ],
)
def test_tiny_mcmc_smoke(model_name: str, profile: str) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")

    from bayes_football.data import load_season_csv, season_from_frame
    from bayes_football.inference import fit_nuts

    root = Path(__file__).resolve().parents[1]
    full = load_season_csv(root / "data" / "raw" / "epl_2023_24.csv")
    tiny = season_from_frame(full.frame.iloc[:20], teams=full.teams)
    mcmc = fit_nuts(
        tiny,
        model_name=model_name,
        profile=profile,
        sampler={
            "warmup": 5,
            "samples": 5,
            "chains": 1,
            "target_accept": 0.8,
            "chain_method": "sequential",
            "progress_bar": False,
        },
        seed=7,
    )
    samples = mcmc.get_samples()
    assert samples["attack"].shape == (5, full.n_teams)
    assert samples["defence"].shape == (5, full.n_teams)
