from pathlib import Path

import numpy as np

from bayes_football.data import load_season_csv
from bayes_football.prediction import posterior_rates

ROOT = Path(__file__).resolve().parents[1]


def test_zero_effects_produce_unit_rates_without_home_advantage() -> None:
    season = load_season_csv(ROOT / "data" / "raw" / "epl_2023_24.csv")
    samples = {
        "home": np.zeros((2, 3)),
        "attack": np.zeros((2, 3, season.n_teams)),
        "defence": np.zeros((2, 3, season.n_teams)),
    }
    home_rate, away_rate = posterior_rates(samples, season)
    assert home_rate.shape == (6, season.n_matches)
    assert away_rate.shape == (6, season.n_matches)
    np.testing.assert_allclose(home_rate, 1.0)
    np.testing.assert_allclose(away_rate, 1.0)
