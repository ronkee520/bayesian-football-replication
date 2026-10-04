from pathlib import Path

import pytest

from bayes_football.data import load_season_csv, on_field_table, split_last_rounds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "epl_2023_24.csv"


def test_full_season_structure() -> None:
    season = load_season_csv(DATA)
    assert season.n_matches == 380
    assert season.n_teams == 20
    assert set(season.frame["home_goals"].dtype.kind) <= set("iu")


def test_on_field_table_controls() -> None:
    table = on_field_table(load_season_csv(DATA))
    assert table["played"].eq(38).all()
    assert int(table["gf"].sum()) == int(table["ga"].sum())
    assert table.iloc[0]["team"] == "Man City"
    assert int(table.iloc[0]["points"]) == 91
    assert int(table["points"].sum()) == 1058


def test_last_six_round_split() -> None:
    train, test = split_last_rounds(load_season_csv(DATA), 6)
    assert train.n_matches == 320
    assert test.n_matches == 60
    assert train.teams == test.teams


def test_invalid_holdout_is_rejected() -> None:
    season = load_season_csv(DATA)
    with pytest.raises(ValueError):
        split_last_rounds(season, 38)
