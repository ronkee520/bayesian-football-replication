"""Data loading, validation, splitting, and observed league-table utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("home_team", "away_team", "home_goals", "away_goals")


@dataclass(frozen=True)
class SeasonData:
    """A validated match table and its array representation."""

    frame: pd.DataFrame
    teams: tuple[str, ...]
    team_to_id: dict[str, int]
    home_id: np.ndarray
    away_id: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_matches(self) -> int:
        return len(self.frame)


def _validate_frame(frame: pd.DataFrame, strict_season: bool) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = frame.loc[:, REQUIRED_COLUMNS].copy()
    if clean.isna().any().any():
        raise ValueError("Match data contain missing values")

    for column in ("home_team", "away_team"):
        clean[column] = clean[column].astype(str).str.strip()
        if (clean[column] == "").any():
            raise ValueError(f"Blank team name in {column}")

    for column in ("home_goals", "away_goals"):
        numeric = pd.to_numeric(clean[column], errors="raise")
        if (numeric < 0).any() or not np.allclose(numeric, np.floor(numeric)):
            raise ValueError(f"{column} must contain non-negative integers")
        clean[column] = numeric.astype(np.int64)

    if (clean["home_team"] == clean["away_team"]).any():
        raise ValueError("A team cannot play itself")
    if clean.duplicated(["home_team", "away_team"]).any():
        raise ValueError("Duplicate ordered fixture found")

    home_teams = set(clean["home_team"])
    away_teams = set(clean["away_team"])
    if strict_season and home_teams != away_teams:
        raise ValueError("Home-team and away-team sets differ")

    if strict_season:
        n_teams = len(home_teams)
        expected_matches = n_teams * (n_teams - 1)
        if len(clean) != expected_matches:
            raise ValueError(
                f"Expected {expected_matches} matches for a double round-robin, found {len(clean)}"
            )
        expected_home = n_teams - 1
        home_counts = clean["home_team"].value_counts()
        away_counts = clean["away_team"].value_counts()
        if not (home_counts == expected_home).all() or not (away_counts == expected_home).all():
            raise ValueError("Each team must play every other team once at home and once away")

    return clean.reset_index(drop=True)


def season_from_frame(
    frame: pd.DataFrame,
    *,
    strict_season: bool = False,
    teams: tuple[str, ...] | None = None,
) -> SeasonData:
    """Build a :class:`SeasonData` object from an in-memory frame."""
    clean = _validate_frame(frame, strict_season=strict_season)
    team_names = teams or tuple(sorted(set(clean["home_team"]) | set(clean["away_team"])))
    present = set(clean["home_team"]) | set(clean["away_team"])
    if not present.issubset(team_names):
        raise ValueError("Provided team list does not cover all matches")

    team_to_id = {team: index for index, team in enumerate(team_names)}
    return SeasonData(
        frame=clean,
        teams=team_names,
        team_to_id=team_to_id,
        home_id=clean["home_team"].map(team_to_id).to_numpy(dtype=np.int32),
        away_id=clean["away_team"].map(team_to_id).to_numpy(dtype=np.int32),
        home_goals=clean["home_goals"].to_numpy(dtype=np.int32),
        away_goals=clean["away_goals"].to_numpy(dtype=np.int32),
    )


def load_season_csv(path: str | Path, *, strict_season: bool = True) -> SeasonData:
    """Load and validate a match-level CSV file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    return season_from_frame(frame, strict_season=strict_season)


def split_last_rounds(data: SeasonData, rounds: int) -> tuple[SeasonData, SeasonData]:
    """Hold out the last ``rounds`` blocks of ``n_teams / 2`` rows.

    The supplied course data do not contain dates or round identifiers, so this
    function makes the row-order assumption explicit and testable.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if data.n_teams % 2:
        raise ValueError("An even number of teams is required for round splitting")
    holdout_matches = rounds * (data.n_teams // 2)
    if holdout_matches >= data.n_matches:
        raise ValueError("Holdout must leave at least one training match")

    boundary = data.n_matches - holdout_matches
    train = season_from_frame(data.frame.iloc[:boundary], teams=data.teams)
    test = season_from_frame(data.frame.iloc[boundary:], teams=data.teams)
    return train, test


def on_field_table(data: SeasonData) -> pd.DataFrame:
    """Calculate the table implied by match outcomes, before point deductions."""
    rows = {
        team: {"team": team, "played": 0, "won": 0, "drawn": 0, "lost": 0,
               "gf": 0, "ga": 0, "points": 0}
        for team in data.teams
    }

    for match in data.frame.itertuples(index=False):
        home = rows[match.home_team]
        away = rows[match.away_team]
        home["played"] += 1
        away["played"] += 1
        home["gf"] += int(match.home_goals)
        home["ga"] += int(match.away_goals)
        away["gf"] += int(match.away_goals)
        away["ga"] += int(match.home_goals)

        if match.home_goals > match.away_goals:
            home["won"] += 1
            away["lost"] += 1
            home["points"] += 3
        elif match.home_goals < match.away_goals:
            away["won"] += 1
            home["lost"] += 1
            away["points"] += 3
        else:
            home["drawn"] += 1
            away["drawn"] += 1
            home["points"] += 1
            away["points"] += 1

    table = pd.DataFrame(rows.values())
    table["gd"] = table["gf"] - table["ga"]
    table = table.sort_values(
        ["points", "gd", "gf", "team"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    table.insert(0, "position", np.arange(1, len(table) + 1))
    return table
