"""Bayesian hierarchical models for football scores."""

from .data import SeasonData, load_season_csv, on_field_table, split_last_rounds

__all__ = [
    "SeasonData",
    "load_season_csv",
    "on_field_table",
    "split_last_rounds",
]

__version__ = "0.1.0"
