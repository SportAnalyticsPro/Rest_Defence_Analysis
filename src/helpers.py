"""
helpers.py
----------
Shared utility functions used across main.py and src/* modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data aggregation & formatting helpers
# ---------------------------------------------------------------------------

def format_value(val, fmt_str=".2f", na="—") -> str:
    """Format a numeric value as a string, with NaN handling."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na
    try:
        return format(float(val), fmt_str)
    except (TypeError, ValueError):
        return str(val)


def col_mean(df: pd.DataFrame, col: str) -> float:
    """Mean of a column, returning NaN if column doesn't exist."""
    if col not in df.columns:
        return float("nan")
    return float(df[col].mean(skipna=True))


def col_delta_mean(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Mean of (col_a − col_b) per row, skipping NaN."""
    if col_a not in df.columns or col_b not in df.columns:
        return float("nan")
    delta = df[col_a].astype(float) - df[col_b].astype(float)
    return float(delta.mean(skipna=True))


def pct_bool(df: pd.DataFrame, col: str) -> float:
    """Percentage of True values in a bool-like column (ignores NaN/None)."""
    if col not in df.columns:
        return float("nan")
    vals = df[col].dropna()
    if len(vals) == 0:
        return float("nan")
    return float((vals == True).sum() / len(vals) * 100)  # noqa: E712


def pct_delta(tdf: pd.DataFrame, col_base: str, from_t: int, to_t: int,
              col_mean_fn) -> float:
    """
    Mean percentage change of col_base from offset from_t to offset to_t across transitions.
    Formula: (val_to - val_from) / |val_from| * 100.
    Positive = value INCREASED (e.g. p increased = pressed MORE; ps increased = more pressure).
    Returns NaN if reference mean is zero or NaN.
    col_mean_fn: callable(df, col_name) -> float  (e.g. col_mean)
    """
    ref = col_mean_fn(tdf, f"{col_base}_t{from_t}")
    val = col_mean_fn(tdf, f"{col_base}_t{to_t}")
    if pd.isna(ref) or pd.isna(val) or ref == 0:
        return float("nan")
    return (val - ref) / abs(ref) * 100


# ---------------------------------------------------------------------------
# Event chain inspection
# ---------------------------------------------------------------------------

def check_event_chain(
    events_df: pd.DataFrame,
    match_id,
    action_id: int,
    team_id: int | None = None,
) -> dict:
    """
    Inspect all events in a possession phase (action_id) for clearances and long passes.
    Clearances: event_name or event_group == "Clearance" (case-insensitive).
    Long passes:  event_group == "Pass" AND event_detail contains "Long ball".
    Returns: has_clearance (bool), n_clearances, has_long_pass (bool), n_long_passes.
    """
    mask = (events_df["match_id"] == match_id) & (events_df["action_id"] == action_id)
    if team_id is not None:
        mask &= (events_df["team_id"] == team_id)
    chain = events_df[mask]
    if chain.empty:
        return {"has_clearance": False, "n_clearances": 0,
                "has_long_pass": False, "n_long_passes": 0}
    clearances = chain[
        chain["event_name"].str.strip().str.lower().eq("clearance") |
        chain["event_group"].str.strip().str.lower().eq("clearance")
    ]
    long_passes = chain[
        chain["event_group"].str.strip().str.lower().eq("pass") &
        chain["event_detail"].notna() &
        chain["event_detail"].str.contains("Long ball", case=False, na=False)
    ]
    return {
        "has_clearance": len(clearances) > 0,
        "n_clearances":  int(len(clearances)),
        "has_long_pass": len(long_passes) > 0,
        "n_long_passes": int(len(long_passes)),
    }
