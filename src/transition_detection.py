"""
transition_detection.py
-----------------------
Identify all qualifying rest-defence transitions from action_data.

A qualifying transition is a consecutive pair of actions where:
  LOSING action  : ended live, with an active ball-loss EndEvent
  GAINING action : started live, in the losing team's attacking third
                   (StartX < 333 from gaining team's perspective),
                   initiated by an intentional outfield recovery

Both actions must be in the same match and the same (or immediately
adjacent) period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EndEvent values on the LOSING team's action that represent genuine
# live-ball possession losses (active errors / turnovers).
# Excluded: Saved, Miss, Post (shot outcomes), Throw in, Free kick,
#           Corner (dead-ball artefacts appearing with EndToDead=0).
LOSING_END_EVENTS: frozenset[str] = frozenset({
    "Pass",
    "Bad ball control",
    "Failed dribble",
    "Dispossessed",
    "Offensive aerial duel",
    "Tackle",
    "Ball touch",
    "Error",
    "Loose Ball Pick Up",
    "Clearance",
})

# StartEvent values on the GAINING team's action that represent
# intentional, outfield recoveries (no GK-specific events).
GAINING_START_EVENTS: frozenset[str] = frozenset({
    "Loose Ball Pick Up",
    "Interception",
    "Tackle",
    "Dribble",
    "Clearance",
    "Regular play",
})

# StartX threshold (gaining team's attacking perspective, 0-999):
# < 333 means they started in their own defensive third
# = the losing team's attacking third
ATK_THIRD_THRESHOLD = 333

# Maximum frame gap between losing action end_frame and gaining action
# start_frame (accounts for minor alignment differences in the data)
MAX_FRAME_GAP = 2

# Periods to include (1 = 1st half, 2 = 2nd half)
VALID_PERIODS: frozenset[int] = frozenset({1, 2})

# Frame offsets for time windows (500 ms per frame)
FRAMES_PER_SECOND = 2
WINDOW_1S = 1 * FRAMES_PER_SECOND   # 2 frames
WINDOW_5S = 5 * FRAMES_PER_SECOND   # 10 frames


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_rest_defence_transitions(
    action_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scan consecutive action pairs within each match and return a DataFrame
    of qualifying rest-defence transitions.

    Parameters
    ----------
    action_df : loaded and sorted action data (from load_action_data)
    raw_df    : loaded raw tracking data (from load_raw_data)

    Returns
    -------
    DataFrame with one row per transition and the following columns:
        match_id, period,
        losing_team_id, gaining_team_id,
        losing_action_id, gaining_action_id,
        t0_frame,
        t0_1s_frame, t0_5s_frame,
        has_1s_window, has_5s_window,
        losing_end_event, gaining_start_event,
        losing_pass_count, losing_duration_s,
        gaining_start_x
    """
    records = []

    # Work per match to avoid cross-match contamination
    for match_id, match_actions in action_df.groupby("match_id", sort=False):
        match_actions = match_actions.sort_values("start_frame").reset_index(drop=True)

        # Pre-build set of frames present in raw_data for this match
        match_raw = raw_df[raw_df["match_id"] == str(match_id)]
        available_frames: set[int] = set(match_raw["frame"].tolist())
        max_frame = max(available_frames) if available_frames else 0

        n = len(match_actions)
        for idx in range(n - 1):
            losing = match_actions.iloc[idx]
            gaining = match_actions.iloc[idx + 1]

            # ----------------------------------------------------------------
            # Basic pairing checks
            # ----------------------------------------------------------------
            # Must be different teams
            if losing["team_id"] == gaining["team_id"]:
                continue

            # Both in valid periods
            if losing["period"] not in VALID_PERIODS:
                continue
            if gaining["period"] not in VALID_PERIODS:
                continue

            # Frame continuity: gaining starts shortly after losing ends
            frame_gap = int(gaining["start_frame"]) - int(losing["end_frame"])
            if frame_gap < 0 or frame_gap > MAX_FRAME_GAP:
                continue

            # ----------------------------------------------------------------
            # Rules on the LOSING action
            # ----------------------------------------------------------------
            # 1. Ended live
            if int(losing.get("EndToDead", 1)) != 0:
                continue

            # 2. EndEvent is a genuine active ball loss
            losing_end_event = str(losing.get("EndEvent", "")).strip()
            if losing_end_event not in LOSING_END_EVENTS:
                continue

            # ----------------------------------------------------------------
            # Rules on the GAINING action
            # ----------------------------------------------------------------
            # 3. Started live
            if int(gaining.get("StartFromDead", 1)) != 0:
                continue

            # 4. Intentional outfield recovery
            gaining_start_event = str(gaining.get("StartEvent", "")).strip()
            if gaining_start_event not in GAINING_START_EVENTS:
                continue

            # 5. Ball was in losing team's attacking third:
            #    gaining team's StartX < 333  (their defensive third)
            try:
                start_x = float(gaining["StartX"])
            except (TypeError, ValueError):
                continue
            if start_x >= ATK_THIRD_THRESHOLD:
                continue

            # ----------------------------------------------------------------
            # Build transition record
            # ----------------------------------------------------------------
            t0_frame = int(losing["end_frame"])
            t0_1s    = t0_frame + WINDOW_1S
            t0_5s    = t0_frame + WINDOW_5S

            has_1s = (t0_1s in available_frames)
            has_5s = (t0_5s in available_frames)

            records.append({
                "match_id":           str(match_id),
                "period":             int(losing["period"]),
                "losing_team_id":     int(losing["team_id"]),
                "gaining_team_id":    int(gaining["team_id"]),
                "losing_action_id":   int(losing["action_id"]),
                "gaining_action_id":  int(gaining["action_id"]),
                "t0_frame":           t0_frame,
                "t0_1s_frame":        t0_1s,
                "t0_5s_frame":        t0_5s,
                "has_1s_window":      has_1s,
                "has_5s_window":      has_5s,
                "losing_end_event":   losing_end_event,
                "gaining_start_event": gaining_start_event,
                "losing_pass_count":  int(losing.get("PassCount", 0) or 0),
                "losing_duration_s":  float(losing.get("Duration", 0.0) or 0.0),
                "gaining_start_x":    start_x,
            })

    transitions = pd.DataFrame(records)
    if transitions.empty:
        return transitions

    transitions = transitions.sort_values(
        ["match_id", "t0_frame"]
    ).reset_index(drop=True)

    return transitions


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def get_gaining_action(
    transitions_df: pd.DataFrame,
    action_df: pd.DataFrame,
    transition_row: pd.Series,
) -> pd.Series | None:
    """Return the full action_data row for the gaining team's action."""
    mask = (
        (action_df["match_id"] == transition_row["match_id"])
        & (action_df["action_id"] == transition_row["gaining_action_id"])
        & (action_df["team_id"] == transition_row["gaining_team_id"])
    )
    subset = action_df[mask]
    return subset.iloc[0] if len(subset) > 0 else None


def transitions_for_match(
    transitions_df: pd.DataFrame,
    match_id: str,
) -> pd.DataFrame:
    """Filter transitions to a single match."""
    return transitions_df[transitions_df["match_id"] == str(match_id)].reset_index(
        drop=True
    )
