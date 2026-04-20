"""
data_loading.py
---------------
Load all data sources for the rest-defence analysis pipeline and
derive per-match, per-period attack direction from goalkeeper positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pitch constants (absolute centimetres, origin at centre)
# ---------------------------------------------------------------------------
PITCH_HALF_LENGTH_CM = 5250   # x: -5250 to +5250  (total 10 500 cm = 105 m)
PITCH_HALF_WIDTH_CM  = 3400   # y: -3400 to +3400  (total  6 800 cm =  68 m)

# Third boundaries in absolute x (cm)
THIRD_BOUNDARY_CM = PITCH_HALF_LENGTH_CM / 3   # ≈ 1750 cm

# Player slots
GK_SLOT = 1
OUTFIELD_SLOTS = list(range(2, 12))   # slots 2-11

# Default fps assumed when detection fails
DEFAULT_FPS = 2.0


def detect_fps(raw_df: pd.DataFrame) -> float:
    """
    Detect tracking data framerate from the median inter-frame timestamp interval.
    Uses one match's period 1 data to avoid cross-match contamination.
    Falls back to DEFAULT_FPS (2.0) if data is insufficient.
    """
    if "t" not in raw_df.columns or len(raw_df) < 10:
        return DEFAULT_FPS

    # Use the first match that has at least 10 period-1 frames
    match_col = "match_id" if "match_id" in raw_df.columns else raw_df.columns[0]
    for match_id in raw_df[match_col].unique():
        subset = raw_df[raw_df[match_col] == match_id]
        if "period" in subset.columns:
            subset = subset[subset["period"] == 1]
        if len(subset) < 10:
            continue
        t_sorted = subset["t"].sort_values()
        diffs = t_sorted.diff().dropna()
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            continue
        median_ms = float(diffs.median())
        if median_ms > 0:
            fps = 1000.0 / median_ms
            return round(fps, 1)

    return DEFAULT_FPS


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load ih_raw_data.csv (semicolon-separated).
    Sets frame as int index; coerces numeric columns where pandas reads
    mixed types due to nulls.
    """
    df = pd.read_csv(path,
        sep=";",
        # low_memory=False
    )
    df["frame"] = df["frame"].astype(int)
    df["match_id"] = df["match_id"].astype(str)
    return df.reset_index(drop=True)


def load_action_data(path: str) -> pd.DataFrame:
    """
    Load ih_action_data.csv.
    Parses the ActionPhases column from its string representation
    (e.g. '[1 2 3]' or '[1, 2, 3]') into a Python list[int].
    """
    df = pd.read_csv(path,
        sep=";",
        # low_memory=False
    )
    df["match_id"] = df["match_id"].astype(str)

    def _parse_phases(val) -> list[int]:
        if pd.isna(val) or str(val).strip() in ("", "[]"):
            return []
        s = str(val).strip("[]").replace(",", " ")
        parts = s.split()
        return [int(p) for p in parts if p.lstrip("-").isdigit()]

    if "ActionPhases" in df.columns:
        df["phases_list"] = df["ActionPhases"].apply(_parse_phases)

    df = df.sort_values(["match_id", "start_frame"]).reset_index(drop=True)
    return df


def load_events(path: str) -> pd.DataFrame:
    """Load ih_events.csv."""
    df = pd.read_csv(path,
        sep=";",
        # low_memory=False
    )
    df["match_id"] = df["match_id"].astype(str)
    df = df.sort_values(["match_id", "timestamp"]).reset_index(drop=True)
    return df


def load_matches(path: str) -> pd.DataFrame:
    """Load matchesList_analisi_transizioni.csv."""
    df = pd.read_csv(path,
        sep=";",
        # low_memory=False
    )
    df["matchId"] = df["matchId"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Direction derivation
# ---------------------------------------------------------------------------

def derive_attack_direction(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (match_id, period) determine which direction each team attacks.

    Method: on live frames (game_status == 1), compute mean x_a_1 (home GK).
      - mean(x_a_1) < 0  →  home GK is on the LEFT  →  home team attacks RIGHT
      - mean(x_a_1) > 0  →  home GK is on the RIGHT →  home team attacks LEFT

    Returns a DataFrame indexed by (match_id, period) with columns:
        team_a_attacks_right  : bool
        goal_x_a              : float  (absolute x of team a's own goal)
        goal_x_b              : float  (absolute x of team b's own goal)
        atk_third_x_min_a     : float  (lower x boundary of team a's attacking third)
        atk_third_x_max_a     : float  (upper x boundary of team a's attacking third)
    """
    live = raw_df[raw_df["game_status"] == 1].copy()
    gk_means = (
        live.groupby(["match_id", "period"])["x_a_1"]
        .mean()
        .reset_index()
        .rename(columns={"x_a_1": "mean_gk_x_a"})
    )

    gk_means["team_a_attacks_right"] = gk_means["mean_gk_x_a"] < 0

    # Own-goal x positions
    gk_means["goal_x_a"] = gk_means["team_a_attacks_right"].apply(
        lambda r: -PITCH_HALF_LENGTH_CM if r else PITCH_HALF_LENGTH_CM
    )
    gk_means["goal_x_b"] = gk_means["team_a_attacks_right"].apply(
        lambda r: PITCH_HALF_LENGTH_CM if r else -PITCH_HALF_LENGTH_CM
    )

    # Attacking third of team a
    gk_means["atk_third_x_min_a"] = gk_means["team_a_attacks_right"].apply(
        lambda r: THIRD_BOUNDARY_CM if r else -PITCH_HALF_LENGTH_CM
    )
    gk_means["atk_third_x_max_a"] = gk_means["team_a_attacks_right"].apply(
        lambda r: PITCH_HALF_LENGTH_CM if r else -THIRD_BOUNDARY_CM
    )

    return gk_means.set_index(["match_id", "period"])


# ---------------------------------------------------------------------------
# Player position helpers
# ---------------------------------------------------------------------------

def get_player_positions(
    frame_row: pd.Series,
    team: str,
    include_gk: bool = False,
) -> np.ndarray:
    """
    Extract absolute (x, y) positions (in cm) for all on-pitch players
    of *team* ('a' or 'b') from a single raw_data row.

    Returns ndarray shape (N, 2).  N may be < 10 when players have
    NaN coordinates (e.g. after a red card).
    GK is slot 1; outfield slots are 2-11.
    """
    slots = range(1, 12) if include_gk else OUTFIELD_SLOTS
    positions = []
    for i in slots:
        x_col = f"x_{team}_{i}"
        y_col = f"y_{team}_{i}"
        if x_col not in frame_row.index or y_col not in frame_row.index:
            continue
        x = frame_row[x_col]
        y = frame_row[y_col]
        if pd.notna(x) and pd.notna(y):
            positions.append([float(x), float(y)])
    return np.array(positions) if positions else np.empty((0, 2))


def get_gk_position(frame_row: pd.Series, team: str) -> np.ndarray | None:
    """Return (x, y) of the GK of *team*, or None if missing."""
    x = frame_row.get(f"x_{team}_1")
    y = frame_row.get(f"y_{team}_1")
    if pd.notna(x) and pd.notna(y):
        return np.array([float(x), float(y)])
    return None


def get_direction_info(
    direction_df: pd.DataFrame,
    match_id: str,
    period: int,
) -> pd.Series:
    """
    Retrieve the direction row for a given (match_id, period).
    Raises KeyError if not found.
    """
    return direction_df.loc[(str(match_id), int(period))]


def last_defender_x(
    frame_row: pd.Series,
    losing_team: str,
    team_a_attacks_right: bool,
) -> float:
    """
    Return the x-coordinate (cm) of the most rearward outfield player
    of *losing_team* at this frame.

    'Rearward' means closest to their own goal:
      - team a attacks right  → lowest x among team a outfield
      - team a attacks left   → highest x among team a outfield
    For team b it is the mirror.
    """
    positions = get_player_positions(frame_row, losing_team, include_gk=False)
    if len(positions) == 0:
        # Fallback: use GK position
        gk = get_gk_position(frame_row, losing_team)
        return float(gk[0]) if gk is not None else 0.0

    xs = positions[:, 0]

    if losing_team == "a":
        return float(xs.min()) if team_a_attacks_right else float(xs.max())
    else:  # team b attacks opposite direction to team a
        return float(xs.max()) if team_a_attacks_right else float(xs.min())


# ---------------------------------------------------------------------------
# Raw-data frame lookup helpers
# ---------------------------------------------------------------------------

def get_frame(raw_df: pd.DataFrame, match_id: str, frame: int) -> pd.Series | None:
    """
    Return the raw_data row for (match_id, frame).
    Returns None if not found.
    """
    mask = (raw_df["match_id"] == str(match_id)) & (raw_df["frame"] == int(frame))
    subset = raw_df[mask]
    if len(subset) == 0:
        return None
    return subset.iloc[0]


def build_team_name_map(matches_df: pd.DataFrame) -> dict[tuple[str, int], str]:
    """
    Build a {(match_id, team_id): 'Team Name'} mapping from match metadata.
    """
    mapping: dict[tuple[str, int], str] = {}
    for _, row in matches_df.iterrows():
        mid = str(row["matchId"])
        mapping[(mid, int(row["homeTeam"]))] = str(row["homeTeamName"])
        mapping[(mid, int(row["awayTeam"]))] = str(row["awayTeamName"])
    return mapping


def build_team_label_map(matches_df: pd.DataFrame) -> dict[tuple[str, int], str]:
    """
    Build a {(match_id, team_id): 'a'|'b'} mapping from match metadata.

    matchesList columns used: matchId, homeTeam, awayTeam.
    Home team → 'a', Away team → 'b'.
    """
    mapping: dict[tuple[str, int], str] = {}
    for _, row in matches_df.iterrows():
        mid = str(row["matchId"])
        home = int(row["homeTeam"])
        away = int(row["awayTeam"])
        mapping[(mid, home)] = "a"
        mapping[(mid, away)] = "b"
    return mapping


def build_label_map_from_raw(
    raw_df: pd.DataFrame,
    action_df: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    """
    Derive {(match_id, team_id): 'a'|'b'} from raw_df.team_owner column.
    team_owner=0 → home team → 'a'; team_owner=1 → away team → 'b'.
    Cross-references action_id between raw_df and action_df to find team_ids.
    Falls back to logging a warning if a match cannot be resolved.
    """
    import warnings
    mapping: dict[tuple[str, int], str] = {}
    for match_id in raw_df["match_id"].unique():
        mraw = raw_df[raw_df["match_id"] == match_id]
        mact = action_df[action_df["match_id"] == match_id]
        resolved = {}
        for owner_val, label in [(0, "a"), (1, "b")]:
            owner_frames = mraw[mraw["team_owner"] == owner_val]
            owner_action_ids = owner_frames["action_id"].dropna().unique()
            owner_actions = mact[mact["action_id"].isin(owner_action_ids)]
            if owner_actions.empty:
                warnings.warn(
                    f"build_label_map_from_raw: no actions found for "
                    f"team_owner={owner_val} in match {match_id}"
                )
                continue
            team_id = int(owner_actions["team_id"].mode()[0])
            resolved[label] = team_id
            mapping[(str(match_id), team_id)] = label
        if len(resolved) < 2:
            warnings.warn(
                f"build_label_map_from_raw: could not resolve both teams "
                f"for match {match_id} — some metrics may be incorrect"
            )
    return mapping


def build_name_map_from_team_ids(
    action_df: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    """
    Fallback name map when matchesList is unavailable.
    Uses 'Team_<id>' as display name until teams_metadata.csv is provided.
    """
    mapping: dict[tuple[str, int], str] = {}
    for _, row in action_df[["match_id", "team_id"]].drop_duplicates().iterrows():
        mapping[(str(row["match_id"]), int(row["team_id"]))] = f"Team_{row['team_id']}"
    return mapping


def get_team_label(
    match_id: str,
    team_id: int,
    team_label_map: dict[tuple[str, int], str],
) -> str:
    """
    Return 'a' (home) or 'b' (away) for a numeric team_id in a match.
    Falls back to 'a' if not found.
    """
    return team_label_map.get((str(match_id), int(team_id)), "a")


def get_window_frames(
    raw_df: pd.DataFrame,
    match_id: str,
    t0_frame: int,
    n_frames: int,
) -> pd.DataFrame:
    """
    Return up to *n_frames* rows from raw_data starting at t0_frame
    for the given match_id, ordered by frame number.
    """
    match_raw = raw_df[raw_df["match_id"] == str(match_id)]
    # Frames at t0, t0+1, ..., t0+(n_frames-1)
    target_frames = set(range(int(t0_frame), int(t0_frame) + n_frames))
    window = match_raw[match_raw["frame"].isin(target_frames)].sort_values("frame")
    return window.reset_index(drop=True)
