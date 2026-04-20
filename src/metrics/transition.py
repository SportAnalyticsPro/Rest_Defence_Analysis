"""
metrics/transition.py
---------------------
Transition-phase metrics, split into two categories:

NEGATIVE TRANSITION METRICS (how well the defending team handles losing the ball):
  8.  Ball Regain Dynamics  (action-level + raw-data, extended to 10s)
  9.  Transition Rating     (qualitative: Best/Good/Okay/Bad)
  10. Structural Prevention Efficiency (match-level aggregate, 10s window)

POSITIVE TRANSITION METRICS (how well the gaining team exploits the transition):
  11. Constructive Progression Rate  (≥3 consecutive passes within 15s)
  12. Own Half Exit Efficiency       (progressive pass crossing midfield within 15s)
  13. Playmaker Dependency Index     (first pass directed to auto-identified playmaker)
  14. Productive Pass Ratio          (% passes with positive xT within 15s)
"""

from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd

from ..data_loading import (
    PITCH_HALF_LENGTH_CM,
    THIRD_BOUNDARY_CM,
    get_player_positions,
    get_frame,
    get_window_frames,
)
from ..rest_defence_area import RestDefenceZone

_DEFAULT_FPS = 2.0   # overridden at runtime via fps parameter

# Pass-like action events used to define a pass chain
PASS_EVENTS = {"Pass", "pass", "Cross", "cross"}


# ===========================================================================
# NEGATIVE TRANSITION METRICS
# ===========================================================================

# ---------------------------------------------------------------------------
# Metric 8 — Ball Regain Dynamics
# ---------------------------------------------------------------------------

def ball_regain_dynamics(
    transition_row: pd.Series,
    gaining_action_row: pd.Series | None,
    raw_df: pd.DataFrame,
    losing_team_label: str,
    team_a_attacks_right: bool,
    fps: float = _DEFAULT_FPS,
) -> dict:
    """
    Returns dynamics at both 5s and 10s windows:
        duration_s               : duration of gaining team's action
        pass_count               : passes by gaining team
        centroid_advance_5s_m    : losing team centroid shift t0→t0+5s (metres)
        centroid_advance_10s_m   : losing team centroid shift t0→t0+10s (metres)
    """
    result = {
        "duration_s":             float("nan"),
        "pass_count":             0.0,            # null → 0 (no passes recorded = 0)
        "centroid_advance_5s_m":  float("nan"),
        "centroid_advance_10s_m": float("nan"),
    }

    if gaining_action_row is not None:
        result["duration_s"] = float(gaining_action_row.get("Duration") or float("nan"))
        pc_raw = gaining_action_row.get("PassCount")
        result["pass_count"] = float(pc_raw) if pd.notna(pc_raw) else 0.0

    t0_frame = int(transition_row["t0_frame"])
    match_id = transition_row["match_id"]

    row_t0  = get_frame(raw_df, match_id, t0_frame)
    row_t5  = get_frame(raw_df, match_id, t0_frame + int(5 * fps))
    row_t10 = get_frame(raw_df, match_id, t0_frame + int(10 * fps))

    def _centroid_advance_m(row_end: pd.Series) -> float:
        """Centroid advance in metres (positive = toward opponent goal)."""
        if row_t0 is None or row_end is None:
            return float("nan")
        pos_t0  = get_player_positions(row_t0,  losing_team_label, include_gk=False)
        pos_end = get_player_positions(row_end, losing_team_label, include_gk=False)
        if len(pos_t0) == 0 or len(pos_end) == 0:
            return float("nan")
        cx_t0  = float(pos_t0[:, 0].mean())
        cx_end = float(pos_end[:, 0].mean())
        ar = (losing_team_label == "a") == team_a_attacks_right
        advance_cm = (cx_end - cx_t0) if ar else -(cx_end - cx_t0)
        return advance_cm / 100.0   # cm → metres

    if row_t0 is not None and row_t5 is not None:
        result["centroid_advance_5s_m"] = _centroid_advance_m(row_t5)

    if row_t0 is not None and row_t10 is not None:
        result["centroid_advance_10s_m"] = _centroid_advance_m(row_t10)

    return result


# ---------------------------------------------------------------------------
# Metric 9 — Transition Rating
# ---------------------------------------------------------------------------

def transition_rating(
    transition_row: pd.Series,
    action_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    losing_team_label: str,
    team_a_attacks_right: bool,
    fps: float = _DEFAULT_FPS,
) -> str:
    """
    Qualitative rating of the transition outcome (worst-case logic):

    Best : Losing team regains possession within 5s
    Bad  : Shot conceded within 15s, OR ball enters defensive third within 5s,
           OR gaining team plays a pass in-behind the defence within 15s
    Good : Ball goes out of play (EndToDead=1) within 15s, OR gaining team
           commits a foul (Foul conceded) within 15s, OR losing team regains
           possession within 5-15s
    Okay : Gaining team was fouled (Foul suffered) with ≤1 player advantage for
           losing team, OR attack persists beyond 15s

    When multiple outcomes are possible, returns the worst (most protective):
    Bad > Okay > Good > Best. If nothing is found, returns Okay.
    """
    match_id          = transition_row["match_id"]
    t0_frame          = int(transition_row["t0_frame"])
    t5_frame          = t0_frame + int(5 * fps)
    t15_frame         = t0_frame + int(15 * fps)
    gaining_team_id   = int(transition_row["gaining_team_id"])
    losing_team_id    = int(transition_row["losing_team_id"])

    ar = (losing_team_label == "a") == team_a_attacks_right
    outcomes = set()

    # ── Priority 1: Best — losing team regains within 5s ──────────────────
    if transition_row.get("has_5s_window", False):
        regains = action_df[
            (action_df["match_id"] == match_id)
            & (action_df["team_id"] == losing_team_id)
            & (action_df["start_frame"] > t0_frame)
            & (action_df["start_frame"] <= t5_frame)
        ]
        if len(regains) > 0:
            return "Best"

    # ── Priority 2: Bad ───────────────────────────────────────────────────
    gaining_15s = action_df[
        (action_df["match_id"] == match_id)
        & (action_df["team_id"] == gaining_team_id)
        & (action_df["start_frame"] >= t0_frame)
        & (action_df["start_frame"] <= t15_frame)
    ]

    # Shot conceded within 15s
    shot_found = False
    for _, act in gaining_15s.iterrows():
        if str(act.get("EndEvent", "")).strip() in {"Saved", "Miss", "Goal", "Post"}:
            shot_found = True
            break
    if shot_found:
        outcomes.add("Bad")

    # Ball enters defensive third within 5s
    if transition_row.get("has_5s_window", False):
        window = get_window_frames(raw_df, match_id, t0_frame, int(5 * fps))
        def_third_found = False
        for _, frow in window.iterrows():
            bx = frow.get("x_ball")
            if pd.isna(bx):
                continue
            bx = float(bx)
            if (ar and bx < -THIRD_BOUNDARY_CM) or (not ar and bx > THIRD_BOUNDARY_CM):
                def_third_found = True
                break
        if def_third_found:
            outcomes.add("Bad")

    # Pass in-behind the defence within 15s (StartX > 750 in gaining-team coords)
    inbehind_found = False
    for _, act in gaining_15s.iterrows():
        start_x = act.get("StartX")
        if pd.notna(start_x) and float(start_x) > 750:
            inbehind_found = True
            break
    if inbehind_found:
        outcomes.add("Bad")

    # ── Priority 3: Good ──────────────────────────────────────────────────
    # Ball out of play within 15s (EndToDead=1)
    for _, act in gaining_15s.iterrows():
        if int(act.get("EndToDead", 0) or 0) == 1:
            outcomes.add("Good")

    # Gaining team commits a foul (Foul conceded) within 15s → Good
    for _, act in gaining_15s.iterrows():
        end_event = str(act.get("EndEvent", "")).strip().lower()
        if end_event == "foul conceded":
            outcomes.add("Good")

    # Losing team regains possession between 5s and 15s → Good
    regains_5_15 = action_df[
        (action_df["match_id"] == match_id)
        & (action_df["team_id"] == losing_team_id)
        & (action_df["start_frame"] > t5_frame)
        & (action_df["start_frame"] <= t15_frame)
    ]
    if len(regains_5_15) > 0:
        outcomes.add("Good")

    # ── Priority 4: Okay ──────────────────────────────────────────────────
    # Gaining team was fouled (Foul suffered) within 15s → check player count
    for _, act in gaining_15s.iterrows():
        end_event = str(act.get("EndEvent", "")).strip().lower()
        if end_event == "foul suffered":
            # Check player positions at end_frame of this action
            end_frame = int(act.get("end_frame", t0_frame))
            frame_data = get_frame(raw_df, match_id, end_frame)

            if frame_data is not None:
                losing_pos = get_player_positions(frame_data, losing_team_label, include_gk=False)
                gaining_team = "b" if losing_team_label == "a" else "a"
                gaining_pos = get_player_positions(frame_data, gaining_team, include_gk=False)

                # Count players behind ball for both teams
                ball_x = frame_data.get("x_ball")
                if pd.notna(ball_x):
                    ball_x = float(ball_x)
                    # Behind ball = toward losing team's goal
                    if ar:
                        # Attacks right: behind ball = x < ball_x
                        losing_behind = len(losing_pos[losing_pos[:, 0] < ball_x])
                        gaining_behind = len(gaining_pos[gaining_pos[:, 0] < ball_x])
                    else:
                        # Attacks left: behind ball = x > ball_x
                        losing_behind = len(losing_pos[losing_pos[:, 0] > ball_x])
                        gaining_behind = len(gaining_pos[gaining_pos[:, 0] > ball_x])

                    # Check losing team player count
                    if losing_behind <= gaining_behind + 1:
                        outcomes.add("Okay")
                    else:
                        # Losing team has 2+ more players → Bad transition
                        outcomes.add("Bad")

    # ── Return worst outcome (Bad > Okay > Good > Best) ────────────────────
    if "Bad" in outcomes:
        return "Bad"
    elif "Okay" in outcomes:
        return "Okay"
    elif "Good" in outcomes:
        return "Good"
    else:
        # Default: no significant outcomes found
        return "Okay"


# ---------------------------------------------------------------------------
# Metric 10 — Structural Prevention Efficiency (match/aggregate level)
# Uses 10-second window (5s window was insensitive — SPE was always ~100%
# because the ball rarely crosses 35m in just 5 seconds after a transition)
# ---------------------------------------------------------------------------

def structural_prevention_efficiency(
    transitions_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    team_label_map: dict[tuple[str, int], str],
    direction_df: pd.DataFrame,
    fps: float = _DEFAULT_FPS,
) -> float:
    if len(transitions_df) == 0:
        return float("nan")

    prevented, valid = 0, 0

    for _, t_row in transitions_df.iterrows():
        if not t_row.get("has_5s_window", False):
            continue
        valid += 1

        match_id = t_row["match_id"]
        period   = int(t_row["period"])
        losing_label = team_label_map.get((str(match_id), int(t_row["losing_team_id"])), "a")
        dir_row  = direction_df.loc[(str(match_id), period)]
        ar = (losing_label == "a") == bool(dir_row["team_a_attacks_right"])

        penetrated = False
        # Use 10s window (FRAMES_10S) — 5s window was always ~100% for organized teams
        window = get_window_frames(raw_df, match_id, int(t_row["t0_frame"]), int(10 * fps))
        for _, frow in window.iterrows():
            bx = frow.get("x_ball")
            if pd.isna(bx):
                continue
            bx = float(bx)
            if (ar and bx < -THIRD_BOUNDARY_CM) or (not ar and bx > THIRD_BOUNDARY_CM):
                penetrated = True
                break
        if not penetrated:
            prevented += 1

    return prevented / valid if valid > 0 else float("nan")


# ===========================================================================
# POSITIVE TRANSITION METRICS
# ===========================================================================

# ---------------------------------------------------------------------------
# Metric 11 — Constructive Progression Rate
# ---------------------------------------------------------------------------

def constructive_progression(
    transition_row: pd.Series,
    action_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    fps: float = _DEFAULT_FPS,
) -> bool:
    """
    Returns True if the gaining team records a possession phase with PassCount ≥ 3
    within 15 seconds of t0 AND their first 3 passes contain no clearances and no long balls.

    Uses action_df PassCount column (number of consecutive passes within a possession
    phase) and events_df event chain inspection. A PassCount of 3 or more with constructive
    first 3 passes indicates the gaining team built play constructively.
    """
    from src.helpers import check_event_chain

    match_id        = transition_row["match_id"]
    t0_frame        = int(transition_row["t0_frame"])
    t15_frame       = t0_frame + int(15 * fps)
    gaining_team_id = int(transition_row["gaining_team_id"])

    subsequent = action_df[
        (action_df["match_id"] == match_id)
        & (action_df["team_id"] == gaining_team_id)
        & (action_df["start_frame"] > t0_frame)
        & (action_df["start_frame"] <= t15_frame)
    ]

    for _, act in subsequent.iterrows():
        pc = act.get("PassCount")
        if pd.notna(pc) and float(pc) >= 3:
            # Check if first 3 passes were constructive (no clearances, no long passes)
            if events_df is not None:
                action_id = act.get("action_id")
                if pd.notna(action_id):
                    chain_flags = check_event_chain(
                        events_df, match_id, int(action_id), team_id=gaining_team_id
                    )
                    # Constructive only if no clearances AND no long passes
                    if not (chain_flags["has_clearance"] or chain_flags["has_long_pass"]):
                        return True
            else:
                # No events_df; use PassCount ≥ 3 alone
                return True
    return False


# ---------------------------------------------------------------------------
# Metric 12 — Own Half Exit Efficiency
# ---------------------------------------------------------------------------

def own_half_exit(
    transition_row: pd.Series,
    action_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    raw_df: pd.DataFrame | None = None,
    fps: float = _DEFAULT_FPS,
) -> bool:
    """
    Returns True if the gaining team plays at least one BALL-PROGRESSING pass
    from their own half (x_start ≤ 500) that crosses the midfield line (x_end > 500)
    within 15 seconds of t0, using Wyscout 0-1000 coordinates.

    Uses events_df (preferred) for the x_start/x_end check. If events_df or
    raw_df are not available, falls back to action_df (StartX only, less accurate).

    Note: checking StartX alone was misleading — nearly 100% of rest-defence
    transitions start with the gaining team in their own half, so the ball crossing
    midfield (x_end > 500) is the meaningful discriminating condition.
    """
    match_id        = str(transition_row["match_id"])
    t0_frame        = int(transition_row["t0_frame"])
    gaining_team_id = int(transition_row["gaining_team_id"])

    # Preferred path: use events_df with x_start + x_end
    if events_df is not None and raw_df is not None:
        t0_ms = _frame_to_ms(transition_row, raw_df)
        if t0_ms is not None:
            t15_ms = t0_ms + 15_000
            passes_after = events_df[
                (events_df["match_id"].astype(str) == match_id)
                & (events_df["team_id"] == gaining_team_id)
                & (events_df["event_group"] == "Pass")
                & (events_df["timestamp"] > t0_ms)
                & (events_df["timestamp"] <= t15_ms)
                & (events_df["x_start"].notna())
                & (events_df["x_end"].notna())
            ]
            for _, ev in passes_after.iterrows():
                xs = float(ev["x_start"])
                xe = float(ev["x_end"])
                if xs <= 500 and xe > 500:   # pass crosses midfield
                    return True
            return False

    # Fallback: action_df StartX only (less discriminating)
    t15_frame = t0_frame + int(15 * fps)
    gaining_15s = action_df[
        (action_df["match_id"] == transition_row["match_id"])
        & (action_df["team_id"] == gaining_team_id)
        & (action_df["start_frame"] > t0_frame)
        & (action_df["start_frame"] <= t15_frame)
    ]
    for _, act in gaining_15s.iterrows():
        sx = act.get("StartX")
        if pd.notna(sx) and float(sx) <= 500:
            return True
    return False


# ---------------------------------------------------------------------------
# Metric 13 — Playmaker Dependency Index (helper + per-transition)
# ---------------------------------------------------------------------------

_MIN_PASSES = 5   # minimum passes to be considered as playmaker candidate


def build_starting_xi(events_df: pd.DataFrame) -> dict[tuple, set]:
    """
    Returns {(str(match_id), int(team_id)): set_of_starter_player_ids}
    by parsing the Setup event JSON for each match+team.

    The Setup event (event_name='Setup', game_time=0) contains an event_detail
    JSON field with:
      "Team player formation": list of ints (1–11 = starter, 0 = bench)
      "Involved":              player_ids in the same order
    """
    result: dict[tuple, set] = {}
    if "event_name" not in events_df.columns or "event_detail" not in events_df.columns:
        return result

    setup = events_df[
        (events_df["event_name"] == "Setup") & (events_df["game_time"] == 0)
    ]
    for _, row in setup.iterrows():
        try:
            raw = row["event_detail"]
            if pd.isna(raw):
                continue
            detail = json.loads(raw) if isinstance(raw, str) else raw

            # Values are comma-separated strings, e.g. "1, 2, 3, ..., 0, 0"
            formations_raw = detail["Team player formation"]
            involved_raw   = detail["Involved"]
            if isinstance(formations_raw, str):
                positions = [s.strip() for s in formations_raw.split(",")]
            else:
                positions = [str(p) for p in formations_raw]
            if isinstance(involved_raw, str):
                involved = [s.strip() for s in involved_raw.split(",")]
            else:
                involved = [str(p) for p in involved_raw]

            starters = {int(pid) for pid, pos in zip(involved, positions) if int(pos) > 0}
            key = (str(row["match_id"]), int(row["team_id"]))
            result[key] = starters
        except Exception:
            pass
    return result


def identify_playmakers(events_df: pd.DataFrame) -> dict[tuple, int]:
    """
    Deep-Lying Playmaker: starting XI midfielder with highest rate-based composite score.

    score = (made_rate + recv_rate + prog_rate) / 3
      made_rate = player_passes / team_total_passes
      recv_rate = received_passes / team_total_passes
      prog_rate = accurate_progressive_passes / player_passes
        (progressive = x_end - x_start > 200; accurate = outcome == 1)

    Filters:
      - role == 'Midfielder'
      - starting XI only (parsed from Setup event JSON)
      - minimum _MIN_PASSES = 5 passes made

    Returns {(str(match_id), int(team_id)): int(player_id)}
    """
    if "player_id" not in events_df.columns or "event_group" not in events_df.columns:
        return {}

    starting_xi = build_starting_xi(events_df)
    passes = events_df[
        (events_df["event_group"] == "Pass") & (events_df["player_id"].notna())
    ].copy()

    result: dict[tuple, int] = {}
    for (mid, tid), grp in passes.groupby(["match_id", "team_id"]):
        team_total = len(grp)
        if team_total == 0:
            continue

        starters = starting_xi.get((str(mid), int(tid)), set())

        # Filter to starting XI midfielders
        mids = grp[
            (grp["role"] == "Midfielder") &
            (grp["player_id"].apply(lambda p: int(p) in starters))
        ]
        players = [int(p) for p in mids["player_id"].unique()]
        if not players:
            continue

        all_passes = passes[(passes["match_id"] == mid) & (passes["team_id"] == tid)]

        scores: dict[int, float] = {}
        for pid in players:
            p_passes = all_passes[all_passes["player_id"] == pid]
            n = len(p_passes)
            if n < _MIN_PASSES:
                continue
            recv = events_df[
                (events_df["match_id"] == mid) &
                (events_df["team_id"] == tid) &
                (events_df["corr_player"] == pid)
            ].shape[0]
            prog_acc = p_passes[
                (p_passes["x_end"] - p_passes["x_start"] > 200) &
                (p_passes["outcome"] == 1)
            ].shape[0]
            made_rate = n        / team_total
            recv_rate = recv     / team_total
            prog_rate = prog_acc / n
            scores[pid] = round((made_rate + recv_rate + prog_rate) / 3, 4)

        if scores:
            result[(str(mid), int(tid))] = max(scores, key=scores.get)

    return result


def _frame_to_ms(transition_row: pd.Series, raw_df: pd.DataFrame) -> float | None:
    """Convert t0_frame to match milliseconds via the raw tracking 't' column."""
    t0_row = get_frame(raw_df, transition_row["match_id"], int(transition_row["t0_frame"]))
    if t0_row is None:
        return None
    t_val = t0_row.get("t")
    return float(t_val) if pd.notna(t_val) else None


def playmaker_dependency(
    transition_row: pd.Series,
    events_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    playmakers: dict[tuple, int],
) -> tuple[bool | None, bool | None]:
    """
    Returns (dep_1st, dep_2nd):
      dep_1st : True if the 1st pass after t0 (gaining team) targets the playmaker
      dep_2nd : True if the 2nd pass after t0 (gaining team) targets the playmaker
    Either is None if not enough pass events or playmaker not identified.
    """
    match_id        = str(transition_row["match_id"])
    gaining_team_id = int(transition_row["gaining_team_id"])
    playmaker_id    = playmakers.get((match_id, gaining_team_id))
    if playmaker_id is None:
        return None, None

    t0_ms = _frame_to_ms(transition_row, raw_df)
    if t0_ms is None:
        return None, None
    t15_ms = t0_ms + 15_000

    if "event_group" not in events_df.columns or "corr_player" not in events_df.columns:
        return None, None

    passes_after = events_df[
        (events_df["match_id"].astype(str) == match_id)
        & (events_df["team_id"] == gaining_team_id)
        & (events_df["event_group"] == "Pass")
        & (events_df["timestamp"] >= t0_ms)
        & (events_df["timestamp"] <= t15_ms)
    ].sort_values("timestamp")

    if len(passes_after) == 0:
        return None, None

    dep_1st: bool | None = None
    dep_2nd: bool | None = None

    if len(passes_after) >= 1:
        recv = passes_after.iloc[0].get("corr_player")
        dep_1st = bool(pd.notna(recv) and int(recv) == playmaker_id)

    if len(passes_after) >= 2:
        recv = passes_after.iloc[1].get("corr_player")
        dep_2nd = bool(pd.notna(recv) and int(recv) == playmaker_id)

    return dep_1st, dep_2nd


# ---------------------------------------------------------------------------
# Metric 14 — Productive Pass Ratio (two forward-angle versions)
# ---------------------------------------------------------------------------

_FORWARD_45 = math.pi / 4   # 0.785 rad — strict forward (within 45° of attack direction)
_FORWARD_90 = math.pi / 2   # 1.571 rad — forward + sideways (excludes backward passes)


def productive_pass_ratio(
    transition_row: pd.Series,
    events_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    max_angle: float = _FORWARD_90,
) -> float:
    """
    Fraction of gaining team's passes in the 15s window that are FORWARD passes.

    'Forward' = abs(pass_angle) < max_angle.
    Denominator: all passes with a known pass_angle in the 15s window.
    Numerator: those that meet the forward-direction threshold.

    pass_angle convention: 0 = towards opponent goal (forward); ±π = backward.
    max_angle: abs(pass_angle) threshold in radians.
      _FORWARD_45 (π/4) = strict forward (within 45° of attack direction)
      _FORWARD_90 (π/2) = forward + sideways (excludes only backward passes)

    Returns NaN if no pass events with known angle found in the window.
    """
    if "event_group" not in events_df.columns or "pass_angle" not in events_df.columns:
        return float("nan")

    t0_ms = _frame_to_ms(transition_row, raw_df)
    if t0_ms is None:
        return float("nan")
    t15_ms = t0_ms + 15_000

    match_id        = str(transition_row["match_id"])
    gaining_team_id = int(transition_row["gaining_team_id"])

    all_passes = events_df[
        (events_df["match_id"].astype(str) == match_id)
        & (events_df["team_id"] == gaining_team_id)
        & (events_df["event_group"] == "Pass")
        & (events_df["timestamp"] >= t0_ms)
        & (events_df["timestamp"] <= t15_ms)
        & (events_df["pass_angle"].notna())
    ]

    if len(all_passes) == 0:
        return float("nan")

    forward = (all_passes["pass_angle"].abs() < max_angle).sum()
    return float(forward / len(all_passes))


# ===========================================================================
# Per-transition orchestrator
# ===========================================================================

def compute_transition_metrics(
    transition_row: pd.Series,
    raw_df: pd.DataFrame,
    action_df: pd.DataFrame,
    zone_app1: RestDefenceZone,
    losing_team_label: str,
    team_a_attacks_right: bool,
    gaining_action_row: pd.Series | None = None,
    events_df: pd.DataFrame | None = None,
    playmakers: dict | None = None,
    fps: float = _DEFAULT_FPS,
) -> dict:
    """
    Compute all transition metrics (8-14) for a single transition.

    Negative transition metrics (8-10): evaluate defending team quality.
    Positive transition metrics (11-14): evaluate gaining team attack quality.
    """
    result = {}

    # --- Negative transition metrics ---
    dynamics = ball_regain_dynamics(
        transition_row, gaining_action_row, raw_df,
        losing_team_label, team_a_attacks_right,
        fps=fps,
    )
    result.update(dynamics)

    result["transition_rating"] = transition_rating(
        transition_row, action_df, raw_df,
        losing_team_label, team_a_attacks_right,
        fps=fps,
    )

    # --- Positive transition metrics ---

    # Metric 11 — Constructive Progression Rate (refined with event chain inspection)
    result["constructive_progression"] = constructive_progression(
        transition_row, action_df, events_df=events_df, fps=fps
    )

    # Metric 12 — Own Half Exit Efficiency (uses events_df x_start/x_end when available)
    result["own_half_exit"] = own_half_exit(
        transition_row, action_df,
        events_df=events_df,
        raw_df=raw_df,
        fps=fps,
    )

    # Metrics 13 & 14 require events_df
    if events_df is not None:
        # Metric 13 — Playmaker Dependency Index (1st pass and 2nd pass, separate)
        pm = playmakers or {}
        dep_1st, dep_2nd = playmaker_dependency(transition_row, events_df, raw_df, pm)
        result["playmaker_dependency_1st"] = dep_1st
        result["playmaker_dependency_2nd"] = dep_2nd

        # Metric 14 — Productive Pass Ratio (45° and 90° versions)
        result["productive_pass_ratio_45"] = productive_pass_ratio(
            transition_row, events_df, raw_df, max_angle=_FORWARD_45
        )
        result["productive_pass_ratio_90"] = productive_pass_ratio(
            transition_row, events_df, raw_df, max_angle=_FORWARD_90
        )
    else:
        result["playmaker_dependency_1st"]  = None
        result["playmaker_dependency_2nd"]  = None
        result["productive_pass_ratio_45"]  = float("nan")
        result["productive_pass_ratio_90"]  = float("nan")

    return result
