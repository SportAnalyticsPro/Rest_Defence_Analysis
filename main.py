"""
main.py
-------
Compute engine for the Rest Defence analysis pipeline.
Outputs: per-transition PNGs (or videos) + metrics CSVs.
Reports (markdown) are generated separately by report_generator.py.

Usage:
  # CSV only (no PNGs, no reports)
  python main.py --output-dir out/                         → all_transitions.csv for all matches
  python main.py --match-id 7418 7754 --output-dir out/   → per-match CSVs + all_transitions.csv

  # PNGs: only when --match-id or --team-id is given
  python main.py --match-id 7418 --output-dir out/        → match CSV + per-transition PNGs
  python main.py --match-id 7418 --n 3 --output-dir out/  → only first 3 PNGs (all transitions analysed)
  python main.py --match-id 7418 --video --output-dir out/ → MP4 videos instead of PNGs
  python main.py --team-id 95 --output-dir out/           → all matches for team 95: CSVs + PNGs

  # Also generate markdown reports in the same run
  python main.py --match-id 7418 --output-dir out/ --report
  python main.py --output-dir out/ --report

  python main.py --summary                                → direction/SPE summary

Metric categories
-----------------
  Metrics 1–7:  Structural prevention     — negative transition (t0 / t0+1s / t0+5s / t0+10s)
  Metric  8:    Ball Regain Dynamics       — negative transition dynamics
  Metric  9:    Transition Rating          — qualitative outcome (Best/Good/Okay/Bad)
  Metric  10:   Structural Prevention Eff  — negative transition aggregate (10-second window)
  Metrics 11–14: Positive transition       — gaining team attack quality
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

RAW_DATA_PATH    = DATA_DIR / "ih_raw_data.csv"
ACTION_DATA_PATH = DATA_DIR / "ih_action_data.csv"
EVENTS_PATH      = DATA_DIR / "ih_events.csv"
MATCHES_PATH     = DATA_DIR / "matchesList_analisi_transizioni.csv"

sys.path.insert(0, str(Path(__file__).parent))

from src.helpers import format_value, col_mean, col_delta_mean, pct_bool, pct_delta, check_event_chain, jersey_str
from src.data_loading import (
    load_raw_data, load_action_data, load_events, load_matches,
    derive_attack_direction, build_team_label_map, build_team_name_map,
    build_label_map_from_raw, build_name_map_from_team_ids,
    get_team_label, get_frame,
    THIRD_BOUNDARY_CM, get_window_frames,
    detect_fps, build_raw_index,
)
from src.transition_detection import (
    detect_rest_defence_transitions, transitions_for_match, get_gaining_action,
)
from src.metrics.prevention import compute_prevention_metrics
from src.metrics.transition import (
    compute_transition_metrics, structural_prevention_efficiency,
    identify_playmakers,
)

# Teams included in multi-match comparison
COMPARISON_TEAMS = {"Juventus", "Hellas Verona", "Como"}

# FPS is detected at runtime from raw data (see _load_all / _cache["fps"])
# Frame-window constants are computed dynamically: int(N_seconds * fps)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

_wall_start: float = time.time()
_logger = logging.getLogger("transition_control")


def _setup_logging(output_dir: Path) -> None:
    """Set up console + file logging. Call once after output_dir exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(message)s")
    _logger.setLevel(logging.DEBUG)
    # Avoid adding duplicate handlers on repeated calls
    if _logger.handlers:
        return
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    _logger.addHandler(ch)
    log_path = output_dir / "run.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    _logger.addHandler(fh)
    _logger.info(f"Log file: {log_path}")


def _log(msg: str, *, elapsed_since: float | None = None) -> None:
    total = time.time() - _wall_start
    suffix = f"  (+{time.time() - elapsed_since:.1f}s)" if elapsed_since is not None else ""
    _logger.info(f"[{total:6.1f}s] {msg}{suffix}")


# ---------------------------------------------------------------------------
# Data loading (lazy, cached)
# ---------------------------------------------------------------------------

_cache: dict = {}


def _load_all() -> tuple:
    if "raw" not in _cache:
        t = time.time()
        _log("Data loading — raw_data ...")
        _cache["raw"] = load_raw_data(str(RAW_DATA_PATH))
        _log(f"  raw_data loaded  ({len(_cache['raw'])} rows)", elapsed_since=t)
        t_idx = time.time()
        build_raw_index(_cache["raw"])
        _log(f"  Frame index built", elapsed_since=t_idx)

        t2 = time.time()
        _cache["actions"] = load_action_data(str(ACTION_DATA_PATH))
        _log(f"  action_data loaded  ({len(_cache['actions'])} rows)", elapsed_since=t2)

        t3 = time.time()
        _cache["events"] = load_events(str(EVENTS_PATH))
        _log(f"  events loaded  ({len(_cache['events'])} rows)", elapsed_since=t3)

        if MATCHES_PATH.exists():
            _cache["matches"] = load_matches(str(MATCHES_PATH))
            _log(f"  matches loaded  ({len(_cache['matches'])} rows)")
            _cache["lmap"]  = build_team_label_map(_cache["matches"])
            _cache["names"] = build_team_name_map(_cache["matches"])
        else:
            _cache["matches"] = None
            _log("  WARNING: matchesList not found — deriving team labels from raw data team_owner column")
            _cache["lmap"]  = build_label_map_from_raw(_cache["raw"], _cache["actions"])
            _cache["names"] = build_name_map_from_team_ids(_cache["actions"])
            teams_meta_path = DATA_DIR / "teams_metadata.csv"
            if teams_meta_path.exists():
                import pandas as _pd
                meta = _pd.read_csv(teams_meta_path)
                for _, row in meta.iterrows():
                    for mid in _cache["actions"]["match_id"].unique():
                        key = (str(mid), int(row["team_id"]))
                        if key in _cache["names"]:
                            _cache["names"][key] = str(row["team_name"])
                _log(f"  teams_metadata.csv loaded — team names resolved")

        _cache["fps"] = detect_fps(_cache["raw"])
        _log(f"  FPS detected: {_cache['fps']}")

        t4 = time.time()
        _cache["dir"] = derive_attack_direction(_cache["raw"])
        _log(f"  attack direction derived  ({len(_cache['dir'])} rows)", elapsed_since=t4)

        t5 = time.time()
        _cache["playmakers"] = identify_playmakers(_cache["events"])
        _log(
            f"  Playmakers identified ({len(_cache['playmakers'])} team/match pairs)",
            elapsed_since=t5,
        )

        _cache["jersey_map"] = _build_jersey_map(_cache["events"])
        _log(f"  Jersey map built ({len(_cache['jersey_map'])} players)")

        _log("Data loading complete.", elapsed_since=t)

    return (
        _cache["raw"],
        _cache["actions"],
        _cache["events"],
        _cache["matches"],
        _cache["dir"],
        _cache["lmap"],
        _cache["names"],
        _cache["playmakers"],
        _cache["jersey_map"],
    )


# ---------------------------------------------------------------------------
# Team / match helpers
# ---------------------------------------------------------------------------

def _get_match_ids_for_team(team_id: int) -> list[str]:
    """Return all match IDs in which team_id participated, sorted."""
    _, action_df, *_ = _load_all()
    ids = sorted(
        action_df[action_df["team_id"] == team_id]["match_id"].astype(str).unique().tolist()
    )
    if ids:
        _log(f"  team_id={team_id}: {len(ids)} match(es) — {ids}")
    else:
        _log(f"  WARNING: no matches found for team_id={team_id}")
    return ids


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def run_detection() -> pd.DataFrame:
    raw_df, action_df, *_ = _load_all()
    t = time.time()
    _log("Transition detection ...")
    transitions = detect_rest_defence_transitions(action_df, raw_df)
    _log(
        f"  Detected {len(transitions)} rest-defence transitions "
        f"across {transitions['match_id'].nunique()} matches.",
        elapsed_since=t,
    )
    return transitions


# ---------------------------------------------------------------------------
# Jersey number resolution
# ---------------------------------------------------------------------------

def _build_jersey_map(events_df: pd.DataFrame) -> dict[int, int]:
    """
    Build {player_id: jersey_no} from events data.
    events_df has a 'jersey_no' column for each event participant.
    """
    sub = events_df[["player_id", "jersey_no"]].dropna()
    sub = sub.drop_duplicates("player_id")
    result: dict[int, int] = {}
    for _, row in sub.iterrows():
        try:
            result[int(row["player_id"])] = int(row["jersey_no"])
        except (ValueError, TypeError):
            pass
    return result




# ---------------------------------------------------------------------------
# Full metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(transitions: pd.DataFrame) -> pd.DataFrame:
    from src.rest_defence_area import build_zones
    raw_df, action_df, events_df, matches_df, direction_df, lmap, names, playmakers, jersey_map = _load_all()
    fps = _cache.get("fps", 2.0)
    records = []
    n_total = len(transitions)
    t_all = time.time()
    _log(f"Computing metrics for {n_total} transitions ...")

    for seq, (_, t_row) in enumerate(transitions.iterrows(), start=1):
        t_row_start = time.time()
        match_id     = t_row["match_id"]
        period       = int(t_row["period"])
        losing_label = get_team_label(match_id, int(t_row["losing_team_id"]), lmap)
        dir_info     = direction_df.loc[(str(match_id), period)]
        team_a_attacks_right = bool(dir_info["team_a_attacks_right"])

        prev = compute_prevention_metrics(
            t_row, raw_df, direction_df, losing_label,
            fps=fps,
        )

        t0_frame  = int(t_row["t0_frame"])
        t0_row    = get_frame(raw_df, match_id, t0_frame)
        zone_app1 = None
        if t0_row is not None:
            zone_app1, _, _, _, _, _ = build_zones(t0_row, losing_label, team_a_attacks_right)

        gaining_action = get_gaining_action(transitions, action_df, t_row)

        trans = compute_transition_metrics(
            t_row, raw_df, action_df, zone_app1,
            losing_label, team_a_attacks_right,
            gaining_action_row=gaining_action,
            events_df=events_df,
            playmakers=playmakers,
            fps=fps,
        ) if zone_app1 is not None else {}

        # Event chain flags for gaining team's opening action
        chain_flags = {"has_clearance": False, "n_clearances": 0,
                       "has_long_pass": False, "n_long_passes": 0}
        if events_df is not None and pd.notna(t_row.get("gaining_action_id")):
            try:
                chain_flags = check_event_chain(
                    events_df, match_id,
                    int(t_row["gaining_action_id"]),
                    team_id=int(t_row["gaining_team_id"]),
                )
            except Exception as e:
                pass  # Fall back to empty flags on error

        record = {
            "match_id":            match_id,
            "period":              period,
            "losing_team_id":      t_row["losing_team_id"],
            "gaining_team_id":     t_row["gaining_team_id"],
            "t0_frame":            t_row["t0_frame"],
            "losing_action_id":    t_row.get("losing_action_id"),
            "gaining_action_id":   t_row.get("gaining_action_id"),
            "losing_end_event":    t_row["losing_end_event"],
            "gaining_start_event": t_row["gaining_start_event"],
            "losing_pass_count":   t_row["losing_pass_count"],
            "losing_duration_s":   t_row["losing_duration_s"],
            "gaining_chain_has_clearance": chain_flags["has_clearance"],
            "gaining_chain_n_clearances":  chain_flags["n_clearances"],
            "gaining_chain_has_long_pass": chain_flags["has_long_pass"],
            "gaining_chain_n_long_passes": chain_flags["n_long_passes"],
        }
        for offset, mdict in prev.items():
            for k, v in mdict.items():
                record[f"{k}_t{offset}"] = v
        record.update(trans)

        losing_tid = int(t_row["losing_team_id"])
        record["losing_team_name"] = (names or {}).get(
            (str(match_id), losing_tid), str(losing_tid)
        )
        gaining_tid = int(t_row["gaining_team_id"])
        record["gaining_team_name"] = (names or {}).get(
            (str(match_id), gaining_tid), str(gaining_tid)
        )

        # SPE flags — allow report_generator to compute SPE without raw data
        b15, b20, w15, w20 = _check_ball_reaches_third(t_row, raw_df, lmap, direction_df, fps=fps)
        record["ball_reached_third_15s"] = b15
        record["ball_reached_third_20s"] = b20
        record["has_15s_window"]         = w15
        record["has_20s_window"]         = w20
        record["losing_team_attacks_right"] = (
            (get_team_label(match_id, int(t_row["losing_team_id"]), lmap) == "a")
            == bool(direction_df.loc[(str(match_id), period)]["team_a_attacks_right"])
        )

        # Resolve playmaker jersey number so report_generator can display without raw data
        _pm_id = record.get("gaining_team_playmaker_id")
        record["gaining_team_playmaker_jersey"] = (
            int(jersey_map.get(int(_pm_id))) if _pm_id is not None and jersey_map.get(int(_pm_id)) is not None else None
        )

        records.append(record)

        row_elapsed = time.time() - t_row_start
        if seq % 10 == 0 or seq == n_total:
            _log(f"  [{seq}/{n_total}] match {match_id}  frame {t0_frame}  ({row_elapsed:.2f}s/row)")

    _log(f"Metrics computation complete — {n_total} rows.", elapsed_since=t_all)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualise / video a match
# ---------------------------------------------------------------------------

def visualise_match(
    match_id: str,
    n_outputs: int | None = None,
    output_dir: str | None = None,
    video: bool = False,
    report: bool = False,
) -> None:
    from src.rest_defence_area import build_zones
    from src.visualisation import plot_transition_analysis
    from src.video import generate_transition_video
    raw_df, action_df, events_df, matches_df, direction_df, lmap, names, playmakers, jersey_map = _load_all()
    fps = _cache.get("fps", 2.0)

    t = time.time()
    _log(f"Transition detection for match {match_id} ...")
    transitions = detect_rest_defence_transitions(action_df, raw_df)
    match_transitions = transitions_for_match(transitions, match_id)
    _log(f"  {len(match_transitions)} transitions found for match {match_id}.", elapsed_since=t)

    if match_transitions.empty:
        _log(f"No qualifying transitions found for match {match_id}.")
        return

    n_all = len(match_transitions)
    # None (unspecified) → generate all
    n_viz = n_all if n_outputs is None else min(n_outputs, n_all)
    mode  = "videos" if video else "images"

    base_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir  = base_dir / f"match_{match_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add losing_team_name for SPE and grouping
    match_transitions = match_transitions.copy()
    match_transitions["losing_team_name"] = match_transitions.apply(
        lambda r: (names or {}).get(
            (str(match_id), int(r["losing_team_id"])), str(r["losing_team_id"])
        ),
        axis=1,
    )

    # ── PHASE 1: Compute metrics for ALL transitions ──────────────────────
    _log(f"Computing metrics for all {n_all} transitions ...")
    all_results: list = []   # (idx, t_row, prev_metrics, trans_metrics, losing_label)

    for idx, t_row in match_transitions.iterrows():
        period       = int(t_row["period"])
        losing_label = get_team_label(match_id, int(t_row["losing_team_id"]), lmap)
        dir_info     = direction_df.loc[(str(match_id), period)]
        team_a_attacks_right = bool(dir_info["team_a_attacks_right"])
        t0_frame     = int(t_row["t0_frame"])

        prev_metrics = compute_prevention_metrics(
            t_row, raw_df, direction_df, losing_label,
            fps=fps,
        )

        t0_row    = get_frame(raw_df, match_id, t0_frame)
        trans_metrics: dict = {}
        if t0_row is not None:
            zone_app1, _, _, _, _, _ = build_zones(t0_row, losing_label, team_a_attacks_right)
            gaining_action = get_gaining_action(transitions, action_df, t_row)
            trans_metrics = compute_transition_metrics(
                t_row, raw_df, action_df, zone_app1, losing_label,
                team_a_attacks_right, gaining_action_row=gaining_action,
                events_df=events_df, playmakers=playmakers,
                fps=fps,
            )

        all_results.append((idx, t_row, prev_metrics, trans_metrics, losing_label))

        done = len(all_results)
        if done % 10 == 0 or done == n_all:
            _log(f"  [{done}/{n_all}] metrics computed")

    # Build full metrics_df
    metrics_records = []
    for idx, t_row, prev_metrics, trans_metrics, losing_label in all_results:
        t0_frame = int(t_row["t0_frame"])

        # Event chain flags for gaining team's opening action
        chain_flags = {"has_clearance": False, "n_clearances": 0,
                       "has_long_pass": False, "n_long_passes": 0}
        if events_df is not None and pd.notna(t_row.get("gaining_action_id")):
            try:
                chain_flags = check_event_chain(
                    events_df, match_id,
                    int(t_row["gaining_action_id"]),
                    team_id=int(t_row["gaining_team_id"]),
                )
            except Exception:
                pass  # Fall back to empty flags on error

        rec = {
            "t0_frame":         t0_frame,
            "period":           int(t_row["period"]),
            "losing_team_id":   t_row["losing_team_id"],
            "gaining_team_id":  t_row["gaining_team_id"],
            "losing_team_name": t_row["losing_team_name"],
            "gaining_team_name": (names or {}).get(
                (str(match_id), int(t_row["gaining_team_id"])), str(t_row["gaining_team_id"])
            ),
            "losing_action_id":  t_row.get("losing_action_id"),
            "gaining_action_id": t_row.get("gaining_action_id"),
            "gaining_chain_has_clearance": chain_flags["has_clearance"],
            "gaining_chain_n_clearances":  chain_flags["n_clearances"],
            "gaining_chain_has_long_pass": chain_flags["has_long_pass"],
            "gaining_chain_n_long_passes": chain_flags["n_long_passes"],
        }
        for offset, mdict in prev_metrics.items():
            for k, v in mdict.items():
                rec[f"{k}_t{offset}"] = v
        rec.update(trans_metrics)

        # SPE flags
        b15, b20, w15, w20 = _check_ball_reaches_third(t_row, raw_df, lmap, direction_df, fps=fps)
        rec["ball_reached_third_15s"]    = b15
        rec["ball_reached_third_20s"]    = b20
        rec["has_15s_window"]            = w15
        rec["has_20s_window"]            = w20
        rec["losing_team_attacks_right"] = (
            (losing_label == "a")
            == bool(direction_df.loc[(str(match_id), int(t_row["period"]))]["team_a_attacks_right"])
        )
        _pm_id = rec.get("gaining_team_playmaker_id")
        rec["gaining_team_playmaker_jersey"] = (
            int(jersey_map.get(int(_pm_id))) if _pm_id is not None and jersey_map.get(int(_pm_id)) is not None else None
        )

        metrics_records.append(rec)
    metrics_df = pd.DataFrame(metrics_records)

    # ── PHASE 2: Generate viz for first n_viz transitions only ────────────
    _log(
        f"Generating {n_viz} {mode} for match {match_id} "
        f"({n_all} transitions analysed) ..."
    )
    for count, (idx, t_row, prev_metrics, trans_metrics, losing_label) in enumerate(
        all_results[:n_viz], start=1
    ):
        t_trans  = time.time()
        t0_frame = int(t_row["t0_frame"])
        period   = int(t_row["period"])
        _log(f"  [{count}/{n_viz}] Transition t0={t0_frame}  period={period} ...")

        if video:
            out_path = out_dir / f"rd_{match_id}_t{t0_frame}_{idx}.mp4"
            t_vid = time.time()
            _log(f"    rendering video → {out_path.name} ...")
            generate_transition_video(
                transition_row=t_row,
                raw_df=raw_df,
                direction_df=direction_df,
                losing_team_label=losing_label,
                team_name_map=names,
                output_path=str(out_path),
            )
            _log(f"    video saved", elapsed_since=t_vid)
        else:
            out_path = out_dir / f"rd_{match_id}_t{t0_frame}_{idx}.png"
            t_png = time.time()
            _log(f"    rendering image → {out_path.name} ...")
            plot_transition_analysis(
                transition_row=t_row,
                raw_df=raw_df,
                direction_df=direction_df,
                losing_team_label=losing_label,
                metrics_by_offset=prev_metrics,
                transition_metrics=trans_metrics,
                team_name_map=names,
                output_path=str(out_path),
            )
            _log(f"    image saved", elapsed_since=t_png)

        _log(f"  [{count}/{n_viz}] done", elapsed_since=t_trans)

    # ── PHASE 3: Summary uses ALL transitions' metrics ────────────────────
    t_sum = time.time()
    _log("Generating match summary ...")
    _print_match_summary(metrics_df, match_id, playmakers, jersey_map)
    _save_match_summary(metrics_df, match_id, out_dir, playmakers, jersey_map, report=report)
    _log("Match summary done.", elapsed_since=t_sum)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

# Helper functions moved to src/helpers.py
# Use: fmt, col_mean, col_delta_mean, pct_bool (imported above)


# ---------------------------------------------------------------------------
# SPE per team (10-second window)
# ---------------------------------------------------------------------------

def _check_ball_reaches_third(
    t_row: pd.Series,
    raw_df: pd.DataFrame,
    lmap: dict,
    direction_df: pd.DataFrame,
    fps: float = 2.0,
) -> tuple[bool, bool, bool, bool]:
    """
    Returns (ball_reached_15s, ball_reached_20s, has_15s_window, has_20s_window).

    has_*_window: enough frames exist after t0 within the same half.
    ball_reached_*: ball entered the defensive third within that window.
    Both ball_reached values are False when the corresponding window does not exist.
    """
    frames_15s = int(15 * fps)
    frames_20s = int(20 * fps)

    match_id  = t_row["match_id"]
    period    = int(t_row["period"])
    t0_frame  = int(t_row["t0_frame"])

    losing_label = get_team_label(match_id, int(t_row["losing_team_id"]), lmap)
    dir_row      = direction_df.loc[(str(match_id), period)]
    ar = (losing_label == "a") == bool(dir_row["team_a_attacks_right"])

    window = get_window_frames(raw_df, match_id, t0_frame, frames_20s)
    n_frames = len(window)

    has_15s_window = n_frames >= frames_15s
    has_20s_window = n_frames >= frames_20s

    ball_reached_15s = False
    ball_reached_20s = False

    for frame_idx, (_, frow) in enumerate(window.iterrows()):
        bx = frow.get("x_ball")
        if pd.isna(bx):
            continue
        bx = float(bx)
        if (ar and bx < -THIRD_BOUNDARY_CM) or (not ar and bx > THIRD_BOUNDARY_CM):
            ball_reached_20s = True
            if frame_idx < frames_15s:
                ball_reached_15s = True
            break

    return ball_reached_15s, ball_reached_20s, has_15s_window, has_20s_window


def _spe_for_team(
    team_name: str,
    transitions_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    lmap: dict,
    direction_df: pd.DataFrame,
    fps: float = 2.0,
) -> tuple[float, float]:
    """
    Compute SPE at both 15s and 20s windows for all transitions defended by team_name.
    Returns (spe_15s, spe_20s).
    """
    team_trans = transitions_df[transitions_df["losing_team_name"] == team_name]
    if team_trans.empty:
        return float("nan"), float("nan")

    prevented_15, prevented_20, valid = 0, 0, 0

    for _, t_row in team_trans.iterrows():
        if not t_row.get("has_5s_window", False):
            continue
        valid += 1
        match_id     = t_row["match_id"]
        period       = int(t_row["period"])
        losing_label = get_team_label(match_id, int(t_row["losing_team_id"]), lmap)
        dir_row      = direction_df.loc[(str(match_id), period)]
        ar = (losing_label == "a") == bool(dir_row["team_a_attacks_right"])

        penetrated_15 = False
        penetrated_20 = False
        frames_15s = int(15 * fps)
        frames_20s = int(20 * fps)
        window = get_window_frames(raw_df, match_id, int(t_row["t0_frame"]), frames_20s)
        for frame_idx, (_, frow) in enumerate(window.iterrows()):
            bx = frow.get("x_ball")
            if pd.isna(bx):
                continue
            bx = float(bx)
            if (ar and bx < -THIRD_BOUNDARY_CM) or (not ar and bx > THIRD_BOUNDARY_CM):
                penetrated_20 = True
                if frame_idx < frames_15s:
                    penetrated_15 = True
                break
        if not penetrated_15:
            prevented_15 += 1
        if not penetrated_20:
            prevented_20 += 1

    spe_15 = prevented_15 / valid if valid > 0 else float("nan")
    spe_20 = prevented_20 / valid if valid > 0 else float("nan")
    return spe_15, spe_20


# ---------------------------------------------------------------------------
# Match summary — console output (grouped by defending team)
# ---------------------------------------------------------------------------

def _print_match_summary(
    metrics_df: pd.DataFrame,
    match_id: str,
    playmakers: dict | None = None,
    jersey_map: dict | None = None,
) -> None:
    if metrics_df.empty:
        return

    print(f"\n{'='*70}")
    print(f"  Match {match_id} — Summary")
    print(f"{'='*70}")

    offset_labels = ["t0", "t0+1s", "t0+5s", "t0+10s"]
    offset_keys   = [0, 2, 10, 20]

    for team_name in sorted(metrics_df["losing_team_name"].unique()):
        tdf = metrics_df[metrics_df["losing_team_name"] == team_name]
        n   = len(tdf)

        # SPE from per-transition CSV flags (no raw data needed)
        valid_15 = tdf[tdf["has_15s_window"].astype(bool)] if "has_15s_window" in tdf.columns else tdf
        valid_20 = tdf[tdf["has_20s_window"].astype(bool)] if "has_20s_window" in tdf.columns else tdf
        spe_15 = (1 - valid_15["ball_reached_third_15s"].astype(float).mean()) * 100 if "ball_reached_third_15s" in tdf.columns else float("nan")
        spe_20 = (1 - valid_20["ball_reached_third_20s"].astype(float).mean()) * 100 if "ball_reached_third_20s" in tdf.columns else float("nan")
        spe_15_str = format(spe_15, ".1f") + "%" if not np.isnan(spe_15) else "—"
        spe_20_str = format(spe_20, ".1f") + "%" if not np.isnan(spe_20) else "—"

        rts = tdf["transition_rating"] if "transition_rating" in tdf.columns else pd.Series(dtype=str)
        pct = lambda r: f"{(rts == r).sum() / n * 100:.0f}%" if n > 0 else "—"

        # Resolve gaining team playmaker
        gaining_tid = None
        if "gaining_team_id" in tdf.columns:
            mode = tdf["gaining_team_id"].mode()
            if len(mode) > 0:
                gaining_tid = int(mode.iloc[0])
        pm_id = (playmakers or {}).get((str(match_id), gaining_tid)) if gaining_tid else None
        pm_str = jersey_str(pm_id, jersey_map or {})

        print(f"\n## {team_name} (defending) — {n} transitions")

        print(f"\n  Overview")
        print(f"    SPE (15s): {spe_15_str}  (% transitions ball did NOT enter defensive third in 15s)")
        print(f"    SPE (20s): {spe_20_str}  (% transitions ball did NOT enter defensive third in 20s)")
        print(f"    Ratings:   Best {pct('Best')}  Good {pct('Good')}  "
              f"Okay {pct('Okay')}  Bad {pct('Bad')}")

        # Structural metrics (multi-offset table)
        print(f"\n  Structural Metrics (mean across {n} transitions)")
        print(f"  {'Metric':<32} " + "  ".join(f"{lbl:>8}" for lbl in offset_labels))
        print(f"  {'-'*72}")

        def _srow(label, col_base, scale=1.0, fmt=".2f"):
            vals = [col_mean(tdf, f"{col_base}_t{k}") * scale for k in offset_keys]
            cells = "  ".join(f"{format_value(v, fmt):>8}" for v in vals)
            print(f"  {label:<32} {cells}")

        _srow("Team Length (m)",       "team_length_m")
        _srow("Line Height (m)",       "line_height_m")
        _srow("Players Behind Ball",   "players_behind_ball",  fmt=".1f")
        _srow("NumSup RD App1 (Rule-Based)",  "num_superiority_app1", fmt=".1f")
        _srow("NumSup RD App2 (Clustering)", "num_superiority_app2", fmt=".1f")
        _srow("Team Compactness (m)",  "team_compactness",     fmt=".2f")

        # Pressing (t0+1s, t0+5s, t0+10s only — t0 is always 0 by definition)
        print(f"\n  Pressing & Escape Pressure (0=no intensity, 100=max; Δ%>0 = higher intensity) "
              f"(t0 omitted: team was in possession)")
        print(f"  {'Metric':<32} {'t0+1s':>8}  {'t0+5s':>8}  {'t0+10s':>8}  {'Δ(1→5s)':>8}  {'Δ(1→10s)':>8}")
        print(f"  {'-'*78}")

        def _prow(label, col_base):
            v1  = col_mean(tdf, f"{col_base}_t2")
            v5  = col_mean(tdf, f"{col_base}_t10")
            v10 = col_mean(tdf, f"{col_base}_t20")
            v15 = col_mean(tdf, f"{col_base}_t30")
            d5  = col_delta_mean(tdf, f"{col_base}_t10", f"{col_base}_t2")
            d10 = col_delta_mean(tdf, f"{col_base}_t20", f"{col_base}_t2")
            d15 = col_delta_mean(tdf, f"{col_base}_t30", f"{col_base}_t2")
            print(f"  {label:<32} {format_value(v1):>8}  {format_value(v5):>8}  {format_value(v10):>8}  {format_value(v15):>8}  {format_value(d5):>8}  {format_value(d10):>8}  {format_value(d15):>8}")

        _prow("Zone Press App1", "zone_press_app1")
        _prow("Zone Press App2", "zone_press_app2")
        _prow("Team Press", "team_press")
        _prow("Zone Esc.Press (App1)", "gaining_ps_zone")
        _prow("Team Esc.Press", "gaining_ps_mean")

        # Transition dynamics
        print(f"\n  Transition Dynamics (negative transition — defending team)")
        print(f"    Centroid Advance 5s (m):   {format_value(col_mean(tdf, 'centroid_advance_5s_m'))}")
        print(f"    Centroid Advance 10s (m):  {format_value(col_mean(tdf, 'centroid_advance_10s_m'))}")

        # Positive transition
        cp     = pct_bool(tdf, "constructive_progression")
        ohe    = pct_bool(tdf, "own_half_exit")
        ppr45  = col_mean(tdf, "productive_pass_ratio_45") * 100
        ppr90  = col_mean(tdf, "productive_pass_ratio_90") * 100
        pmd1   = pct_bool(tdf, "playmaker_dependency_1st")
        pmd2   = pct_bool(tdf, "playmaker_dependency_2nd")
        print(f"\n  Positive Transition (gaining team)")
        print(f"    Opponent playmaker (Deep-Lying, composite score): {pm_str}")
        print(f"    Constructive Progression (PassCount≥3):           {format_value(cp, '.1f')}%")
        print(f"    Own Half Exit (possession in own half):           {format_value(ohe, '.1f')}%")
        print(f"    Forward Pass Ratio (45°, within 45° of goal):     {format_value(ppr45, '.1f')}%")
        print(f"    Forward Pass Ratio (90°, within 90° of goal):     {format_value(ppr90, '.1f')}%")
        print(f"    Playmaker Dep. (1st pass→PM):                     {format_value(pmd1, '.1f')}%")
        print(f"    Playmaker Dep. (2nd pass→PM):                     {format_value(pmd2, '.1f')}%")

    print()


# ---------------------------------------------------------------------------
# Match summary — file output (.md + .csv)
# ---------------------------------------------------------------------------
# MD generation lives in report_generator.py — this function saves CSV only
# and delegates report writing to generate_match_report().

def _save_match_summary(
    metrics_df: pd.DataFrame,
    match_id: str,
    out_dir: Path,
    playmakers: dict | None = None,
    jersey_map: dict | None = None,
    report: bool = False,
) -> None:
    if metrics_df.empty:
        return
    csv_path = out_dir / f"match_{match_id}_summary.csv"
    float_cols = metrics_df.select_dtypes(include="float").columns
    metrics_df[float_cols] = metrics_df[float_cols].round(4)
    metrics_df.to_csv(str(csv_path), index=False)
    _log(f"  Metrics CSV saved: {csv_path}")
    if report:
        from report_generator import generate_match_report
        generate_match_report(metrics_df, match_id, out_dir, playmakers=playmakers, jersey_map=jersey_map)


# ---------------------------------------------------------------------------
# Multi-match comparison
# ---------------------------------------------------------------------------

def multi_match_comparison(
    output_dir: str | None = None,
    match_ids: list[str] | None = None,
    report: bool = False,
) -> None:
    """
    match_ids: if given, restrict analysis to those match IDs and use all
               teams found in them. If None, use all matches and COMPARISON_TEAMS.
    report: if True, generate team_comparison.md via report_generator.
    """
    raw_df, action_df, events_df, matches_df, direction_df, lmap, names, playmakers, jersey_map = _load_all()
    fps = _cache.get("fps", 2.0)

    t_detect = time.time()
    _log("Transition detection (all matches) ...")
    transitions = detect_rest_defence_transitions(action_df, raw_df)
    _log(
        f"  {len(transitions)} transitions detected across "
        f"{transitions['match_id'].nunique()} matches.",
        elapsed_since=t_detect,
    )

    # Filter to requested match IDs when provided
    if match_ids:
        match_ids_str = {str(m) for m in match_ids}
        transitions = transitions[transitions["match_id"].astype(str).isin(match_ids_str)].copy()
        _log(f"  Filtered to {len(transitions)} transitions across {len(match_ids_str)} requested match(es).")

    # Add losing_team_name to transitions for grouping / SPE
    transitions = transitions.copy()
    transitions["losing_team_name"] = transitions.apply(
        lambda r: (names or {}).get(
            (str(r["match_id"]), int(r["losing_team_id"])), str(r["losing_team_id"])
        ),
        axis=1,
    )

    t_metrics = time.time()
    _log("Computing metrics for all transitions ...")
    metrics_df = compute_all_metrics(transitions)
    _log(f"Metrics done — {len(metrics_df)} rows.", elapsed_since=t_metrics)

    # Determine which teams to include in the comparison
    if match_ids:
        comparison_teams = (
            set(metrics_df["losing_team_name"].unique()) |
            set(metrics_df["gaining_team_name"].unique())
        )
    else:
        comparison_teams = COMPARISON_TEAMS

    comp_df = metrics_df[
        metrics_df["losing_team_name"].isin(comparison_teams) |
        metrics_df["gaining_team_name"].isin(comparison_teams)
    ].copy()

    if comp_df.empty:
        _log("No transitions found for comparison teams.")
        return

    def _build_foul_row_console(tdf: pd.DataFrame) -> dict:
        if "foul_committed" not in tdf.columns:
            return {k: "—" for k in ("n_fouls", "foul_rate", "foul_time_s_avg",
                                      "foul_x_m_avg", "bad_pct", "okay_pct")}
        foul_df = tdf[tdf["foul_committed"].astype(bool)]
        n, n_f  = len(tdf), len(foul_df)
        bad_f   = (foul_df["foul_superiority_rating"] == "Bad").sum()
        okay_f  = (foul_df["foul_superiority_rating"] == "Okay").sum()
        return {
            "n_fouls":        n_f,
            "foul_rate":      f"{n_f/n*100:.1f}%" if n else "—",
            "foul_time_s_avg":format_value(foul_df["foul_time_s"].mean(), ".1f") if n_f else "—",
            "foul_x_m_avg":   format_value(foul_df["foul_x_m"].mean(), ".1f") if n_f else "—",
            "bad_pct":        f"{bad_f} ({bad_f/n_f*100:.0f}%)" if n_f else "—",
            "okay_pct":       f"{okay_f} ({okay_f/n_f*100:.0f}%)" if n_f else "—",
        }

    # Build per-team rows
    team_rows = []
    for team_name in sorted(comparison_teams):
        tdf = comp_df[comp_df["losing_team_name"] == team_name]
        gdf = comp_df[comp_df["gaining_team_name"] == team_name]
        if tdf.empty:
            continue

        n   = len(tdf)
        n_m = tdf["match_id"].nunique()
        rts = tdf["transition_rating"] if "transition_rating" in tdf.columns else pd.Series(dtype=str)
        pct = lambda r: f"{(rts == r).sum() / n * 100:.0f}%" if n > 0 else "—"

        t_spe = time.time()
        _log(f"  Computing SPE for {team_name} ({n} transitions, 15s+20s window) ...")
        spe_15, spe_20 = _spe_for_team(team_name, transitions, raw_df, lmap, direction_df, fps=fps)
        _log(
            f"    SPE(15s)={spe_15:.1%}  SPE(20s)={spe_20:.1%}" if not np.isnan(spe_15) else "    SPE = —",
            elapsed_since=t_spe,
        )

        ppr45_mean = col_mean(tdf, "productive_pass_ratio_45")
        ppr90_mean = col_mean(tdf, "productive_pass_ratio_90")

        team_rows.append({
            "team":     team_name,
            "n":        n,
            "n_m":      n_m,
            "spe_15":   spe_15,
            "spe_20":   spe_20,
            # Section 1
            "pct_best": pct("Best"),
            "pct_good": pct("Good"),
            "pct_okay": pct("Okay"),
            "pct_bad":  pct("Bad"),
            # Section 2
            "team_len":    format_value(col_mean(tdf, "team_length_m_t0")),
            "line_ht":     format_value(col_mean(tdf, "line_height_m_t0")),
            "behind_ball": format_value(col_mean(tdf, "players_behind_ball_t0"), ".1f"),
            "numsup1":     format_value(col_mean(tdf, "num_superiority_app1_t0"), ".1f"),
            "numsup2":     format_value(col_mean(tdf, "num_superiority_app2_t0"), ".1f"),
            "compact":     format_value(col_mean(tdf, "team_compactness_t0"), ".2f"),
            # Section 3
            "numsup1_5s":  format_value(col_mean(tdf, "num_superiority_app1_t10"), ".1f"),
            "numsup1_10s": format_value(col_mean(tdf, "num_superiority_app1_t20"), ".1f"),
            "numsup2_5s":  format_value(col_mean(tdf, "num_superiority_app2_t10"), ".1f"),
            "numsup2_10s": format_value(col_mean(tdf, "num_superiority_app2_t20"), ".1f"),
            "compact_d5s": format_value(col_mean(tdf, "team_compactness_t10") - col_mean(tdf, "team_compactness_t0"), ".2f"),
            # Press metrics — from tdf (losing team transitions)
            "zp1_1s":  format_value(col_mean(tdf, "zone_press_app1_t2")),
            "zp1_d5":  format_value(pct_delta(tdf, "zone_press_app1", 2, 10, col_mean)),
            "zp1_d10": format_value(pct_delta(tdf, "zone_press_app1", 2, 20, col_mean)),
            "zp2_d5":  format_value(pct_delta(tdf, "zone_press_app2", 2, 10, col_mean)),
            "tp_1s":   format_value(col_mean(tdf, "team_press_t2")),
            "tp_d5":   format_value(pct_delta(tdf, "team_press", 2, 10, col_mean)),
            "tp_d10":  format_value(pct_delta(tdf, "team_press", 2, 20, col_mean)),
            # Escape metrics — from gdf (gaining team transitions)
            "escz_d5":  format_value(pct_delta(gdf, "gaining_ps_zone", 2, 10, col_mean)),
            "escz_d10": format_value(pct_delta(gdf, "gaining_ps_zone", 2, 20, col_mean)),
            "esct_d5":  format_value(pct_delta(gdf, "gaining_ps_mean", 2, 10, col_mean)),
            "esct_d10": format_value(pct_delta(gdf, "gaining_ps_mean", 2, 20, col_mean)),
            # Section 4a — Defense recovery when losing ball (tdf)
            "cadv5":       format_value(col_mean(tdf, "centroid_advance_5s_m")),
            "cadv10":      format_value(col_mean(tdf, "centroid_advance_10s_m")),
            "team_len_d5s": format_value(col_mean(tdf, "team_length_m_t10") - col_mean(tdf, "team_length_m_t0"), ".2f"),
            "team_len_d10s": format_value(col_mean(tdf, "team_length_m_t20") - col_mean(tdf, "team_length_m_t0"), ".2f"),
            "compact_d5s": format_value(col_mean(tdf, "team_compactness_t10") - col_mean(tdf, "team_compactness_t0"), ".2f"),
            "compact_d10s": format_value(col_mean(tdf, "team_compactness_t20") - col_mean(tdf, "team_compactness_t0"), ".2f"),
            # Section 4b — Team's own attacking performance when gaining ball (gdf)
            "cp_pct_own":      f"{format_value(pct_bool(gdf, 'constructive_progression'), '.1f')}%" if not gdf.empty else "—",
            "ohe_pct_own":     f"{format_value(pct_bool(gdf, 'own_half_exit'), '.1f')}%" if not gdf.empty else "—",
            "ppr45_pct_own":   f"{format_value(col_mean(gdf, 'productive_pass_ratio_45') * 100 if not gdf.empty else float('nan'), '.1f')}%" if not gdf.empty else "—",
            "ppr90_pct_own":   f"{format_value(col_mean(gdf, 'productive_pass_ratio_90') * 100 if not gdf.empty else float('nan'), '.1f')}%" if not gdf.empty else "—",
            "pmd1_pct_own":    f"{format_value(pct_bool(gdf, 'playmaker_dependency_1st'), '.1f')}%" if not gdf.empty else "—",
            "pmd2_pct_own":    f"{format_value(pct_bool(gdf, 'playmaker_dependency_2nd'), '.1f')}%" if not gdf.empty else "—",
            # Section 5 — Foul analysis
            **_build_foul_row_console(tdf),
        })

    # Sort by SPE (15s) descending
    team_rows.sort(
        key=lambda r: r["spe_15"] if not (isinstance(r["spe_15"], float) and np.isnan(r["spe_15"])) else -1,
        reverse=True,
    )

    def _build_sec(rows, field_map: dict) -> pd.DataFrame:
        return pd.DataFrame([
            {"Team": r["team"], **{hdr: r[key] for hdr, key in field_map.items()}}
            for r in rows
        ])

    sec1 = _build_sec(team_rows, {
        "N":          "n",
        "Matches":    "n_m",
        "SPE (15s)":  "spe_15",
        "SPE (20s)":  "spe_20",
        "% Best":     "pct_best",
        "% Good":     "pct_good",
        "% Okay":     "pct_okay",
        "% Bad":      "pct_bad",
    })
    sec1["SPE (15s)"] = sec1["SPE (15s)"].apply(
        lambda v: format(v, ".1%") if isinstance(v, float) and not np.isnan(v) else "—"
    )
    sec1["SPE (20s)"] = sec1["SPE (20s)"].apply(
        lambda v: format(v, ".1%") if isinstance(v, float) and not np.isnan(v) else "—"
    )

    sec2 = _build_sec(team_rows, {
        "TeamLen(m)":                  "team_len",
        "LineHt(m)":                   "line_ht",
        "BehindBall":                  "behind_ball",
        "NumSup RD App1 (Rule-Based)": "numsup1",
        "NumSup RD App2 (Clustering)": "numsup2",
        "TeamComp(m)":                 "compact",
    })

    sec3 = _build_sec(team_rows, {
        "NumSup App1 (5s)":   "numsup1_5s",
        "NumSup App1 (10s)":  "numsup1_10s",
        "NumSup App2 (5s)":   "numsup2_5s",
        "NumSup App2 (10s)":  "numsup2_10s",
        "CompΔ(5s)":          "compact_d5s",
        "ZPress1(t1s)":       "zp1_1s",
        "ZPress1Δ%(5s)":      "zp1_d5",
        "ZPress1Δ%(10s)":     "zp1_d10",
        "ZPress2Δ%(5s)":      "zp2_d5",
        "TmPress(t1s)":       "tp_1s",
        "TmPressΔ%(5s)":      "tp_d5",
        "TmPressΔ%(10s)":     "tp_d10",
        "EscZ-Δ%(5s)":        "escz_d5",
        "EscZ-Δ%(10s)":       "escz_d10",
        "EscT-Δ%(5s)":        "esct_d5",
        "EscT-Δ%(10s)":       "esct_d10",
    })

    sec4a = _build_sec(team_rows, {
        "CAdv5s(m)":      "cadv5",
        "CAdv10s(m)":     "cadv10",
        "TeamLen Δ(5s)":  "team_len_d5s",
        "TeamLen Δ(10s)": "team_len_d10s",
        "Compact Δ(5s)":  "compact_d5s",
        "Compact Δ(10s)": "compact_d10s",
    })

    sec4b = _build_sec(team_rows, {
        "ConstrProg%":      "cp_pct_own",
        "OwnHalfExit%":     "ohe_pct_own",
        "ProdPass(45°)%":   "ppr45_pct_own",
        "ProdPass(90°)%":   "ppr90_pct_own",
        "PM Dep(1st)%":     "pmd1_pct_own",
        "PM Dep(2nd)%":     "pmd2_pct_own",
    })
    sec5 = _build_sec(team_rows, {
        "N Fouls":          "n_fouls",
        "Foul Rate":        "foul_rate",
        "Avg Time (s)":     "foul_time_s_avg",
        "Avg Loc (m)":      "foul_x_m_avg",
        "Bad (sup.)":       "bad_pct",
        "Okay (eq./inf.)":  "okay_pct",
    })

    # Console output
    try:
        from tabulate import tabulate as _tab
        def _print_table(df: pd.DataFrame) -> None:
            print(_tab(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    except ImportError:
        def _print_table(df: pd.DataFrame) -> None:
            print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("  Rest Defence — Multi-Match Comparison")
    print("=" * 80)
    print("\n### Section 1 — Ratings & SPE (15s & 20s windows)")
    _print_table(sec1)
    print("\n### Section 2 — Structural Metrics at t0")
    _print_table(sec2)
    print("\n### Section 3 — Pressing & Escape Pressure  [Press scale: 0=no press, 100=max; Δ%>0 = pressed harder; Escape: Δ%>0 = more pressure]")
    _print_table(sec3)
    print("\n### Section 4a — Transition Dynamics (Positive Transition)")
    print("  Metrics when team loses possession and must defend. CAdv = centroid advance (positive = recovering shape).")
    print("  TeamLen/Compact Δ = change in structural metrics at 5s and 10s.")
    _print_table(sec4a)
    print("\n### Section 4b — Attack Quality (Negative Transition)")
    print("  Metrics show team's own attacking performance when they gain possession.")
    print("  ConstrProg/OwnHalfExit/ProdPass/PM Dep = quality of possessions when team has the ball.")
    _print_table(sec4b)
    print("\n### Section 5 — Foul Analysis (Defending Team, within 15 s)")
    print("  Bad = fouled in numerical superiority. Okay = fouled in equality/inferiority.")
    print("  Avg Loc (m) = distance from defending team's own goal (0=own goal, 105=opponent goal).")
    _print_table(sec5)
    print()

    # Save all_transitions.csv to output root
    base_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    t_save = time.time()
    _log("Saving comparison outputs ...")

    all_csv_path = base_dir / "all_transitions.csv"
    float_cols = comp_df.select_dtypes(include="float").columns
    comp_df[float_cols] = comp_df[float_cols].round(4)
    comp_df.to_csv(str(all_csv_path), index=False)
    _log(f"  All transitions CSV saved: {all_csv_path}", elapsed_since=t_save)

    if report:
        from report_generator import generate_comparison_report
        generate_comparison_report(comp_df, base_dir, teams=comparison_teams)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary() -> None:
    raw_df, action_df, events_df, matches_df, direction_df, lmap, names, playmakers, jersey_map = _load_all()

    print("\n=== Direction check ===")
    print(direction_df[["mean_gk_x_a", "team_a_attacks_right"]].to_string())

    transitions = run_detection()

    print("\n=== Transitions per match ===")
    per_match = transitions.groupby("match_id").size().reset_index(name="count")
    print(per_match.to_string(index=False))

    print("\n=== Structural Prevention Efficiency ===")
    spe = structural_prevention_efficiency(transitions, raw_df, lmap, direction_df)
    print(f"  Overall: {spe:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rest Defence Analysis pipeline")
    parser.add_argument("--match-id",   type=str,  nargs="+", default=None,
                        help="One or more match IDs to analyse (space-separated)")
    parser.add_argument("--team-id",    type=int,  default=None,
                        help="Analyse all matches for this team ID")
    parser.add_argument("--n",          type=int,  default=None,
                        help="Max number of viz outputs to generate per match "
                             "(default: ALL transitions; all transitions are always analysed)")
    parser.add_argument("--video",      action="store_true",
                        help="Generate MP4 videos instead of PNG images (requires --match-id or --team-id)")
    parser.add_argument("--output-dir", type=str,  default=None,
                        help="Base output directory (default: output/)")
    parser.add_argument("--report",     action="store_true",
                        help="Also generate markdown report(s) via report_generator "
                             "(default: CSV + PNGs only)")
    parser.add_argument("--summary",    action="store_true",
                        help="Print direction/SPE summary")
    parser.add_argument("--export-csv", type=str,  default=None,
                        help="Export full metrics CSV to this path")
    args = parser.parse_args()

    # Determine output root and set up logging before any work begins
    output_root = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    _setup_logging(output_root)

    try:
        if args.summary:
            print_summary()
            return

        if args.export_csv:
            transitions = run_detection()
            metrics_df  = compute_all_metrics(transitions)
            float_cols = metrics_df.select_dtypes(include="float").columns
            metrics_df[float_cols] = metrics_df[float_cols].round(4)
            metrics_df.to_csv(args.export_csv, index=False)
            _log(f"Metrics exported to {args.export_csv}")
            return

        # Warn if visualisation flags are given without a match specifier
        if args.match_id is None and args.team_id is None:
            if args.video:
                _log("WARNING: --video has no effect without --match-id or --team-id")
            if args.n is not None:
                _log("WARNING: --n has no effect without --match-id or --team-id")

        # Resolve target match IDs from --match-id or --team-id
        target_ids: list[str] | None = None
        if args.team_id is not None:
            target_ids = _get_match_ids_for_team(args.team_id)
            if not target_ids:
                return
        elif args.match_id:
            target_ids = [str(m) for m in args.match_id]

        if target_ids:
            for mid in target_ids:
                visualise_match(
                    mid,
                    n_outputs=args.n,
                    output_dir=args.output_dir,
                    video=args.video,
                    report=args.report,
                )
            if len(target_ids) > 1:
                multi_match_comparison(
                    output_dir=args.output_dir,
                    match_ids=target_ids,
                    report=args.report,
                )
            return

        # Default mode: all matches, COMPARISON_TEAMS
        multi_match_comparison(output_dir=args.output_dir, report=args.report)

    except Exception:
        _logger.exception("Fatal error — full traceback follows")
        raise


if __name__ == "__main__":
    main()
