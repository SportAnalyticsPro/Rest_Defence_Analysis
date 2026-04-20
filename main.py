"""
main.py
-------
Entry point for the Rest Defence analysis pipeline.

Usage:
  python main.py                              → multi-match comparison (Juventus, Hellas Verona, Como)
  python main.py --match-id 7418             → single match, ALL transitions analysed + viz
  python main.py --match-id 7418 --video     → single match, MP4 videos instead of images
  python main.py --match-id 7418 --n 3       → generate only 3 viz; ALL transitions still analysed
  python main.py --match-id 7418 --output-dir my_dir/
  python main.py --export-csv out.csv        → export full metrics CSV
  python main.py --summary                   → direction/SPE summary

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

from src.helpers import format_value, col_mean, col_delta_mean, pct_bool, pct_delta, check_event_chain
from src.data_loading import (
    load_raw_data, load_action_data, load_events, load_matches,
    derive_attack_direction, build_team_label_map, build_team_name_map,
    build_label_map_from_raw, build_name_map_from_team_ids,
    get_team_label, get_frame,
    THIRD_BOUNDARY_CM, get_window_frames,
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

# SPE windows at 2fps
# (5s window was insensitive — ball rarely travels 35m+ after a rest-defence transition)
FRAMES_15S = 30
FRAMES_20S = 40


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


def _jersey_str(player_id: int | None, jersey_map: dict) -> str:
    """Return '#N (ID player_id)' or just 'ID player_id' if jersey unknown."""
    if player_id is None:
        return "—"
    jersey = jersey_map.get(player_id)
    if jersey is not None:
        return f"#{jersey} (player_id={player_id})"
    return f"player_id={player_id} (jersey unknown)"


# ---------------------------------------------------------------------------
# Full metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(transitions: pd.DataFrame) -> pd.DataFrame:
    from src.rest_defence_area import build_zones
    raw_df, action_df, events_df, matches_df, direction_df, lmap, names, playmakers, jersey_map = _load_all()
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
            time_offsets=(0, 2, 10, 20),
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
) -> None:
    from src.rest_defence_area import build_zones
    from src.visualisation import plot_transition_analysis
    from src.video import generate_transition_video
    raw_df, action_df, events_df, matches_df, direction_df, lmap, names, playmakers, jersey_map = _load_all()

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
            time_offsets=(0, 2, 10, 20),
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
    _print_match_summary(
        metrics_df, match_id, match_transitions,
        raw_df, lmap, direction_df, playmakers, jersey_map,
    )
    _save_match_summary(
        metrics_df, match_id, out_dir, match_transitions,
        raw_df, lmap, direction_df, playmakers, jersey_map,
    )
    _log("Match summary done.", elapsed_since=t_sum)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

# Helper functions moved to src/helpers.py
# Use: fmt, col_mean, col_delta_mean, pct_bool (imported above)


# ---------------------------------------------------------------------------
# SPE per team (10-second window)
# ---------------------------------------------------------------------------

def _spe_for_team(
    team_name: str,
    transitions_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    lmap: dict,
    direction_df: pd.DataFrame,
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
        window = get_window_frames(raw_df, match_id, int(t_row["t0_frame"]), FRAMES_20S)
        for frame_idx, (_, frow) in enumerate(window.iterrows()):
            bx = frow.get("x_ball")
            if pd.isna(bx):
                continue
            bx = float(bx)
            if (ar and bx < -THIRD_BOUNDARY_CM) or (not ar and bx > THIRD_BOUNDARY_CM):
                penetrated_20 = True
                if frame_idx < FRAMES_15S:
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
    match_transitions: pd.DataFrame,
    raw_df: pd.DataFrame,
    lmap: dict,
    direction_df: pd.DataFrame,
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

        spe_15, spe_20 = _spe_for_team(team_name, match_transitions, raw_df, lmap, direction_df)
        spe_15_str = format(spe_15, ".1%") if not np.isnan(spe_15) else "—"
        spe_20_str = format(spe_20, ".1%") if not np.isnan(spe_20) else "—"

        rts = tdf["transition_rating"] if "transition_rating" in tdf.columns else pd.Series(dtype=str)
        pct = lambda r: f"{(rts == r).sum() / n * 100:.0f}%" if n > 0 else "—"

        # Resolve gaining team playmaker
        gaining_tid = None
        if "gaining_team_id" in tdf.columns:
            mode = tdf["gaining_team_id"].mode()
            if len(mode) > 0:
                gaining_tid = int(mode.iloc[0])
        pm_id = (playmakers or {}).get((str(match_id), gaining_tid)) if gaining_tid else None
        pm_str = _jersey_str(pm_id, jersey_map or {})

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

def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


_SINGLE_MATCH_GLOSSARY = """
---

## Metric Glossary

### Time Offsets
| Label | Meaning |
|---|---|
| **t0** | Exact frame when possession is lost (transition moment) |
| **t0+1s** | 1 second after t0. Press values are meaningful here; at t0 the team was in possession so p_ = 0 by definition |
| **t0+5s** | 5 seconds after t0 |
| **t0+10s** | 10 seconds after t0 |

### Negative Transition Metrics (defending team quality)

| Metric | Abbreviation / Column | Interpretation |
|---|---|---|
| SPE (15s) | Structural Prevention Efficiency | % of transitions where ball does NOT enter the defensive third within 15 s. **Higher = better defensive organisation.** |
| SPE (20s) | Structural Prevention Efficiency (20s) | Same metric with 20-second window. |
| Team Length | `team_length_m` | Distance (m) from rearmost to foremost outfield player along the pitch axis. **Lower = more compact block.** |
| Line Height | `line_height_m` | Distance (m) from the rearmost defensive line centroid to the team's own goal line. **Higher = more aggressive pressing line; lower = deeper, more cautious.** |
| Players Behind Ball | `players_behind_ball` | Count of outfield defenders with position behind the ball at t0. **More = better immediate cover.** |
| NumSup RD App1 (Rule-Based) / App2 (Clustering) | `num_superiority_app1/2` | Defenders minus attackers in the App1 / App2 rest-defence zones. App1 uses rule-based zone construction; App2 uses k-means clustering. **Positive = numerical advantage; negative = gap in zone.** |
| Team Compactness (m) | `team_compactness` | Mean Euclidean distance (m) of outfield defenders from their team centre. **Lower = more compact shape; higher = more spread.** |
| Zone Press | `zone_press_app1/2` | Mean pressing intensity (`p_` column, 0–100 scale) of defending players inside the App1/App2 zone. **0 = no pressing, 100 = maximum pressing.** Reported as percentage change (Δ%) from t0+1s baseline. Negative Δ = pressing harder over time (good). |
| Team Press | `team_press` | Mean pressing intensity of ALL defending outfield players. Same inverted 0–100 scale. **Negative Δ = team increases pressing over time.** |
| Centroid Advance | `centroid_advance_5s_m` / `_10s_m` | Forward movement (m) of the defending team's centroid toward the opponent goal. **Positive = recovering/advancing shape; negative = retreating.** |

### Transition Rating
| Rating | Meaning |
|---|---|
| **Best** | Defending team regains possession within 5 seconds |
| **Bad** | Shot conceded within 15 s, OR ball enters defensive third within 5 s, OR opponent plays an in-behind pass (StartX > 750) within 15 s, OR gaining team was fouled with ≥2 player numerical advantage for defending team |
| **Good** | Ball goes out of play within 15 s, OR opponent commits a foul (defending team wins free kick), OR defending team regains 5–15 s without dangerous outcome |
| **Okay** | Gaining team was fouled with ≤1 player advantage for defending team, OR attack persists beyond 15 s without matching Bad/Good conditions |


### Positive Transition Metrics (gaining team attack quality)

| Metric | Column | Interpretation |
|---|---|---|
| Constructive Progression | `constructive_progression` | % of transitions where the gaining team records a possession phase with ≥ 3 passes (PassCount ≥ 3) within 15 s. **Higher = opponent builds play more often.** |
| Own Half Exit | `own_half_exit` | % of transitions where the gaining team has a possession phase starting from their own half (StartX ≤ 500 in Wyscout 0–1000 coords) within 15 s. **Higher = opponent involves deeper players.** |
| Forward Pass Ratio (45°) | `productive_pass_ratio_45` | % of gaining team's passes in 15 s that are strict forward passes (within 45° of attack direction). **Higher = opponent attacks more directly.** |
| Forward Pass Ratio (90°) | `productive_pass_ratio_90` | % of gaining team's passes in 15 s that are forward or sideways (within 90°, excludes only backward passes). **Higher = opponent tends to advance, not play backward.** |
| Playmaker Dep. (1st pass) | `playmaker_dependency_1st` | % of transitions where the 1st post-transition pass targets the gaining team's auto-identified Deep-Lying Playmaker. **Higher = opponent's first instinct is to find the playmaker.** |
| Playmaker Dep. (2nd pass) | `playmaker_dependency_2nd` | % of transitions where the 2nd pass targets the playmaker. Together with the 1st-pass metric, shows how quickly the team routes through their key player. |
"""

_MULTI_MATCH_GLOSSARY = """
---

## Metric Glossary

### Column Abbreviations

| Column | Full Name | Interpretation |
|---|---|---|
| **N** | Transitions | Number of rest-defence transitions analysed for this team |
| **SPE (15s)** | Structural Prevention Efficiency | % of transitions where ball does NOT reach defensive third in 15 s. Higher = better. |
| **SPE (20s)** | Structural Prevention Efficiency (20s) | Same metric with 20-second window. |
| **% Best/Good/Okay/Bad** | Transition Rating | Distribution of qualitative transition outcomes. Best = regain ≤5s; Good = ball dead or opponent fouls; Okay = opponent fouled with small advantage or prolonged attack; Bad = shot/penetration/dangerous play conceded. |
| **TeamLen(m)** | Team Length | Distance (m) from rearmost to foremost outfield player at t0. Lower = more compact. |
| **LineHt(m)** | Line Height | Distance (m) from rearmost defensive line to own goal at t0. Higher = higher, more aggressive line. |
| **BehindBall** | Players Behind Ball | Count of outfield players positioned behind the ball at t0. More = better cover. |
| **NumSup RD App1 (Rule-Based) / App2 (Clustering)** | Numerical Superiority Rest Defence | Defenders minus attackers in App1/App2 zone at t0. App1 = rule-based; App2 = clustering. Positive = advantage; negative = gap. |
| **TeamComp(m)** | Team Compactness | Mean Euclidean distance (m) of outfield defenders from team centre at t0. Lower = more compact shape. |
| **NumSup App1 (5s)** | Numerical Superiority App1 at t0+5s | Defenders minus attackers in App1 (rule-based) zone 5 s after transition. Positive = defending team advantage. |
| **NumSup App1 (10s)** | Numerical Superiority App1 at t0+10s | Same metric 10 s after transition. |
| **NumSup App2 (5s)** | Numerical Superiority App2 at t0+5s | Defenders minus attackers in App2 (clustering) zone 5 s after transition. Positive = defending team advantage. |
| **NumSup App2 (10s)** | Numerical Superiority App2 at t0+10s | Same metric 10 s after transition. |
| **CompΔ(5s)** | Compactness Delta at 5 s | Change in Team Compactness (m) from t0 to t0+5s. Negative = team tightened up (good). |
| **ZPress1(t1s)** | Zone Press App1 at t0+1s | Mean pressing intensity (`p_` column, 0–100 scale) of defenders inside App1 zone, 1 s after transition. 0 = no pressing, 100 = maximum pressing. |
| **ZPress1Δ%(5s)** | Zone Press App1 Δ% 1→5s | Percentage change in zone pressing intensity from t0+1s to t0+5s. **Positive = team pressed harder.** |
| **ZPress1Δ%(10s)** | Zone Press App1 Δ% 1→10s | Percentage change from t0+1s to t0+10s. Positive = increased pressing. |
| **ZPress2Δ%(5s)** | Zone Press App2 Δ% 1→5s | Same for App2 (clustering) zone approach. |
| **TmPress(t1s)** | Team Press at t0+1s | Mean pressing intensity of all outfield defenders, 1 s after transition. Same 0–100 scale. |
| **TmPressΔ%(5s)** | Team Press Δ% 1→5s | Percentage change in overall team pressing from t0+1s to t0+5s. Positive = pressed harder. |
| **TmPressΔ%(10s)** | Team Press Δ% 1→10s | Percentage change from t0+1s to t0+10s. |
| **EscZ-Δ%(5s)** | Escape Pressure Zone Δ% 1→5s | Percentage change in pressure received (`ps_` column, 0–100 scale) by gaining-team players inside zone. Positive = more pressure (not escaping). Computed from transitions where team was gaining. |
| **EscZ-Δ%(10s)** | Escape Pressure Zone Δ% 1→10s | Same from t0+1s to t0+10s. |
| **EscT-Δ%(5s)** | Escape Pressure Team Δ% 1→5s | Percentage change in pressure received across all gaining-team outfield players. |
| **EscT-Δ%(10s)** | Escape Pressure Team Δ% 1→10s | Same from t0+1s to t0+10s. |
| **CAdv5s(m)** | Centroid Advance 5 s | Forward movement (m) of the defending team centroid in 5 s. Positive = recovering shape. |
| **CAdv10s(m)** | Centroid Advance 10 s | Forward movement (m) of the defending team centroid in 10 s. |
| **ConstrProg%** | Constructive Progression % | % of transitions where **team made ≥ 3 passes within 15 s when attacking**. Higher = more structured build-up. Computed from team's gaining transitions. |
| **OwnHalfExit%** | Own Half Exit % | % of transitions where **team controlled ball in their own half within 15 s when attacking**. Higher = ball retained in safe area. |
| **ProdPass(45°)%** | Forward Pass Ratio (45°) | % of **team's passes in 15 s that go strictly forward** (within 45° of attack direction) when attacking. |
| **ProdPass(90°)%** | Forward Pass Ratio (90°) | % of **team's passes in 15 s that go forward or sideways** (within 90°) when attacking. |
| **PM Dep (1st)%** | Playmaker Dep. 1st pass | % of transitions where **team's 1st pass targets the Deep-Lying Playmaker** when attacking. |
| **PM Dep (2nd)%** | Playmaker Dep. 2nd pass | % of transitions where **team's 2nd pass targets the Deep-Lying Playmaker** when attacking. |

### Pressure Scale Notes
- **Pressing (`p_` columns)**: 0 = no pressing, 100 = maximum pressing intensity. Reported as **percentage change (Δ%) from t0+1s baseline**. Positive Δ% = team pressed harder.
- **Escape Pressure (`ps_` columns)**: 0 = no pressure received, 100 = maximum pressure received. Reported as **percentage change (Δ%) from t0+1s baseline**. Positive Δ% = more pressure received (unsuccessful escape); negative Δ% = pressure relieved (successful escape).

### Transition Types

| Type | Definition | Metrics |
|---|---|---|
| **Positive Transition** | When team **loses possession** and transitions to defending. Measures how the team responds structurally and tactically. | CAdv5s/10s, TeamLen Δ, Compact Δ |
| **Negative Transition** | When team **gains possession** and transitions to attacking. Measures how effectively the team builds play and progresses the ball. | ConstrProg%, OwnHalfExit%, ProdPass, PM Dep |
"""


def _save_match_summary(
    metrics_df: pd.DataFrame,
    match_id: str,
    out_dir: Path,
    match_transitions: pd.DataFrame,
    raw_df: pd.DataFrame,
    lmap: dict,
    direction_df: pd.DataFrame,
    playmakers: dict | None = None,
    jersey_map: dict | None = None,
) -> None:
    if metrics_df.empty:
        return

    offset_keys   = [0, 2, 10, 20]
    offset_labels = ["t0", "t0+1s", "t0+5s", "t0+10s"]

    md_lines: list[str] = [f"# Match {match_id} — Rest Defence Summary\n"]

    for team_name in sorted(metrics_df["losing_team_name"].unique()):
        tdf = metrics_df[metrics_df["losing_team_name"] == team_name]
        gdf = metrics_df[metrics_df["gaining_team_name"] == team_name]
        n   = len(tdf)

        spe_15, spe_20 = _spe_for_team(team_name, match_transitions, raw_df, lmap, direction_df)
        spe_15_str = format(spe_15, ".1%") if not np.isnan(spe_15) else "—"
        spe_20_str = format(spe_20, ".1%") if not np.isnan(spe_20) else "—"

        rts = tdf["transition_rating"] if "transition_rating" in tdf.columns else pd.Series(dtype=str)
        pct = lambda r: f"{(rts == r).sum() / n * 100:.0f}%" if n > 0 else "—"

        # Resolve playmaker for the team's own gaining transitions
        team_tid = None
        if len(gdf) > 0 and "gaining_team_id" in gdf.columns:
            mode = gdf["gaining_team_id"].mode()
            if len(mode) > 0:
                team_tid = int(mode.iloc[0])
        pm_id  = (playmakers or {}).get((str(match_id), team_tid)) if team_tid else None
        pm_str = _jersey_str(pm_id, jersey_map or {})

        md_lines += [
            f"## {team_name} (defending) — {n} transitions\n",
            "### Overview\n",
            _md_table(
                ["Metric", "Value", "Notes"],
                [
                    ["SPE (15s)",  spe_15_str,
                     "% transitions where ball didn't reach defensive third in 15 s — higher is better"],
                    ["SPE (20s)",  spe_20_str,
                     "% transitions where ball didn't reach defensive third in 20 s — higher is better"],
                    ["% Best",  pct("Best"),  "Possession regained within 5 s"],
                    ["% Good",  pct("Good"),  "Ball out of play or foul won within 15 s"],
                    ["% Okay",  pct("Okay"),  "Foul committed or attack delayed past 15 s"],
                    ["% Bad",   pct("Bad"),   "Shot, penetration, or in-behind pass conceded within 15 s"],
                ],
            ),
            "",
            "### Structural Metrics (mean)\n",
            "> Lower Team Length and Team Compactness = more compact/organised block. "
            "Higher Line Height = more aggressive pressing line position.\n",
            _md_table(
                ["Metric"] + offset_labels,
                [
                    ["Team Length (m)"]
                    + [format_value(col_mean(tdf, f"team_length_m_t{k}")) for k in offset_keys],
                    ["Line Height (m)"]
                    + [format_value(col_mean(tdf, f"line_height_m_t{k}")) for k in offset_keys],
                    ["Players Behind Ball"]
                    + [format_value(col_mean(tdf, f"players_behind_ball_t{k}"), ".1f") for k in offset_keys],
                    ["NumSup RD App1 (Rule-Based)"]
                    + [format_value(col_mean(tdf, f"num_superiority_app1_t{k}"), ".1f") for k in offset_keys],
                    ["NumSup RD App2 (Clustering)"]
                    + [format_value(col_mean(tdf, f"num_superiority_app2_t{k}"), ".1f") for k in offset_keys],
                    ["Team Compactness (m)"]
                    + [format_value(col_mean(tdf, f"team_compactness_t{k}"), ".2f") for k in offset_keys],
                ],
            ),
            "",
            "### Pressing & Escape Pressure (mean)\n",
            "> **Press scale:** 0 = no pressing, 100 = maximum pressing. Δ values show change from t0+1s. Negative Δ = team pressed harder. "
            "> **Escape scale:** 0 = no pressure, 100 = maximum pressure received. Negative Δ = team escaped pressure successfully. "
            "t0 is omitted because the losing team was in possession (press = 0 by definition). "
            "Negative Δ = team is pressing harder later in the transition.\n",
            _md_table(
                ["Metric", "t0+1s", "t0+5s", "t0+10s", "Δ (1s→5s)", "Δ (1s→10s)"],
                [
                    ["Zone Press App1",
                     format_value(col_mean(tdf, "zone_press_app1_t2")),
                     format_value(col_mean(tdf, "zone_press_app1_t10")),
                     format_value(col_mean(tdf, "zone_press_app1_t20")),
                     format_value(col_delta_mean(tdf, "zone_press_app1_t10", "zone_press_app1_t2")),
                     format_value(col_delta_mean(tdf, "zone_press_app1_t20", "zone_press_app1_t2"))],
                    ["Zone Press App2",
                     format_value(col_mean(tdf, "zone_press_app2_t2")),
                     format_value(col_mean(tdf, "zone_press_app2_t10")),
                     format_value(col_mean(tdf, "zone_press_app2_t20")),
                     format_value(col_delta_mean(tdf, "zone_press_app2_t10", "zone_press_app2_t2")),
                     format_value(col_delta_mean(tdf, "zone_press_app2_t20", "zone_press_app2_t2"))],
                    ["Team Press (all players)",
                     format_value(col_mean(tdf, "team_press_t2")),
                     format_value(col_mean(tdf, "team_press_t10")),
                     format_value(col_mean(tdf, "team_press_t20")),
                     format_value(col_delta_mean(tdf, "team_press_t10", "team_press_t2")),
                     format_value(col_delta_mean(tdf, "team_press_t20", "team_press_t2"))],
                    ["Zone Esc.Press (App1)",
                     format_value(col_mean(gdf, "gaining_ps_zone_t2")),
                     format_value(col_mean(gdf, "gaining_ps_zone_t10")),
                     format_value(col_mean(gdf, "gaining_ps_zone_t20")),
                     format_value(col_delta_mean(gdf, "gaining_ps_zone_t10", "gaining_ps_zone_t2")),
                     format_value(col_delta_mean(gdf, "gaining_ps_zone_t20", "gaining_ps_zone_t2"))],
                    ["Team Esc.Press (all players)",
                     format_value(col_mean(gdf, "gaining_ps_mean_t2")),
                     format_value(col_mean(gdf, "gaining_ps_mean_t10")),
                     format_value(col_mean(gdf, "gaining_ps_mean_t20")),
                     format_value(col_delta_mean(gdf, "gaining_ps_mean_t10", "gaining_ps_mean_t2")),
                     format_value(col_delta_mean(gdf, "gaining_ps_mean_t20", "gaining_ps_mean_t2"))],
                ],
            ),
            "",
            "### Transition Dynamics (Positive Transition)\n",
            "> **Positive Transition:** Metrics when team loses possession and must defend. CAdv = centroid advance (positive = recovering shape). TeamLen/Compact Δ = change in structural metrics at 5s and 10s.\n",
            _md_table(
                ["Metric", "Value", "Notes"],
                [
                    ["Centroid Advance 5s (m)",
                     format_value(col_mean(tdf, "centroid_advance_5s_m")),
                     "Positive = team centroid moved forward (recovering shape)"],
                    ["Centroid Advance 10s (m)",
                     format_value(col_mean(tdf, "centroid_advance_10s_m")),
                     "Positive = continued advance after 10 s"],
                    ["Team Length Δ 5s (m)",
                     format_value(col_mean(tdf, "team_length_m_t10") - col_mean(tdf, "team_length_m_t0"), ".2f"),
                     "Change in team length from t0 to 5s. Negative = more compact."],
                    ["Team Length Δ 10s (m)",
                     format_value(col_mean(tdf, "team_length_m_t20") - col_mean(tdf, "team_length_m_t0"), ".2f"),
                     "Change in team length from t0 to 10s."],
                    ["Compactness Δ 5s (m)",
                     format_value(col_mean(tdf, "team_compactness_t10") - col_mean(tdf, "team_compactness_t0"), ".2f"),
                     "Change in team compactness from t0 to 5s. Negative = tighter shape."],
                    ["Compactness Δ 10s (m)",
                     format_value(col_mean(tdf, "team_compactness_t20") - col_mean(tdf, "team_compactness_t0"), ".2f"),
                     "Change in team compactness from t0 to 10s. Negative = tighter shape."],
                ],
            ),
            "",
            "### Transition Dynamics (Negative Transition)\n",
            "> **Negative Transition:** Metrics show team's own attacking performance when they gain possession. ConstrProg/OwnHalfExit/ProdPass/PM Dep = quality of possessions when team has the ball.\n",
            f"> Gaining Team Playmaker (Deep-Lying, composite score): **{pm_str}**\n",
            "",
            _md_table(
                ["Metric", "Value", "Notes"],
                [
                    ["Constructive Progression (PassCount ≥ 3)",
                     f"{format_value(pct_bool(tdf, 'constructive_progression'), '.1f')}%",
                     "% transitions where gaining team made ≥3 passes in 15 s"],
                    ["Own Half Exit (StartX ≤ 500)",
                     f"{format_value(pct_bool(tdf, 'own_half_exit'), '.1f')}%",
                     "% transitions where gaining team had possession in own half within 15 s"],
                    ["Forward Pass Ratio (45°)",
                     f"{format_value(col_mean(tdf, 'productive_pass_ratio_45') * 100, '.1f')}%",
                     "% of gaining team passes in 15 s that are strictly forward (within 45° of attack dir)"],
                    ["Forward Pass Ratio (90°)",
                     f"{format_value(col_mean(tdf, 'productive_pass_ratio_90') * 100, '.1f')}%",
                     "% of gaining team passes in 15 s that are forward or sideways (within 90°)"],
                    ["Playmaker Dep. (1st pass → PM)",
                     f"{format_value(pct_bool(tdf, 'playmaker_dependency_1st'), '.1f')}%",
                     f"1st pass targets playmaker ({pm_str})"],
                    ["Playmaker Dep. (2nd pass → PM)",
                     f"{format_value(pct_bool(tdf, 'playmaker_dependency_2nd'), '.1f')}%",
                     f"2nd pass targets playmaker ({pm_str})"],
                ],
            ),
            "",
        ]

    md_lines.append(_SINGLE_MATCH_GLOSSARY)

    md_path = out_dir / f"match_{match_id}_summary.md"
    md_path.write_text("\n".join(md_lines))
    _log(f"  Summary saved: {md_path}")

    csv_path = out_dir / f"match_{match_id}_summary.csv"
    float_cols = metrics_df.select_dtypes(include="float").columns
    metrics_df[float_cols] = metrics_df[float_cols].round(4)
    metrics_df.to_csv(str(csv_path), index=False)
    _log(f"  Metrics CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Multi-match comparison
# ---------------------------------------------------------------------------

def multi_match_comparison(output_dir: str | None = None) -> None:
    raw_df, action_df, events_df, matches_df, direction_df, lmap, names, playmakers, jersey_map = _load_all()

    t_detect = time.time()
    _log("Transition detection (all matches) ...")
    transitions = detect_rest_defence_transitions(action_df, raw_df)
    _log(
        f"  {len(transitions)} transitions detected across "
        f"{transitions['match_id'].nunique()} matches.",
        elapsed_since=t_detect,
    )

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

    comp_df = metrics_df[
        metrics_df["losing_team_name"].isin(COMPARISON_TEAMS) |
        metrics_df["gaining_team_name"].isin(COMPARISON_TEAMS)
    ].copy()

    if comp_df.empty:
        _log("No transitions found for comparison teams.")
        return

    # Build per-team rows
    team_rows = []
    for team_name in sorted(COMPARISON_TEAMS):
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
        spe_15, spe_20 = _spe_for_team(team_name, transitions, raw_df, lmap, direction_df)
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
    print()

    # Save outputs
    base_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    t_save = time.time()
    _log("Saving comparison outputs ...")

    csv_path = base_dir / "team_comparison.csv"
    float_cols = comp_df.select_dtypes(include="float").columns
    comp_df[float_cols] = comp_df[float_cols].round(4)
    comp_df.to_csv(str(csv_path), index=False)
    _log(f"  Comparison CSV saved: {csv_path}", elapsed_since=t_save)

    md_path = base_dir / "team_comparison.md"
    _write_comparison_md(md_path, sec1, sec2, sec3, sec4a, sec4b)
    _log(f"  Comparison markdown saved: {md_path}")


def _write_comparison_md(
    path: Path,
    sec1: pd.DataFrame,
    sec2: pd.DataFrame,
    sec3: pd.DataFrame,
    sec4a: pd.DataFrame,
    sec4b: pd.DataFrame,
) -> None:
    def _df_to_md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
        data_rows = [
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, sep] + data_rows)

    lines = [
        "# Rest Defence — Multi-Match Comparison\n",
        "_Sorted by SPE (Structural Prevention Efficiency, 15-second window), descending._\n",
        "",
        "## Section 1 — Ratings & SPE (15s & 20s windows)\n",
        "_Higher SPE = better at keeping the ball out of the defensive third after losing possession._\n",
        _df_to_md(sec1),
        "",
        "## Section 2 — Structural Metrics at t0\n",
        "_Lower Team Length and Team Compactness = more compact block. "
        "Higher Line Height = more aggressive position. "
        "Positive Num.Sup = defenders outnumber attackers in zone._\n",
        _df_to_md(sec2),
        "",
        "## Section 3 — Pressing & Escape Pressure\n",
        "_**Press scale:** 0 = no pressing, 100 = maximum pressing. Δ% reported as percentage change from t0+1s baseline. Positive Δ% = team pressed harder. "
        "**Escape scale:** 0 = no pressure, 100 = maximum pressure. Positive Δ% = more pressure received (not escaping); negative Δ% = pressure relieved._\n",
        _df_to_md(sec3),
        "",
        "## Section 4a — Transition Dynamics (Positive Transition)\n",
        "_**Positive Transition:** Metrics when team loses possession and must defend. CAdv = centroid advance (positive = recovering shape). "
        "TeamLen/Compact Δ = change in structural metrics at 5s and 10s after possession loss._\n",
        _df_to_md(sec4a),
        "",
        "## Section 4b — Transition Dynamics (Negative Transition)\n",
        "_**Negative Transition:** Metrics show team's own attacking performance when they gain possession. "
        "ConstrProg/OwnHalfExit/ProdPass/PM Dep = quality of possessions when team has the ball._\n",
        _df_to_md(sec4b),
        "",
        _MULTI_MATCH_GLOSSARY,
    ]
    path.write_text("\n".join(lines))


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
    parser.add_argument("--match-id",   type=str,  default=None,
                        help="Single match ID to analyse")
    parser.add_argument("--n",          type=int,  default=None,
                        help="Max number of viz outputs to generate "
                             "(default: ALL transitions; all transitions are always analysed)")
    parser.add_argument("--video",      action="store_true",
                        help="Generate MP4 videos instead of PNG images (requires --match-id)")
    parser.add_argument("--output-dir", type=str,  default=None,
                        help="Base output directory (default: output/)")
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

        if args.match_id:
            visualise_match(
                args.match_id,
                n_outputs=args.n,
                output_dir=args.output_dir,
                video=args.video,
            )
            return

        # Default mode: multi-match comparison
        multi_match_comparison(output_dir=args.output_dir)

    except Exception:
        _logger.exception("Fatal error — full traceback follows")
        raise


if __name__ == "__main__":
    main()
