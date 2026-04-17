"""
visualisation.py
----------------
2×2 grid of pitch panels (t0, t0+1s, t0+5s, t0+10s) plus a full metrics table.

Each pitch panel:
  - Uniform team colours: HOME=blue (#1f77b4), AWAY=red (#d62728)
  - Ball: white circle with black edge
  - Both teams' players with y-sorted cluster connecting lines
  - Jersey numbers displayed inside each player dot
  - Two zone overlays (App3 is currently disabled):
      App1 (rule-based)  : semi-transparent fill, losing-team colour
      App2 (pure k-means): yellow dotted outline, no fill
  - Ball-line at t0 only
  - Players behind ball highlighted at t0 only

Zones are recomputed per panel (passed in pre-built dict).
Figure background is white.
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mplsoccer import Pitch

from .data_loading import (
    PITCH_HALF_LENGTH_CM,
    PITCH_HALF_WIDTH_CM,
    get_frame,
    get_player_positions,
    get_gk_position,
)
from .rest_defence_area import RestDefenceZone, build_zones
from .logos import add_team_logos

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
HOME_COLOUR = "#1f77b4"   # blue  — team 'a' (home)
AWAY_COLOUR = "#d62728"   # red   — team 'b' (away)
BALL_COLOUR = "white"


def _team_colour(team: str) -> str:
    return HOME_COLOUR if team == "a" else AWAY_COLOUR


# ---------------------------------------------------------------------------
# Coordinate helpers (cm → mplsoccer metres, origin at top-left of pitch)
# ---------------------------------------------------------------------------

def _cm_to_m(x: float) -> float:
    return x / 100.0


def _x_to_pitch(x_cm: float) -> float:
    return _cm_to_m(x_cm + PITCH_HALF_LENGTH_CM)


def _y_to_pitch(y_cm: float) -> float:
    return _cm_to_m(y_cm + PITCH_HALF_WIDTH_CM)


def _pos_to_pitch(positions: np.ndarray) -> np.ndarray:
    """(N,2) absolute cm → (N,2) mplsoccer metres."""
    if len(positions) == 0:
        return positions
    out = np.empty_like(positions, dtype=float)
    out[:, 0] = _x_to_pitch(positions[:, 0])
    out[:, 1] = _y_to_pitch(positions[:, 1])
    return out


# ---------------------------------------------------------------------------
# Single pitch panel
# ---------------------------------------------------------------------------

def _draw_pitch_panel(
    ax: plt.Axes,
    frame_row: pd.Series,
    zone_app1: RestDefenceZone,
    zone_app2: RestDefenceZone,
    zone_app3: RestDefenceZone,   # accepted but not drawn (App3 disabled)
    losing_labels: np.ndarray,
    gaining_labels: np.ndarray,
    losing_team: str,
    gaining_team: str,
    team_a_attacks_right: bool,
    title: str,
    draw_ball_line: bool = False,
    highlight_behind_ball: bool = False,
) -> None:
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=_cm_to_m(PITCH_HALF_LENGTH_CM * 2),
        pitch_width=_cm_to_m(PITCH_HALF_WIDTH_CM * 2),
        pitch_color="#4a7c4a",
        stripe=True,
        stripe_color="#3d6b3d",
        line_color="white",
        linewidth=1.2,
    )
    pitch.draw(ax=ax)

    losing_colour  = _team_colour(losing_team)
    gaining_colour = _team_colour(gaining_team)

    # Attack direction for losing team
    ar_losing = (losing_team == "a") == team_a_attacks_right

    # ----------------------------------------------------------------
    # Zone overlays — two approaches drawn back to front
    # App1 (rule-based):   filled + solid border, losing-team colour
    # App2 (pure k-means): yellow dotted border, no fill
    # App3 is DISABLED — zone_app3 parameter accepted but not drawn
    # ----------------------------------------------------------------
    ZONE_COLOURS = {
        "app1": (losing_colour, losing_colour, 0.18, "-",  2.5),   # fill + border
        "app2": ("#ffee00",     "none",        0.95, ":",  2.5),   # yellow dotted
        # "app3": ("#ff9900", "none", 0.95, "--", 2.5),             # orange dashed — disabled
    }

    def _draw_zone(zone: RestDefenceZone, edge_color: str, face_color: str,
                   alpha: float, linestyle: str, linewidth: float = 2.5) -> None:
        rx = _x_to_pitch(zone.x_min)
        rw = _cm_to_m(zone.x_max - zone.x_min)
        ry = _y_to_pitch(zone.y_min)
        rh = _cm_to_m(zone.y_max - zone.y_min)
        patch = mpatches.FancyBboxPatch(
            (rx, ry), rw, rh,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor=face_color,
            linestyle=linestyle,
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)

    _draw_zone(zone_app1, *ZONE_COLOURS["app1"])
    _draw_zone(zone_app2, *ZONE_COLOURS["app2"])
    # _draw_zone(zone_app3, *ZONE_COLOURS["app3"])   # App3 disabled

    # Zone legend (top-left corner)
    legend_handles = [
        mpatches.Patch(facecolor=losing_colour, edgecolor=losing_colour,
                       alpha=0.5, label="App1 (rule-based)"),
        mpatches.Patch(facecolor="none", edgecolor="#ffee00",
                       linestyle=":", linewidth=2, label="App2 (pure k-means)"),
        # App3 legend entry disabled:
        # mpatches.Patch(facecolor="none", edgecolor="#ff9900",
        #                linestyle="--", linewidth=2, label="App3 (adaptive k-means)")
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=6,
        framealpha=0.85,
        facecolor="white",
        labelcolor="#111111",
        edgecolor="#cccccc",
        handlelength=1.8,
        borderpad=0.5,
    )

    # ----------------------------------------------------------------
    # Ball
    # ----------------------------------------------------------------
    bx = frame_row.get("x_ball")
    by = frame_row.get("y_ball")
    ball_x_cm = float("nan")
    if pd.notna(bx) and pd.notna(by):
        ball_x_cm = float(bx)
        ax.plot(
            _x_to_pitch(float(bx)), _y_to_pitch(float(by)),
            marker="o", color=BALL_COLOUR, markersize=6,
            markeredgecolor="black", markeredgewidth=1.0, zorder=6,
        )
        if draw_ball_line:
            ax.axvline(
                x=_x_to_pitch(float(bx)),
                color="white", linestyle="--", linewidth=1.2, alpha=0.8, zorder=3,
            )

    # ----------------------------------------------------------------
    # Helper: draw a team's players with shirt numbers
    # ----------------------------------------------------------------
    def _draw_team(team: str, colour: str, is_losing: bool) -> None:
        # Outfield players: slots 2–11
        for i in range(2, 12):
            x = frame_row.get(f"x_{team}_{i}")
            y = frame_row.get(f"y_{team}_{i}")
            shirt = frame_row.get(f"id_{team}_{i}")
            if pd.isna(x) or pd.isna(y):
                continue
            x, y = float(x), float(y)
            xp, yp = _x_to_pitch(x), _y_to_pitch(y)

            is_behind = (
                is_losing
                and highlight_behind_ball
                and not math.isnan(ball_x_cm)
                and ((x < ball_x_cm) if ar_losing else (x > ball_x_cm))
            )
            size = 280 if is_behind else 200
            ax.scatter(xp, yp, s=size, c=colour,
                       edgecolors="white", linewidths=0.8, zorder=4)
            if pd.notna(shirt):
                ax.text(xp, yp, str(int(shirt)),
                        ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold", zorder=5)

        # GK: slot 1 (square marker)
        x_gk = frame_row.get(f"x_{team}_1")
        y_gk = frame_row.get(f"y_{team}_1")
        shirt_gk = frame_row.get(f"id_{team}_1")
        if not (pd.isna(x_gk) or pd.isna(y_gk)):
            xp, yp = _x_to_pitch(float(x_gk)), _y_to_pitch(float(y_gk))
            ax.scatter(xp, yp, s=220, c=colour, marker="s",
                       edgecolors="white", linewidths=0.8, zorder=4)
            if pd.notna(shirt_gk):
                ax.text(xp, yp, str(int(shirt_gk)),
                        ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold", zorder=5)

    # ----------------------------------------------------------------
    # Losing team — draw players + cluster lines
    # ----------------------------------------------------------------
    _draw_team(losing_team, losing_colour, is_losing=True)

    losing_pos = get_player_positions(frame_row, losing_team, include_gk=False)
    if losing_labels is not None and len(losing_labels) == len(losing_pos):
        for cid in np.unique(losing_labels):
            mask = losing_labels == cid
            cpos = losing_pos[mask]
            if len(cpos) < 2:
                continue
            spos = cpos[cpos[:, 1].argsort()]
            ax.plot(
                _x_to_pitch(spos[:, 0]), _y_to_pitch(spos[:, 1]),
                color=losing_colour, linewidth=1.5, alpha=0.75, linestyle="-", zorder=3,
            )

    # ----------------------------------------------------------------
    # Gaining team — draw players + cluster lines
    # ----------------------------------------------------------------
    _draw_team(gaining_team, gaining_colour, is_losing=False)

    gaining_pos = get_player_positions(frame_row, gaining_team, include_gk=False)
    if gaining_labels is not None and len(gaining_labels) == len(gaining_pos):
        for cid in np.unique(gaining_labels):
            mask = gaining_labels == cid
            cpos = gaining_pos[mask]
            if len(cpos) < 2:
                continue
            spos = cpos[cpos[:, 1].argsort()]
            ax.plot(
                _x_to_pitch(spos[:, 0]), _y_to_pitch(spos[:, 1]),
                color=gaining_colour, linewidth=1.5, alpha=0.75, linestyle="-", zorder=3,
            )

    ax.set_title(title, fontsize=8.5, color="#111111", pad=3)


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def _draw_metrics_table(
    ax: plt.Axes,
    metrics_by_offset: dict[int, dict],
    transition_row: pd.Series,
    losing_name: str,
    gaining_name: str,
    transition_metrics: dict,
) -> None:
    ax.axis("off")
    ax.set_facecolor("white")

    offsets = [0, 2, 10, 20]
    col_labels = ["Metric", "t0", "t0 +1s", "t0 +5s", "t0 +10s"]

    def _fmt(val, fmt=".1f", suffix="") -> str:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        if isinstance(val, str):
            return val
        try:
            return f"{val:{fmt}}{suffix}"
        except Exception:
            return str(val)

    # Per-offset metrics (rows with 4 values — one per time offset)
    per_offset_rows = [
        ("Total Team Length (m)",        "team_length_m",        ".1f",  ""),
        ("Rest Defence Line Height (m)", "line_height_m",         ".1f",  ""),
        ("Behind Ball",                  "players_behind_ball",   ".0f",  ""),
        ("NumSup RD App1 (Rule-Based)",   "num_superiority_app1",  "+.0f", ""),
        ("NumSup RD App2 (Clustering)",  "num_superiority_app2",  "+.0f", ""),
        ("Team Compactness (m)",         "team_compactness",      ".1f",  ""),
        ("Compact Δ (m)",                "compactness_delta",     "+.1f", ""),
        ("Pitch Control",                "pitch_control",         ".2f",  ""),
        ("Coverage Ratio",               "coverage_ratio",        ".2f",  ""),
        ("Zone Press App1",              "zone_press_app1",       ".1f",  ""),
        ("Team Press",                   "team_press",            ".1f",  ""),
    ]

    def _pct_fmt(ref, val):
        """Return '±XX.X%' string, or '—' if ref is None/NaN/0."""
        try:
            ref, val = float(ref), float(val)
            if ref == 0 or math.isnan(ref) or math.isnan(val):
                return "—"
            return f"{(val - ref) / abs(ref) * 100:+.1f}%"
        except (TypeError, ValueError):
            return "—"

    table_data = []
    for display_name, key, fmt, suffix in per_offset_rows:
        row = [display_name]
        for offset in offsets:
            m = metrics_by_offset.get(offset, {})
            row.append(_fmt(m.get(key), fmt, suffix))
        table_data.append(row)

    # Transition-level metrics
    t_m = transition_metrics or {}
    adv5  = _fmt(t_m.get("centroid_advance_5s_m"),  ".1f")
    adv10 = _fmt(t_m.get("centroid_advance_10s_m"), ".1f")
    table_data.append(["Centroid Adv (m)", "—", "—", adv5, adv10])

    pc5  = _fmt(t_m.get("pitch_control_delta_5s"),  "+.2f")
    pc10 = _fmt(t_m.get("pitch_control_delta_10s"), "+.2f")
    table_data.append(["Pitch Ctrl Δ", "—", "—", pc5, pc10])

    pr5  = _fmt(t_m.get("pressure_delta_5s"),  "+.1f")
    pr10 = _fmt(t_m.get("pressure_delta_10s"), "+.1f")
    table_data.append(["Pressure Δ", "—", "—", pr5, pr10])

    # Positive transition metrics
    cp = t_m.get("constructive_progression")
    table_data.append(["Constr. Progression", "Yes" if cp else "No" if cp is not None else "—", "—", "—", "—"])
    table_data.append(["Own Half Exit",   _fmt(t_m.get("own_half_exit"),  ""), "—", "—", "—"])
    table_data.append(["ProdPass (45°)",  _fmt(t_m.get("productive_pass_ratio_45"), ".0%"), "—", "—", "—"])
    table_data.append(["ProdPass (90°)",  _fmt(t_m.get("productive_pass_ratio_90"), ".0%"), "—", "—", "—"])
    table_data.append(["PM Dep. (1st pass)",  _fmt(t_m.get("playmaker_dependency_1st"), ""), "—", "—", "—"])
    table_data.append(["PM Dep. (2nd pass)",  _fmt(t_m.get("playmaker_dependency_2nd"), ""), "—", "—", "—"])

    # Scalar transition metrics — show in t0 column
    rating = _fmt(t_m.get("transition_rating"))
    dur    = _fmt(t_m.get("duration_s"),  ".1f")
    pc_val = t_m.get("pass_count")
    pcount = "0" if pc_val is not None and not (isinstance(pc_val, float) and math.isnan(pc_val)) and float(pc_val) == 0 else _fmt(pc_val, ".0f")
    table_data.append(["Rating",       rating, "—", "—", "—"])
    table_data.append(["Duration (s)", dur,    "—", "—", "—"])
    table_data.append(["Pass Count",   pcount, "—", "—", "—"])

    # Pressure delta rows (percentage change from t0+1s baseline)
    pv = metrics_by_offset   # shorthand: dict keyed by offset
    zp_d5 = _pct_fmt(pv.get(2, {}).get("zone_press_app1"), pv.get(10, {}).get("zone_press_app1"))
    zp_d10 = _pct_fmt(pv.get(2, {}).get("zone_press_app1"), pv.get(20, {}).get("zone_press_app1"))
    tp_d5 = _pct_fmt(pv.get(2, {}).get("team_press"), pv.get(10, {}).get("team_press"))
    es_d5 = _pct_fmt(pv.get(2, {}).get("gaining_ps_zone"), pv.get(10, {}).get("gaining_ps_zone"))
    table_data.extend([
        ["ZPress Δ%(1→5s)", "—", "—", zp_d5, "—"],
        ["ZPress Δ%(1→10s)", "—", "—", "—", zp_d10],
        ["TmPress Δ%(1→5s)", "—", "—", tp_d5, "—"],
        ["EscZ Δ%(1→5s)", "—", "—", es_d5, "—"],
    ])

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.3)

    n_cols = len(col_labels)
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#e0e0e0")
        tbl[0, j].set_text_props(color="#111111", fontweight="bold")

    for i in range(1, len(table_data) + 1):
        for j in range(n_cols):
            tbl[i, j].set_facecolor("#f5f5f5" if i % 2 == 0 else "white")
            tbl[i, j].set_text_props(color="#111111")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_transition_analysis(
    transition_row: pd.Series,
    raw_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    losing_team_label: str,
    metrics_by_offset: dict[int, dict] | None = None,
    transition_metrics: dict | None = None,
    team_name_map: dict[tuple[str, int], str] | None = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Produce a 2×2 + table figure for a single rest-defence transition.
    Zones are rebuilt fresh for each panel from pre-cached per-offset data.
    Figure background is white.
    """
    match_id  = transition_row["match_id"]
    period    = int(transition_row["period"])
    t0_frame  = int(transition_row["t0_frame"])

    dir_row = direction_df.loc[(str(match_id), period)]
    team_a_attacks_right = bool(dir_row["team_a_attacks_right"])

    gaining_team_label = "b" if losing_team_label == "a" else "a"

    # Team names
    def _name(team_label: str) -> str:
        tid = int(transition_row["losing_team_id"] if team_label == losing_team_label
                  else transition_row["gaining_team_id"])
        if team_name_map:
            return team_name_map.get((str(match_id), tid), str(tid))
        return str(tid)

    losing_name  = _name(losing_team_label)
    gaining_name = _name(gaining_team_label)

    # Home/away team names for title
    losing_team_id  = int(transition_row["losing_team_id"])
    gaining_team_id = int(transition_row["gaining_team_id"])
    home_name = (team_name_map or {}).get(
        (str(match_id), losing_team_id if losing_team_label == "a" else gaining_team_id), "Home"
    )
    away_name = (team_name_map or {}).get(
        (str(match_id), losing_team_id if losing_team_label == "b" else gaining_team_id), "Away"
    )

    # Game time from `t` column
    t0_row = get_frame(raw_df, match_id, t0_frame)
    game_time_str = ""
    if t0_row is not None:
        t_val = t0_row.get("t")
        if pd.notna(t_val):
            t_ms  = int(float(t_val))
            mins  = t_ms // 60_000
            secs  = (t_ms % 60_000) // 1_000
            game_time_str = f"{mins}'{secs:02d}\""

    # Precompute zones at each offset
    offsets = [0, 2, 10, 20]
    titles  = [
        f"t0  —  {transition_row['losing_end_event']} → {transition_row['gaining_start_event']}",
        "t0 + 1s",
        "t0 + 5s",
        "t0 + 10s",
    ]

    zones_by_offset: dict[int, tuple] = {}
    for offset in offsets:
        frow = get_frame(raw_df, match_id, t0_frame + offset)
        if frow is not None:
            app1, app2, app3, bk, ll, gl = build_zones(frow, losing_team_label, team_a_attacks_right)
            zones_by_offset[offset] = (app1, app2, app3, bk, ll, gl)

    if metrics_by_offset is None:
        metrics_by_offset = {}
    if transition_metrics is None:
        transition_metrics = {}

    # ----------------------------------------------------------------
    # Figure layout: 2 rows of 2 pitches + 1 metrics row
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(22, 26), facecolor="white")
    gs  = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.0, 2.0, 4.0],
        hspace=0.18,
        wspace=0.06,
    )

    pitch_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    table_ax = fig.add_subplot(gs[2, :])
    table_ax.set_facecolor("white")

    for col_idx, (offset, title) in enumerate(zip(offsets, titles)):
        row_gs, col_gs = pitch_positions[col_idx]
        ax = fig.add_subplot(gs[row_gs, col_gs])
        ax.set_facecolor("white")

        frow = get_frame(raw_df, match_id, t0_frame + offset)
        if frow is None or offset not in zones_by_offset:
            ax.set_title(f"{title}\n(no data)", color="#111111", fontsize=8.5)
            continue

        app1, app2, app3, bk, ll, gl = zones_by_offset[offset]

        _draw_pitch_panel(
            ax=ax,
            frame_row=frow,
            zone_app1=app1,
            zone_app2=app2,
            zone_app3=app3,
            losing_labels=ll,
            gaining_labels=gl,
            losing_team=losing_team_label,
            gaining_team=gaining_team_label,
            team_a_attacks_right=team_a_attacks_right,
            title=title,
            draw_ball_line=(offset == 0),
            highlight_behind_ball=(offset == 0),
        )

    _draw_metrics_table(
        table_ax,
        metrics_by_offset,
        transition_row,
        losing_name,
        gaining_name,
        transition_metrics,
    )

    fig.suptitle(
        f"{home_name} (Blue Team)  vs  {away_name} (Red Team)   |   Period {period}   |   {game_time_str}\n"
        f"Defending (rest defence): {losing_name}   |   Attacking: {gaining_name}",
        fontsize=11,
        color="#111111",
        y=0.99,
    )
    # Team logos flanking the suptitle (home left, away right)
    add_team_logos(fig, home_name, away_name, y_bottom=0.958, logo_height_in=0.70)

    if output_path:
        fig.savefig(output_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    return fig
