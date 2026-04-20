"""
video.py
--------
Generates a short MP4 video around a rest-defence transition.

The clip starts 5 seconds before t0 (the moment possession is lost) and
runs 10 seconds after, giving 15 seconds of context at tracking-data rate.

Raw tracking data is at 2 fps (one frame per 500 ms).  The video is saved
at 10 fps via linear interpolation of player and ball positions between
consecutive tracking frames, yielding smooth on-screen movement.

Zones (App1, App2) are recomputed only at actual tracking frames and held
constant for the 4 interpolated sub-frames between each pair, which avoids
zone flickering while keeping computation fast.

No metrics table is included — this is a pitch-only visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data_loading import (
    get_frame,
    get_player_positions,
)
from .logos import add_team_logos
from .rest_defence_area import build_zones
from .visualisation import _draw_pitch_panel, _team_colour

# Tracking data rate (frames per second)
TRACKING_FPS = 2
# Output video fps (must be a multiple of TRACKING_FPS for clean duplication)
OUTPUT_FPS   = 10
# Number of interpolated sub-frames per tracking interval
N_SUB = OUTPUT_FPS // TRACKING_FPS   # = 5

# Default clip window
PRE_CONTEXT_FRAMES  = TRACKING_FPS * 5   # 5s before t0
POST_CONTEXT_FRAMES = TRACKING_FPS * 10  # 10s after t0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_row(row_a: pd.Series, row_b: pd.Series, alpha: float) -> pd.Series:
    """
    Linear interpolation of position columns between two tracking frames.
    Only x_ and y_ prefixed columns (players + ball) are interpolated.
    All other columns are copied from row_a unchanged.
    """
    if alpha == 0.0:
        return row_a
    interp = row_a.copy()
    for col in row_a.index:
        if not (col.startswith("x_") or col.startswith("y_")):
            continue
        va = row_a.get(col)
        vb = row_b.get(col)
        if pd.notna(va) and pd.notna(vb):
            interp[col] = float(va) + alpha * (float(vb) - float(va))
    return interp


def _game_time_str(t_ms_series_val) -> str:
    """Convert a raw `t` column value (ms) to mm'ss" string."""
    if pd.isna(t_ms_series_val):
        return ""
    t_ms = int(float(t_ms_series_val))
    mins = t_ms // 60_000
    secs = (t_ms % 60_000) // 1_000
    return f"{mins}'{secs:02d}\""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_transition_video(
    transition_row: pd.Series,
    raw_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    losing_team_label: str,
    team_name_map: Optional[dict] = None,
    output_path: str = "transition.mp4",
    pre_context_s: int = 5,
    output_fps: int = OUTPUT_FPS,
    tracking_fps: int = TRACKING_FPS,
) -> Path:
    """
    Render an MP4 video for a single rest-defence transition.

    Parameters
    ----------
    transition_row   : one row from the transitions DataFrame
    raw_df           : full raw tracking data
    direction_df     : attack-direction table (indexed by match_id, period)
    losing_team_label: 'a' or 'b'
    team_name_map    : {(match_id, team_id): name} for title
    output_path      : destination .mp4 file
    pre_context_s    : seconds before t0 to start the clip (default 5)
    output_fps       : output video frame rate (default 10)
    tracking_fps     : raw data frame rate (default 2)

    Returns
    -------
    Path to the saved video file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    n_sub = output_fps // tracking_fps

    match_id   = transition_row["match_id"]
    period     = int(transition_row["period"])
    t0_frame   = int(transition_row["t0_frame"])

    dir_row = direction_df.loc[(str(match_id), period)]
    team_a_attacks_right = bool(dir_row["team_a_attacks_right"])

    gaining_team_label = "b" if losing_team_label == "a" else "a"

    # Team names for title
    def _name(team_label: str) -> str:
        tid = int(transition_row["losing_team_id"] if team_label == losing_team_label
                  else transition_row["gaining_team_id"])
        if team_name_map:
            return team_name_map.get((str(match_id), tid), str(tid))
        return str(tid)

    losing_name  = _name(losing_team_label)
    gaining_name = _name(gaining_team_label)

    # Home/away names for suptitle
    losing_team_id  = int(transition_row["losing_team_id"])
    gaining_team_id = int(transition_row["gaining_team_id"])
    home_name = (team_name_map or {}).get(
        (str(match_id), losing_team_id if losing_team_label == "a" else gaining_team_id), "Home"
    )
    away_name = (team_name_map or {}).get(
        (str(match_id), losing_team_id if losing_team_label == "b" else gaining_team_id), "Away"
    )

    # Clip frame range
    pre_frames  = pre_context_s * tracking_fps
    post_frames = POST_CONTEXT_FRAMES
    start_frame = t0_frame - pre_frames
    end_frame   = t0_frame + post_frames

    # Collect all tracking frames in the clip window
    tracking_frames: list[pd.Series | None] = [
        get_frame(raw_df, match_id, f)
        for f in range(start_frame, end_frame + 1)
    ]
    n_tracking = len(tracking_frames)

    # Pre-compute zones at each actual tracking frame (cached)
    zones_cache: dict[int, tuple] = {}
    for ti, frow in enumerate(tracking_frames):
        if frow is not None:
            zones_cache[ti] = build_zones(frow, losing_team_label, team_a_attacks_right)

    # Game time string from the t0 row
    t0_row = tracking_frames[pre_frames] if pre_frames < n_tracking else None
    game_time_str = ""
    if t0_row is not None:
        game_time_str = _game_time_str(t0_row.get("t"))

    # ----------------------------------------------------------------
    # Figure setup
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.suptitle(
        f"{home_name} (Blue)  vs  {away_name} (Red)   |   Period {period}   |   {game_time_str}\n"
        f"Defending: {losing_name}   |   Attacking: {gaining_name}",
        fontsize=10,
        color="#111111",
        y=0.99,
    )
    fig.subplots_adjust(top=0.93)
    add_team_logos(fig, home_name, away_name, y_bottom=0.935, logo_height_in=0.55)

    # ----------------------------------------------------------------
    # Animation function
    # ----------------------------------------------------------------
    n_anim_frames = (n_tracking - 1) * n_sub + 1

    def animate(anim_i: int):
        ax.clear()
        ax.set_facecolor("white")

        ti = anim_i // n_sub        # tracking frame index
        si = anim_i % n_sub         # sub-frame index (0 = exact tracking frame)

        row_a = tracking_frames[ti]
        row_b = (tracking_frames[ti + 1]
                 if ti + 1 < n_tracking else row_a)

        if row_a is None:
            return []

        alpha = si / n_sub
        frow  = _interpolate_row(row_a, row_b, alpha) if si > 0 and row_b is not None else row_a

        # Use zones from the nearest actual tracking frame
        zone_tuple = zones_cache.get(ti) or zones_cache.get(min(zones_cache, key=lambda k: abs(k - ti)))
        if zone_tuple is None:
            return []
        app1, app2, app3, bk, ll, gl = zone_tuple

        # Time label relative to t0
        tracking_frame_num = start_frame + ti
        elapsed_s = (tracking_frame_num - t0_frame) / tracking_fps
        if elapsed_s < 0:
            title = f"t − {abs(elapsed_s):.0f}s"
        elif elapsed_s == 0:
            title = "▶  t0  (TRANSITION)"
        else:
            title = f"t + {elapsed_s:.0f}s"

        is_t0_exact = (si == 0 and tracking_frame_num == t0_frame)

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
            draw_ball_line=is_t0_exact,
            highlight_behind_ball=is_t0_exact,
        )
        return []

    anim_obj = animation.FuncAnimation(
        fig, animate,
        frames=n_anim_frames,
        interval=1000 // output_fps,
        blit=False,
        repeat=False,
    )

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.FFMpegWriter(
        fps=output_fps,
        bitrate=2000,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    anim_obj.save(str(out), writer=writer, dpi=100)
    plt.close(fig)

    return out
