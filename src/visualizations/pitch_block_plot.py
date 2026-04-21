"""
pitch_block_plot.py
-------------------
Per-team pitch block plot: 3 mplsoccer pitches stacked vertically showing the
average defensive shape at Start (t0), t0+5s, t0+10s after possession loss.
Shared x-axis enables easy time comparison. Generates one PNG per team.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from src.visualizations import BRAND_PALETTE, HEADER_COLOUR, EXCLUDED_TEAMS, PRIMARY_ORANGE, PRIMARY_RED

_SNAP_COLOURS  = [PRIMARY_RED, PRIMARY_RED, PRIMARY_RED]
_SNAP_LABELS   = ["Start", "After 5s", "After 10s"]
_SNAP_SUFFIXES = ["t0", "t10", "t20"]

_PITCH_LENGTH = 105.0
_PITCH_WIDTH  = 68.0
_Y_MID        = _PITCH_WIDTH / 2   # 34 m

# Extra space (in data coords) added around the pitch for annotations
_PAD_BELOW = 5.0   # for x-axis labels (line height, centroid)
_PAD_ABOVE = 5.0   # for team-length bracket
_PAD_RIGHT = 10.0   # for compactness bracket
_PAD_LEFT  = 2.0


def _snap_means(tdf: pd.DataFrame, suffix: str) -> dict:
    return {
        "centroid_x":  tdf[f"team_centroid_x_norm_{suffix}"].mean(),
        "length_m":    tdf[f"team_length_m_{suffix}"].mean(),
        "compactness": tdf[f"team_compactness_{suffix}"].mean(),
        "line_height": tdf[f"line_height_m_{suffix}"].mean(),
    }


def _draw_snap(ax: plt.Axes, snap: dict, colour: str) -> None:
    cx  = snap["centroid_x"]
    lm  = snap["length_m"]
    com = snap["compactness"]
    lh  = snap["line_height"]

    if any(np.isnan(v) for v in (cx, lm, com, lh)):
        return

    y_bot = -_PAD_BELOW

    # ------------------------------------------------------------------
    # 1. Rest Defence Line — solid vertical
    # ------------------------------------------------------------------
    ax.plot([lh, lh], [0, _PITCH_WIDTH], color=PRIMARY_ORANGE,
            lw=1.8, ls="-", alpha=0.8, zorder=3)
    ax.text(lh, y_bot,
            f"Rest Defence\nLine (mean)\n{lh:.1f} m",
            ha="center", va="top", fontsize=7, color=PRIMARY_ORANGE,
            fontweight="bold", zorder=6, clip_on=False)

    # ------------------------------------------------------------------
    # 2. Block Circle: Radius = Compactness
    # ------------------------------------------------------------------
    circle = mpatches.Circle(
        (cx, _Y_MID), com,
        linewidth=1.5, edgecolor=PRIMARY_ORANGE, facecolor=PRIMARY_ORANGE,
        alpha=0.18, zorder=4,
    )
    ax.add_patch(circle)

    # ------------------------------------------------------------------
    # 3. Radius Line and Compactness Text (Horizontal inside circle)
    # ------------------------------------------------------------------
    ax.plot([cx, cx + com], [_Y_MID, _Y_MID], color=PRIMARY_ORANGE,
            lw=1.5, ls="-", zorder=6)
    ax.text(cx + com/2, _Y_MID + 0.8,
            f"Compactness \n (mean): {com:.1f} m",
            ha="center", va="bottom", fontsize=6.5, color=PRIMARY_RED,
            fontweight="bold", zorder=7)

    # ------------------------------------------------------------------
    # 4. Team Length — bracket at the top
    # ------------------------------------------------------------------
    x_l, x_r = cx - lm / 2, cx + lm / 2
    bracket_y = _PITCH_WIDTH + _PAD_ABOVE * 0.45

    for x_edge in (x_l, x_r):
        ax.plot([x_edge, x_edge], [_PITCH_WIDTH, bracket_y],
                color=PRIMARY_ORANGE, lw=0.8, ls="--", alpha=0.7, zorder=3, clip_on=False)

    ax.annotate(
        "", xy=(x_r, bracket_y), xytext=(x_l, bracket_y),
        arrowprops=dict(arrowstyle="<->", color=PRIMARY_ORANGE, lw=1.4,
                        mutation_scale=10),
        zorder=6, clip_on=False
    )
    ax.text((x_l + x_r) / 2, bracket_y + 2.5,
            f"Team Length (mean): {lm:.1f} m",
            ha="center", va="bottom", fontsize=7, color=PRIMARY_ORANGE,
            fontweight="bold", zorder=6, clip_on=False)

    # ------------------------------------------------------------------
    # 5. Centroid — markers and vertical line to bottom
    # ------------------------------------------------------------------
    ax.plot([cx, cx], [0, _Y_MID], color=PRIMARY_ORANGE,
            lw=1.2, ls=":", alpha=0.7, zorder=3)
    
    ax.scatter(cx, _Y_MID, color=PRIMARY_RED, s=70, zorder=10,
               edgecolors="#333333", linewidths=0.8)

    # Dynamic horizontal offset to avoid overlap between labels
    centroid_x_display = cx
    if abs(cx - lh) < 15.0:
        direction = 1 if cx >= lh else -1
        centroid_x_display = cx + direction * 4.0

    ax.text(centroid_x_display, y_bot,
            f"Centroid (mean)\n{cx:.1f} m",
            ha="center", va="top", fontsize=7, color=PRIMARY_ORANGE,
            fontweight="bold", zorder=6, clip_on=False)


def plot_pitch_block(
    df: pd.DataFrame,
    team: str,
    output_path: str | None = None,
) -> plt.Figure:
    """Draw 3 vertical pitches (Start / t0+5s / t0+10s) for *team* with shared x-axis."""
    from src.logos import get_logo_image

    tdf = df[df["losing_team_name"] == team]
    if len(tdf) == 0:
        raise ValueError(f"No transitions found for team '{team}'")

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=_PITCH_LENGTH,
        pitch_width=_PITCH_WIDTH,
        pitch_color="#FAFAFA",
        line_color="#BBBBBB",
        linewidth=0.9,
        line_zorder=2,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, suffix, colour, snap_label in zip(
        axes, _SNAP_SUFFIXES, _SNAP_COLOURS, _SNAP_LABELS
    ):
        pitch.draw(ax=ax)
        ax.set_axis_on()

        snap = _snap_means(tdf, suffix)
        _draw_snap(ax, snap, colour)

        ax.set_title(snap_label, fontsize=10, fontweight="bold",
                     color=colour, pad=15)

        # ------------------------------------------------------------------
        # Visual axes coinciding with pitch lines (0,0)
        # ------------------------------------------------------------------
        # 1. Restore the view padding so labels aren't cut off
        ax.set_xlim(-_PAD_LEFT, _PITCH_LENGTH + _PAD_RIGHT)
        ax.set_ylim(-_PAD_BELOW, _PITCH_WIDTH + _PAD_ABOVE)

        # 2. Position spines exactly on the pitch boundaries
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['right'].set_position(('data', _PITCH_LENGTH))
        ax.spines['bottom'].set_bounds(0, 105)
        ax.spines['right'].set_bounds(0, 68)

        # 3. Configure ticks on all subplots
        ax.set_xticks(list(range(0, 101, 10)) + [105])
        ax.set_yticks(list(range(0, 61, 10)) + [68])
        ax.tick_params(axis='x', which='major', labelsize=7, colors='#888888',
                       bottom=True, labelbottom=True)
        ax.tick_params(axis='y', which='major', labelsize=7, colors='#888888',
                       left=False, labelleft=False, right=True, labelright=True)

        for spine in ['top', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_color('#BBBBBB')
        ax.spines['right'].set_color('#BBBBBB')

    fig.suptitle(
        f"{team} — Defensive Block Shape After Losing the Ball",
        fontsize=13, fontweight="bold", color=PRIMARY_RED, y=1.01,
    )

    # Legend removed as requested

    plt.tight_layout()

    # Logo — placed after layout is finalised, positioned near top-right
    logo = get_logo_image(team, size=256)
    if logo is not None:
        ax_pos = axes[0].get_position()
        logo_ax = fig.add_axes([
            ax_pos.x1 - 0.01,
            0.96,
            0.05, 0.09,
        ])
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig
