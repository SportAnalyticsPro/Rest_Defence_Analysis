"""
scatterplots.py
---------------
Scatter plots with team logos as point markers.
Each point = one team (mean across all their defending transitions).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.visualizations import BRAND_PALETTE, HEADER_COLOUR, EXCLUDED_TEAMS, PRIMARY_RED
from src.logos import get_logo_image

_LOGO_ZOOM  = 0.10   # fixed size — logos are the label, no size encoding needed
_FALLBACK_C = BRAND_PALETTE[0]  # red dot for teams without a logo


def _plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str | None,
    x_unit: str = "m",
    y_unit: str = "m",
) -> plt.Figure:
    df = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]

    team_means = df.groupby("losing_team_name")[[x_col, y_col]].mean()
    league_x   = team_means[x_col].median()
    league_y   = team_means[y_col].median()

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Crosshair at league median — label on each line
    ax.axvline(league_x, color="#AAAAAA", linewidth=1, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(league_y, color="#AAAAAA", linewidth=1, linestyle="--", alpha=0.6, zorder=1)

    # Subtle quadrant labels
    x_range = team_means[x_col].max() - team_means[x_col].min()
    y_range = team_means[y_col].max() - team_means[y_col].min()
    pad_x = x_range * 0.03
    pad_y = y_range * 0.03

    for team, row in team_means.iterrows():
        x_val, y_val = row[x_col], row[y_col]
        logo = get_logo_image(str(team), size=256)
        if logo is not None:
            img = OffsetImage(logo, zoom=_LOGO_ZOOM)
            ab = AnnotationBbox(
                img, (x_val, y_val),
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                zorder=3,
            )
            ax.add_artist(ab)
        else:
            ax.plot(x_val, y_val, "o", color=_FALLBACK_C, markersize=14, zorder=3)
            ax.annotate(
                str(team), (x_val, y_val),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=7.5, color=_FALLBACK_C,
            )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", color=PRIMARY_RED, pad=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, linewidth=0.6)

    # Add legend for the league median lines
    from matplotlib.lines import Line2D
    legend_line = Line2D([0], [0], color="#AAAAAA", linewidth=1, linestyle="--", label="League median")
    ax.legend(handles=[legend_line], fontsize=9, loc="best", framealpha=0.7)

    # Add a little padding so logos at the edges aren't clipped
    x_pad = x_range * 0.12
    y_pad = y_range * 0.12
    ax.set_xlim(team_means[x_col].min() - x_pad, team_means[x_col].max() + x_pad)
    ax.set_ylim(team_means[y_col].min() - y_pad, team_means[y_col].max() + y_pad)

    # Add median values as axis annotations (aligned with tick labels)
    ax.annotate(f"{league_x:.1f} {x_unit}", 
                xy=(league_x, 0), xycoords=("data", "axes fraction"),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, color="#888888", fontweight="bold")

    # Y-axis median label (right of the y-axis)
    ax.annotate(f"{league_y:.1f} {y_unit}", 
                xy=(0, league_y), xycoords=("axes fraction", "data"),
                xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color="#888888", fontweight="bold")

    plt.tight_layout(pad=2.0)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")
    return fig


def plot_compactness_vs_length(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    """Team compactness (x) vs team length (y) at t0."""
    return _plot_scatter(
        df,
        x_col="team_compactness_t0",
        y_col="team_length_m_t0",
        x_label="Team Compactness at Start (m)",
        y_label="Team Length at Start (m)",
        title="Team Shape after Ball Lost — Compactness vs Length",
        output_path=output_path,
    )


def plot_lineheight_vs_cadv(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    """Line height at t0 (x) vs centroid advance at 5s (y)."""
    return _plot_scatter(
        df,
        x_col="line_height_m_t0",
        y_col="centroid_advance_5s_m",
        x_label="Defensive Line Height at Start (m from own goal)",
        y_label="Centroid Advance in 5 s (m)",
        title="Defensive Line Position vs Recovery Speed",
        output_path=output_path,
    )


def plot_foul_time_vs_location(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Avg time to foul (x) vs avg foul location (y) — one point per team.
    Only transitions where foul_committed == True are included.
    foul_x_m: 0 = own goal, 105 = opponent goal → higher = foul high up the pitch.
    """
    df = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]
    foul_df = df[df["foul_committed"].astype(bool)].copy()

    return _plot_scatter(
        foul_df,
        x_col="foul_time_s",
        y_col="foul_x_m",
        x_label="Avg Time to Foul after Transition (s)",
        y_label="Avg Foul Location — Distance from Own Goal (m)",
        title="Foul Analysis — When and Where Teams Foul",
        output_path=output_path,
        x_unit="s",
        y_unit="m",
    )
