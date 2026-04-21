"""
spider_plots.py
---------------
Radar / spider plots showing Negative Transition attack quality per team.
One mplsoccer Radar per team, laid out side by side in a single figure.
Data source: gaining_team_name filter (team's own attacking transitions).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Radar

from src.visualizations import BRAND_PALETTE, HEADER_COLOUR, EXCLUDED_TEAMS, PRIMARY_RED, PRIMARY_ORANGE

_PARAMS = [
    "Constructive\nProgression",
    "Own Half\nExit",
    "Fwd Pass\nRatio 45°",
    "Fwd Pass\nRatio 90°",
    "Playmaker\nIndep. 1st",
    "Playmaker\nIndep. First 2",
]

_radar = Radar(
    params=_PARAMS,
    min_range=[0.0] * 6,
    max_range=[100.0] * 6,
    round_int=[False] * 6,
    num_rings=4,
    ring_width=1,
    center_circle_radius=1,
)


def _team_values(df: pd.DataFrame, team: str) -> list[float]:
    gdf = df[df["gaining_team_name"] == team]
    if len(gdf) == 0:
        return [0.0] * 6

    def b(col):  # bool / object columns → 0–100
        return float(pd.to_numeric(gdf[col], errors="coerce").astype(float).mean() * 100)

    def r(col):  # ratio columns already 0–1 → 0–100
        return float(pd.to_numeric(gdf[col], errors="coerce").mean() * 100)

    # Playmaker independency: invert dependency (higher = more independent)
    pm_dep_1st = b("playmaker_dependency_1st")
    pm_indep_1st = 100.0 - pm_dep_1st

    # Playmaker independency over first 2 passes: 100 - dep_1st - dep_2nd
    pm_dep_2nd = b("playmaker_dependency_2nd")
    pm_indep_either = max(0.0, 100.0 - pm_dep_1st - pm_dep_2nd)

    vals = [
        b("constructive_progression"),
        b("own_half_exit"),
        r("productive_pass_ratio_45"),
        r("productive_pass_ratio_90"),
        pm_indep_1st,
        pm_indep_either,
    ]
    return [0.0 if np.isnan(v) else round(v, 1) for v in vals]


def plot_spider_absolute(
    df: pd.DataFrame,
    teams: list[str] | None = None,
    output_path: str | None = None,
) -> plt.Figure:
    """
    One mplsoccer Radar per team, side by side in a single figure.
    teams: list of team names (default: Como, Hellas Verona, Juventus).
    """
    from src.logos import get_logo_image

    if teams is None:
        teams = ["Como", "Hellas Verona", "Juventus"]

    df = df[~df["gaining_team_name"].isin(EXCLUDED_TEAMS)]
    n = len(teams)

    # mplsoccer Radar uses regular (non-polar) Cartesian axes — no projection needed
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))
    fig.patch.set_facecolor("#FAFAFA")
    if n == 1:
        axes = [axes]

    for i, (team, ax) in enumerate(zip(teams, axes)):
        vals = _team_values(df, team)

        _radar.setup_axis(ax=ax, facecolor="#FAFAFA")
        _radar.draw_circles(ax=ax, facecolor="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.9)
        radar_poly, rings_outer, vertices = _radar.draw_radar(
            vals, ax=ax,
            kwargs_radar={"facecolor": PRIMARY_ORANGE, "alpha": 0.35},
            kwargs_rings={"facecolor": PRIMARY_ORANGE, "alpha": 0.12},
        )
        ax.scatter(vertices[:, 0], vertices[:, 1],
                   c=PRIMARY_RED, edgecolors="#333333", marker="o", s=60, zorder=5)
        _radar.draw_range_labels(ax=ax, fontsize=9, color="#888888")
        _radar.draw_param_labels(ax=ax, fontsize=10, color="#222222")
        _radar.spoke(ax=ax, color="#CCCCCC", linestyle="--", linewidth=0.7, zorder=2)

        ax.set_title(team, color=PRIMARY_RED, fontsize=12, fontweight="bold", pad=14)

    fig.suptitle(
        "Transition Dynamics - Negative Transition",
        fontsize=14, fontweight="bold", color=PRIMARY_RED, y=0.98
    )

    fig.text(
        0.5, 0.01,
        "All metrics from transitions where the team was GAINING possession.  Scale 0–100.  "
        "Constructive Progression = ≥3 passes within 15s.  Own Half Exit = possession advanced beyond own half.  "
        "Forward Pass Ratio = % passes directed forward.  Playmaker Indep. 1st = % transitions NOT dependent on key playmaker in 1st pass.  "
        "Playmaker Indep. First 2 = sum of 1st pass independency + 2nd pass involvement.",
        ha="center", va="bottom", fontsize=7.5, color="#555", style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Place logos after layout is finalised so ax.get_position() is accurate
    for i, (team, ax) in enumerate(zip(teams, axes)):
        logo = get_logo_image(team, size=128)
        if logo is not None:
            ax_pos = ax.get_position()
            logo_w, logo_h = 0.055, 0.07
            logo_ax = fig.add_axes([
                ax_pos.x1 - logo_w - 0.002,
                ax_pos.y1 + 0.01,   # just above the top edge → beside the title
                logo_w, logo_h,
            ])
            logo_ax.imshow(logo)
            logo_ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig


def plot_spider_single_team(
    df: pd.DataFrame,
    team: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Single radar for one team showing Negative Transition metrics.
    Compares team values to league mean (overlaid as dashed gray reference).
    teams: single team name (e.g., "Juventus")
    """
    from src.logos import get_logo_image

    df = df[~df["gaining_team_name"].isin(EXCLUDED_TEAMS)]

    team_vals = _team_values(df, team)

    # Compute league mean values across all teams (same metrics)
    all_teams = df["gaining_team_name"].unique()
    league_vals = [0.0] * 6
    for i in range(6):
        vals = [_team_values(df, t)[i] for t in all_teams]
        league_vals[i] = float(np.nanmean(vals)) if vals else 0.0

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#FAFAFA")

    # Setup radar
    _radar.setup_axis(ax=ax, facecolor="#FAFAFA")
    _radar.draw_circles(ax=ax, facecolor="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.9)

    # Draw league mean first (background reference, dashed gray)
    league_poly, _, league_verts = _radar.draw_radar(
        league_vals, ax=ax,
        kwargs_radar={"facecolor": "#CCCCCC", "alpha": 0.25, "edgecolor": "#555555", "linestyle": "--", "linewidth": 2.2},
        kwargs_rings={"facecolor": "#E8E8E8", "alpha": 0.05},
    )

    # Draw team on top (solid orange)
    team_poly, _, team_verts = _radar.draw_radar(
        team_vals, ax=ax,
        kwargs_radar={"facecolor": PRIMARY_ORANGE, "alpha": 0.35, "edgecolor": PRIMARY_ORANGE, "linewidth": 1.8},
        kwargs_rings={"facecolor": PRIMARY_ORANGE, "alpha": 0.12},
    )

    # Markers
    ax.scatter(team_verts[:, 0], team_verts[:, 1],
               c=PRIMARY_RED, edgecolors="#333333", marker="o", s=80, zorder=5)
    ax.scatter(league_verts[:, 0], league_verts[:, 1],
               c="#888888", edgecolors="#666666", marker="o", s=60, zorder=4, alpha=0.6)

    _radar.draw_range_labels(ax=ax, fontsize=9, color="#888888")
    _radar.draw_param_labels(ax=ax, fontsize=10, color="#222222")
    _radar.spoke(ax=ax, color="#CCCCCC", linestyle="--", linewidth=0.7, zorder=2)

    ax.set_title(f"{team} — Transition Dynamics (Negative Transition)",
                 color=PRIMARY_RED, fontsize=13, fontweight="bold", pad=14)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=PRIMARY_ORANGE, linewidth=2, label=team),
        Line2D([0], [0], color="#888888", linewidth=1.5, linestyle="--", label="League mean"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="upper left",
              framealpha=0.7, edgecolor="#ccc")

    fig.text(
        0.5, 0.01,
        "All metrics from transitions where the team was GAINING possession. Scale 0–100.\n"
        "Constructive Progression = ≥3 passes within 15s. Own Half Exit = possession advanced beyond own half.\n"
        "Forward Pass Ratio = % passes directed forward. Playmaker Indep. = % transitions NOT dependent on key playmaker.",
        ha="center", va="bottom", fontsize=7.5, color="#555", style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Logo placement
    logo = get_logo_image(team, size=128)
    if logo is not None:
        ax_pos = ax.get_position()
        logo_w, logo_h = 0.08, 0.1
        logo_ax = fig.add_axes([
            ax_pos.x1 - logo_w - 0.01,
            ax_pos.y1 - 0.02,   # Lowered position (inside the plot area slightly)
            logo_w, logo_h,
        ])
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig
