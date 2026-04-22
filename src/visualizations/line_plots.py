"""
line_plots.py
-------------
Evolution line plots: one figure per structural metric, showing how the
average value changes from Start → t0+5s → t0+10s for each defending team.
Logos are shown at each data point, sized proportionally to the metric value.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualizations import BRAND_PALETTE, HEADER_COLOUR, EXCLUDED_TEAMS, PRIMARY_RED

_METRICS = {
    "team_length_m": (
        "Team Length",
        "metres",
        "Team Length — Evolution after Ball Lost",
    ),
    "team_compactness": (
        "Team Compactness",
        "metres",
        "Team Compactness — Evolution after Ball Lost",
    ),
    "team_centroid_x_norm": (
        "Centroid Distance from Own Goal",
        "metres",
        "Team Centroid Position — Evolution after Ball Lost",
    ),
    "n_pressing_team": (
        "Number of Pressing Players",
        "players",
        "Pressing Players — Evolution after Ball Lost",
    ),
}

_X_LABELS = ["Start", "After 5s", "After 10s"]
_OFFSETS  = ["t0", "t50", "t100"]


def plot_metric_evolution(
    df: pd.DataFrame,
    metric: str,
    teams: list[str] | None = None,
    output_path: str | None = None,
    offsets: tuple[str, ...] | None = None,
    labels: list[str] | None = None,
) -> plt.Figure:
    """
    Line plot for *metric* at specified time offsets.
    teams: optional list of team names to show (default: all, minus EXCLUDED_TEAMS).
    offsets: tuple of offset suffixes (e.g., ("t0", "t10", "t20")) — defaults to _OFFSETS
    labels: list of labels for x-axis (e.g., ["Start", "After 5s", "After 10s"]) — defaults to _X_LABELS
    Logos are placed at each data point, sized proportionally to the metric value.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from src.logos import get_logo_image

    if metric not in _METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(_METRICS)}")

    label, unit, title = _METRICS[metric]

    df = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]
    if teams:
        df = df[df["losing_team_name"].isin(teams)]

    if offsets is None:
        offsets = _OFFSETS
    if labels is None:
        labels = _X_LABELS

    cols = [f"{metric}_{suf}" for suf in offsets]
    team_means = (
        df.groupby("losing_team_name")[cols]
        .mean()
        .sort_values(cols[0])
    )

    all_teams = team_means.index.tolist()
    x = np.arange(len(labels))

    _FIXED_ZOOM = 0.07   # fixed logo size — same at every point

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # League mean (all teams in filtered set)
    league_means = df[cols].mean()
    ax.plot(x, league_means.values, color="#BBBBBB", linewidth=1.5,
            linestyle="--", marker="o", markersize=5, zorder=1, label="League mean")

    # Per-team lines
    for i, team in enumerate(all_teams):
        vals = team_means.loc[team, cols].values
        colour = BRAND_PALETTE[i % len(BRAND_PALETTE)]
        ax.plot(x, vals, color=colour, linewidth=2, linestyle="-",
                zorder=2, alpha=0.85)

        # Logo at each data point
        logo = get_logo_image(team, size=256)
        for xi, val in zip(x, vals):
            if logo is not None:
                img = OffsetImage(logo, zoom=_FIXED_ZOOM)
                ab = AnnotationBbox(
                    img, (xi, val),
                    frameon=False,
                    box_alignment=(0.5, 0.5),  # centre of logo on the data point
                    pad=0,
                    zorder=3,
                )
                ax.add_artist(ab)
            else:
                # Fallback: coloured dot for teams without a logo
                ax.plot(xi, val, "o", color=colour, markersize=10, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f"{label} ({unit})", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=PRIMARY_RED, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_xlim(-0.4, x[-1] + 0.4)

    # Legend: team name + colour patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=BRAND_PALETTE[i % len(BRAND_PALETTE)],
               linewidth=2, label=t)
        for i, t in enumerate(all_teams)
    ]
    legend_handles.append(
        Line2D([0], [0], color="#BBBBBB", linewidth=1.5,
               linestyle="--", label="League mean")
    )
    ax.legend(handles=legend_handles, fontsize=9, loc="best",
              framealpha=0.7, edgecolor="#ccc")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig
