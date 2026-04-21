"""
ratings_bar.py
--------------
100% stacked horizontal bar chart showing transition rating distribution per team.
Sorted by %Best descending.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.visualizations import RATING_COLOURS, HEADER_COLOUR, EXCLUDED_TEAMS, PRIMARY_RED


def plot_ratings_bar(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> plt.Figure:
    """
    100% stacked horizontal bar per team.
    Stack order (left→right): Best, Good, Okay, Bad.
    Sorted by %Best descending.
    """
    ratings = ["Best", "Good", "Okay", "Bad"]

    # Exclude non-league / incomplete teams
    df = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]

    # Compute per-team rating percentages
    counts = (
        df.groupby(["losing_team_name", "transition_rating"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all rating columns exist
    for r in ratings:
        if r not in counts.columns:
            counts[r] = 0
    counts = counts[ratings]
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100
    pcts = pcts.sort_values("Best", ascending=True)   # ascending so best is at top of horizontal bar

    teams = pcts.index.tolist()
    n = len(teams)
    y = np.arange(n)

    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(13, max(5, n * 0.55 + 1.5)),
        gridspec_kw={"width_ratios": [10, 1]},
    )
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    left = np.zeros(n)
    handles = []
    for r in ratings:
        vals = pcts[r].values
        colour = RATING_COLOURS[r]
        bars = ax.barh(y, vals, left=left, color=colour, height=0.65, linewidth=0)
        # Label inside segment if wide enough
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 3:
                ax.text(
                    l + v / 2, i,
                    f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if r in ("Bad", "Best") else "#1a1a1a",
                )
        left += vals
        handles.append(mpatches.Patch(color=colour, label=r))

    ax.set_yticks(y)
    ax.set_yticklabels(teams, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of transitions", fontsize=10)
    ax.set_title("Transition Rating Distribution (sorted by % Best)", fontsize=12,
                 fontweight="bold", color=PRIMARY_RED, pad=10)
    ax.axvline(x=50, color="#888", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    # Glossary below chart
    glossary = (
        "Best = Defending team regains possession within 5 s   |   "
        "Good = Ball out of play or opponent fouls within 15 s   |   "
        "Okay = Defending team fouls or attack delayed past 15 s   |   "
        "Bad = Shot, penetration or in-behind pass conceded within 15 s"
    )
    fig.text(0.5, -0.02, glossary, ha="center", va="top", fontsize=7,
             color="#555", wrap=True, style="italic")

    # Legend panel
    ax_leg.axis("off")
    ax_leg.legend(
        handles=handles[::-1],
        loc="center",
        fontsize=9,
        frameon=False,
        title="Rating",
        title_fontsize=9,
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig
