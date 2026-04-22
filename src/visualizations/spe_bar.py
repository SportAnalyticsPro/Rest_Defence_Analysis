"""
spe_bar.py
----------
Horizontal bar plots showing Structural Prevention Efficiency (SPE) metrics.
Two subplots: SPE (15s) and SPE (20s) side by side.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualizations import PRIMARY_RED, PRIMARY_ORANGE, EXCLUDED_TEAMS, spe_from_csv
from src.logos import get_logo_image


def _plot_spe_bars_single(
    df: pd.DataFrame,
    spe_key: str,
    title: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Helper function to plot a single SPE bar chart.
    spe_key: "spe_15" or "spe_20"
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    df_filtered = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]
    teams = sorted(df_filtered["losing_team_name"].unique())

    # Compute SPE for each team
    spe_data = {}
    for team in teams:
        spe_15, spe_20 = spe_from_csv(df_filtered, team)
        spe_data[team] = {"spe_15": spe_15, "spe_20": spe_20}

    # Create DataFrame and sort by the selected SPE metric descending
    spe_df = pd.DataFrame(spe_data).T
    spe_df = spe_df.sort_values(spe_key, ascending=True)  # ascending for horizontal bars

    teams = spe_df.index.tolist()
    spe_vals = spe_df[spe_key].values

    n = len(teams)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.55)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Create horizontal bars
    bars = ax.barh(y_pos, spe_vals, color=PRIMARY_ORANGE, height=0.65, linewidth=0)

    # Add percentage labels INSIDE bars on the LEFT (white text)
    for i, (team, val) in enumerate(zip(teams, spe_vals)):
        if not pd.isna(val):
            ax.text(val * 0.05, i, f"{val:.1f}%",
                    ha="left", va="center",
                    fontsize=10, fontweight="bold", color="white")

    # Add logos at the END of each bar (center of logo aligns with bar end)
    for i, (team, val) in enumerate(zip(teams, spe_vals)):
        logo = get_logo_image(team, size=256)
        if logo is not None and not pd.isna(val):
            img = OffsetImage(logo, zoom=0.12)
            ab = AnnotationBbox(
                img, (val, i),
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                zorder=3,
            )
            ax.add_artist(ab)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(teams, fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("SPE (%)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=PRIMARY_RED, pad=10)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)

    # Add SPE definition below the chart (two lines)
    spe_line1 = (
        "SPE = % of transitions where the defending team prevents the opponent from reaching the defensive third "
        "within the given time window after losing possession."
    )
    spe_line2 = "Higher SPE indicates stronger defensive structure."

    fig.text(0.5, -0.01, spe_line1, ha="center", va="top", fontsize=9.5, color="#333", style="italic", wrap=True)
    fig.text(0.5, -0.05, spe_line2, ha="center", va="top", fontsize=9.5, color="#333", style="italic")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig


def plot_spe_bars(
    df: pd.DataFrame,
    output_path_15: str | None = None,
    output_path_20: str | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Generate two separate SPE bar charts: one for 15s, one for 20s.
    Returns both figures as a tuple.
    """
    fig_15 = _plot_spe_bars_single(df, "spe_15", "Structural Prevention Efficiency (15s Window)", output_path_15)
    fig_20 = _plot_spe_bars_single(df, "spe_20", "Structural Prevention Efficiency (20s Window)", output_path_20)
    return fig_15, fig_20
