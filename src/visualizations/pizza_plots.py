"""
pizza_plots.py
--------------
Pizza charts using mplsoccer.PyPizza with percentile rankings (league-based).
Three separate pizzas per team: attacking, defending, foul analysis.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from mplsoccer import PyPizza

from src.visualizations import EXCLUDED_TEAMS, spe_from_csv, PRIMARY_RED, PRIMARY_ORANGE, PRIMARY_YELLOW
from src.logos import get_logo_image


# ---------------------------------------------------------------------------
# Percentile ranking helper
# ---------------------------------------------------------------------------

def _percentile_rank(values: np.ndarray, lower_is_better: bool = False) -> np.ndarray:
    """
    Rank values as percentiles (0-100) across the league.
    NaN values get 50 (league median).
    Returns integers (no decimals).
    """
    arr = np.array(values, dtype=float)
    valid = ~np.isnan(arr)
    result = np.full_like(arr, 50.0, dtype=float)

    if lower_is_better:
        arr_to_rank = -arr[valid]
    else:
        arr_to_rank = arr[valid]

    if len(arr_to_rank) > 0:
        ranks = rankdata(arr_to_rank, method="average")
        result[valid] = (ranks / valid.sum()) * 100

    return np.round(result).astype(int)


# ---------------------------------------------------------------------------
# Pizza 1 — Defending Transition (5 slices)
# ---------------------------------------------------------------------------

def plot_pizza_defending(
    df: pd.DataFrame,
    team: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    5-slice defending pizza for one team (percentile-based, no comparisons).
    """
    df = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]

    params = [
        "SPE 20s",
        "% Best",
        "Num. Superiority",
        "Recovery Speed",
        "Compactness",
    ]

    team_vals = _team_pizza_values_defending(df, team)

    fig, ax = plt.subplots(figsize=(9, 10), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#FAFAFA")

    pizza = PyPizza(
        params=params,
        straight_line_color="#F2F2F2",
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    pizza.make_pizza(
        team_vals,
        ax=ax,
        color_blank_space="same",
        blank_alpha=0.4,
        kwargs_slices=dict(
            facecolor=PRIMARY_YELLOW, edgecolor="#F2F2F2",
            zorder=2, linewidth=1,
        ),
        kwargs_params=dict(
            color="#000000", fontsize=11, va="center", fontweight="bold",
        ),
        kwargs_values=dict(
            color="#000000", fontsize=10, fontweight="bold",
            bbox=dict(
                edgecolor="#000000", facecolor=PRIMARY_YELLOW,
                boxstyle="round,pad=0.3", lw=0.5, alpha=0.9,
            ),
        ),
    )

    # Title + subtitle
    fig.text(
        0.5, 0.98, f"{team} — Defending Transitions",
        fontsize=14, fontweight="bold", color=PRIMARY_RED, ha="center", va="top",
    )
    fig.text(
        0.5, 0.95,
        "Percentile Ranking vs League (0–100)",
        fontsize=10, color="#666666", ha="center", va="top", style="italic",
    )

    # Definitions at bottom
    definitions = (
        "SPE 20s: % transitions where defending team prevents opponent reaching defensive third within 20s  |  "
        "% Best: % of transitions rated as Best  |  Num. Superiority: average numerical advantage at start  |  "
        "Recovery Speed: avg centroid advance in first 5s  |  Compactness: team spacing tightness (lower is better)"
    )
    fig.text(
        0.5, 0.02, definitions,
        fontsize=7, color="#666666", ha="center", va="bottom", wrap=True, style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    # Logo in centre
    logo = get_logo_image(team, size=128)
    if logo is not None:
        ax_pos = ax.get_position()
        logo_w, logo_h = 0.10, 0.12
        logo_ax = fig.add_axes([
            ax_pos.x0 + ax_pos.width * 0.5 - logo_w / 2,
            ax_pos.y0 + ax_pos.height * 0.5 - logo_h / 2,
            logo_w, logo_h,
        ])
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Pizza 2 — Attacking Transition (5 slices)
# ---------------------------------------------------------------------------

def plot_pizza_attacking(
    df: pd.DataFrame,
    team: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    5-slice attacking pizza for one team (percentile-based, no comparisons).
    """
    df = df[~df["gaining_team_name"].isin(EXCLUDED_TEAMS)]

    params = [
        "Constructive\nProgression",
        "Own Half Exit",
        "Forward Pass\n45°",
        "Playmaker\nIndependence",
        "Transition Speed",
    ]

    team_vals = _team_pizza_values_attacking(df, team)

    fig, ax = plt.subplots(figsize=(9, 10), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#FAFAFA")

    pizza = PyPizza(
        params=params,
        straight_line_color="#F2F2F2",
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    pizza.make_pizza(
        team_vals,
        ax=ax,
        color_blank_space="same",
        blank_alpha=0.4,
        kwargs_slices=dict(
            facecolor=PRIMARY_ORANGE, edgecolor="#F2F2F2",
            zorder=2, linewidth=1,
        ),
        kwargs_params=dict(
            color="#000000", fontsize=11, va="center", fontweight="bold",
        ),
        kwargs_values=dict(
            color="#000000", fontsize=10, fontweight="bold",
            bbox=dict(
                edgecolor="#000000", facecolor=PRIMARY_ORANGE,
                boxstyle="round,pad=0.3", lw=0.5, alpha=0.9,
            ),
        ),
    )

    # Title + subtitle
    fig.text(
        0.5, 0.98, f"{team} — Attacking Transitions",
        fontsize=14, fontweight="bold", color=PRIMARY_RED, ha="center", va="top",
    )
    fig.text(
        0.5, 0.95,
        "Percentile Ranking vs League (0–100)",
        fontsize=10, color="#666666", ha="center", va="top", style="italic",
    )

    # Definitions at bottom
    definitions = (
        "Constructive Progression: % transitions with forward progress  |  "
        "Own Half Exit: % transitions exiting own half  |  Forward Pass 45°: % forward passes within 45° cone  |  "
        "Playmaker Independence: % transitions not dependent on playmaker  |  "
        "Transition Speed: avg duration (lower is faster)"
    )
    fig.text(
        0.5, 0.02, definitions,
        fontsize=7, color="#666666", ha="center", va="bottom", wrap=True, style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    # Logo in centre
    logo = get_logo_image(team, size=128)
    if logo is not None:
        ax_pos = ax.get_position()
        logo_w, logo_h = 0.10, 0.12
        logo_ax = fig.add_axes([
            ax_pos.x0 + ax_pos.width * 0.5 - logo_w / 2,
            ax_pos.y0 + ax_pos.height * 0.5 - logo_h / 2,
            logo_w, logo_h,
        ])
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Pizza 3 — Foul Analysis (5 slices)
# ---------------------------------------------------------------------------

def plot_pizza_foul(
    df: pd.DataFrame,
    team: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    5-slice foul analysis pizza for one team (percentile-based, no comparisons).
    """
    df = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]

    params = [
        "Foul Rate",
        "Time to Foul",
        "Foul Location",
        "Smart Fouls",
        "Bad Fouls",
    ]

    team_vals = _team_pizza_values_foul(df, team)

    fig, ax = plt.subplots(figsize=(9, 10), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#FAFAFA")

    pizza = PyPizza(
        params=params,
        straight_line_color="#F2F2F2",
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    pizza.make_pizza(
        team_vals,
        ax=ax,
        color_blank_space="same",
        blank_alpha=0.4,
        kwargs_slices=dict(
            facecolor=PRIMARY_RED, edgecolor="#F2F2F2",
            zorder=2, linewidth=1,
        ),
        kwargs_params=dict(
            color="#000000", fontsize=11, va="center", fontweight="bold",
        ),
        kwargs_values=dict(
            color="#FFFFFF", fontsize=10, fontweight="bold",
            bbox=dict(
                edgecolor="#000000", facecolor=PRIMARY_RED,
                boxstyle="round,pad=0.3", lw=0.5, alpha=0.9,
            ),
        ),
    )

    # Title + subtitle
    fig.text(
        0.5, 0.98, f"{team} — Foul Analysis",
        fontsize=14, fontweight="bold", color=PRIMARY_RED, ha="center", va="top",
    )
    fig.text(
        0.5, 0.95,
        "Percentile Ranking vs League (0–100)",
        fontsize=10, color="#666666", ha="center", va="top", style="italic",
    )

    # Definitions at bottom
    definitions = (
        "Foul Rate: % of transitions with a foul committed  |  "
        "Time to Foul: avg seconds until foul occurs (lower is faster)  |  "
        "Foul Location: avg x-position of foul (further up = higher press)  |  "
        "Smart Fouls: % of fouls rated as Okay  |  "
        "Bad Fouls: % of fouls rated as Bad (lower is better)"
    )
    fig.text(
        0.5, 0.02, definitions,
        fontsize=7, color="#666666", ha="center", va="bottom", wrap=True, style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    # Logo in centre
    logo = get_logo_image(team, size=128)
    if logo is not None:
        ax_pos = ax.get_position()
        logo_w, logo_h = 0.10, 0.12
        logo_ax = fig.add_axes([
            ax_pos.x0 + ax_pos.width * 0.5 - logo_w / 2,
            ax_pos.y0 + ax_pos.height * 0.5 - logo_h / 2,
            logo_w, logo_h,
        ])
        logo_ax.imshow(logo)
        logo_ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Helpers: compute team percentile values
# ---------------------------------------------------------------------------

def _team_pizza_values_defending(df: pd.DataFrame, team: str) -> list[int]:
    """Compute 5 percentile values for defending (all integers 0-100)."""
    tdf = df[df["losing_team_name"] == team]
    all_df = df

    # Raw values
    _, spe_20 = spe_from_csv(all_df, team)
    pct_best = (tdf["transition_rating"] == "Best").sum() / len(tdf) * 100 if len(tdf) else float("nan")
    num_sup = tdf["num_superiority_app1_t0"].mean() if len(tdf) else float("nan")
    recovery = tdf["centroid_advance_5s_m"].mean() if len(tdf) else float("nan")
    compactness = tdf["team_compactness_t0"].mean() if len(tdf) else float("nan")

    # All teams for ranking
    all_teams = list((set(all_df["losing_team_name"].unique())) - EXCLUDED_TEAMS)

    spe_vals = [spe_from_csv(all_df, t)[1] for t in all_teams]
    pct_best_vals = [(all_df[all_df["losing_team_name"] == t]["transition_rating"] == "Best").sum() / len(all_df[all_df["losing_team_name"] == t]) * 100
                     if len(all_df[all_df["losing_team_name"] == t]) else float("nan") for t in all_teams]
    num_sup_vals = [all_df[all_df["losing_team_name"] == t]["num_superiority_app1_t0"].mean() for t in all_teams]
    recovery_vals = [all_df[all_df["losing_team_name"] == t]["centroid_advance_5s_m"].mean() for t in all_teams]
    compactness_vals = [all_df[all_df["losing_team_name"] == t]["team_compactness_t0"].mean() for t in all_teams]

    # Percentiles
    spe_pct = _percentile_rank(np.array(spe_vals))
    pct_best_pct = _percentile_rank(np.array(pct_best_vals))
    num_sup_pct = _percentile_rank(np.array(num_sup_vals))
    recovery_pct = _percentile_rank(np.array(recovery_vals))
    compactness_pct = _percentile_rank(np.array(compactness_vals), lower_is_better=True)

    team_idx = all_teams.index(team)

    return [
        int(spe_pct[team_idx]),
        int(pct_best_pct[team_idx]),
        int(num_sup_pct[team_idx]),
        int(recovery_pct[team_idx]),
        int(compactness_pct[team_idx]),
    ]


def _team_pizza_values_attacking(df: pd.DataFrame, team: str) -> list[int]:
    """Compute 5 percentile values for attacking (all integers 0-100)."""
    gdf = df[df["gaining_team_name"] == team]
    all_df = df

    # Raw values
    prog = gdf["constructive_progression"].astype(float).mean() * 100 if len(gdf) else float("nan")
    own_exit = gdf["own_half_exit"].astype(float).mean() * 100 if len(gdf) else float("nan")
    pass_45 = gdf["productive_pass_ratio_45"].mean() * 100 if len(gdf) else float("nan")
    playmaker_indep = (1 - gdf["playmaker_dependency_1st"].astype(float).mean()) * 100 if len(gdf) else float("nan")
    duration = gdf["duration_s"].mean() if len(gdf) else float("nan")

    # All teams for ranking
    all_teams = list((set(all_df["gaining_team_name"].unique())) - EXCLUDED_TEAMS)

    prog_vals = [all_df[all_df["gaining_team_name"] == t]["constructive_progression"].astype(float).mean() * 100 for t in all_teams]
    own_exit_vals = [all_df[all_df["gaining_team_name"] == t]["own_half_exit"].astype(float).mean() * 100 for t in all_teams]
    pass_45_vals = [all_df[all_df["gaining_team_name"] == t]["productive_pass_ratio_45"].mean() * 100 for t in all_teams]
    playmaker_indep_vals = [(1 - all_df[all_df["gaining_team_name"] == t]["playmaker_dependency_1st"].astype(float).mean()) * 100 for t in all_teams]
    duration_vals = [all_df[all_df["gaining_team_name"] == t]["duration_s"].mean() for t in all_teams]

    # Percentiles
    prog_pct = _percentile_rank(np.array(prog_vals))
    own_exit_pct = _percentile_rank(np.array(own_exit_vals))
    pass_45_pct = _percentile_rank(np.array(pass_45_vals))
    playmaker_indep_pct = _percentile_rank(np.array(playmaker_indep_vals))
    duration_pct = _percentile_rank(np.array(duration_vals), lower_is_better=True)

    team_idx = all_teams.index(team)

    return [
        int(prog_pct[team_idx]),
        int(own_exit_pct[team_idx]),
        int(pass_45_pct[team_idx]),
        int(playmaker_indep_pct[team_idx]),
        int(duration_pct[team_idx]),
    ]


def _team_pizza_values_foul(df: pd.DataFrame, team: str) -> list[int]:
    """Compute 5 percentile values for foul analysis (all integers 0-100)."""
    tdf = df[df["losing_team_name"] == team]
    foul_df = tdf[tdf["foul_committed"].astype(bool)]
    n = len(tdf)
    n_f = len(foul_df)

    foul_rate = n_f / n * 100 if n else float("nan")
    time_to_foul = foul_df["foul_time_s"].mean() if n_f else float("nan")
    foul_loc = foul_df["foul_x_m"].mean() if n_f else float("nan")
    smart_fouls = (foul_df["foul_superiority_rating"] == "Okay").sum() / n_f * 100 if n_f else float("nan")
    bad_fouls = (foul_df["foul_superiority_rating"] == "Bad").sum() / n_f * 100 if n_f else float("nan")

    # All teams for ranking
    all_teams = list(set(df["losing_team_name"].unique()) - EXCLUDED_TEAMS)

    foul_rate_vals = []
    time_vals = []
    loc_vals = []
    smart_vals = []
    bad_vals = []

    for t in all_teams:
        ttdf = df[df["losing_team_name"] == t]
        tfoul = ttdf[ttdf["foul_committed"].astype(bool)]
        tn = len(ttdf)
        tnf = len(tfoul)

        foul_rate_vals.append(tnf / tn * 100 if tn else float("nan"))
        time_vals.append(tfoul["foul_time_s"].mean() if tnf else float("nan"))
        loc_vals.append(tfoul["foul_x_m"].mean() if tnf else float("nan"))
        smart_vals.append((tfoul["foul_superiority_rating"] == "Okay").sum() / tnf * 100 if tnf else float("nan"))
        bad_vals.append((tfoul["foul_superiority_rating"] == "Bad").sum() / tnf * 100 if tnf else float("nan"))

    # Percentiles
    foul_rate_pct = _percentile_rank(np.array(foul_rate_vals))
    time_pct = _percentile_rank(np.array(time_vals), lower_is_better=True)
    loc_pct = _percentile_rank(np.array(loc_vals))
    smart_pct = _percentile_rank(np.array(smart_vals))
    bad_pct = _percentile_rank(np.array(bad_vals), lower_is_better=True)

    team_idx = all_teams.index(team)

    return [
        int(foul_rate_pct[team_idx]),
        int(time_pct[team_idx]),
        int(loc_pct[team_idx]),
        int(smart_pct[team_idx]),
        int(bad_pct[team_idx]),
    ]
