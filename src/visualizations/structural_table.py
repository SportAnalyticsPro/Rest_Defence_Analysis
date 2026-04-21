"""
structural_table.py
-------------------
Styled matplotlib tables with per-column gradient cell colouring.
Green = better value, Red = worse value (direction specified per column).
"""
from __future__ import annotations
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from src.visualizations import BRAND_PALETTE, HEADER_TEXT, EXCLUDED_TEAMS, spe_from_csv, PRIMARY_RED

# Red → White → Green (white midpoint for neutral/middle values)
_RG = mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", ["#E22C0F", "#FFFFFF", "#90EE90"])
_HEADER_BG = "#FFFFFF"   # white headers


# ---------------------------------------------------------------------------
# Generic table renderer
# ---------------------------------------------------------------------------

def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _cell_text_colour(bg_hex: str) -> str:
    return "#111111" if _luminance(bg_hex) > 0.45 else "#FFFFFF"


def _wrap(text: str, width: int = 22) -> str:
    """Wrap column header to at most 4 lines."""
    lines = textwrap.wrap(text, width)
    return "\n".join(lines[:4])


def _render_styled_table(
    team_data: dict[str, list],
    col_config: list[dict],
    title: str,
    explanation: str,
    sort_col_idx: int = 0,
    output_path: str | None = None,
) -> plt.Figure:
    """
    col_config keys:
      name (str)  — column header
      fmt  (str)  — format string e.g. '.1f'
      hib  (bool) — higher_is_better (green=high if True, green=low if False)
    """
    teams = sorted(team_data.keys())
    hib_sort = col_config[sort_col_idx].get("hib", True)
    teams = sorted(
        teams,
        key=lambda t: team_data[t][sort_col_idx]
            if not np.isnan(team_data[t][sort_col_idx]) else -np.inf,
        reverse=hib_sort,
    )

    n_rows = len(teams)
    n_cols = len(col_config)

    cell_text = []
    cell_vals = []
    for team in teams:
        row_text, row_vals = [], []
        for i, cfg in enumerate(col_config):
            v = team_data[team][i]
            row_text.append("—" if np.isnan(v) else f"{v:{cfg['fmt']}}")
            row_vals.append(v)
        cell_text.append(row_text)
        cell_vals.append(row_vals)

    col_arrays = np.array([[r[i] for r in cell_vals] for i in range(n_cols)], dtype=float)

    cell_colours = []
    for r_idx in range(n_rows):
        row_colours = []
        for c_idx in range(n_cols):
            cfg = col_config[c_idx]
            col = col_arrays[c_idx]
            valid = col[~np.isnan(col)]
            v = col_arrays[c_idx, r_idx]
            if len(valid) < 2 or np.isnan(v):
                row_colours.append("#E8E8E8")
                continue
            vmin, vmax = valid.min(), valid.max()
            if vmin == vmax:
                row_colours.append("#E8E8E8")
                continue
            norm = (v - vmin) / (vmax - vmin)
            rgba = _RG(norm) if cfg.get("hib", True) else _RG(1 - norm)
            row_colours.append(mcolors.to_hex(rgba))
        cell_colours.append(row_colours)

    # --- Figure ---
    row_h  = 0.40   # inches per data row
    hdr_h  = 1.2    # inches for header row (4 lines of wrapped text)
    col_w  = max(1.55, 11.0 / n_cols)
    row_lbl_w = 1.4
    fig_w  = col_w * n_cols + row_lbl_w
    fig_h  = row_h * n_rows + hdr_h + 1.0   # +1 for title + explanation

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FAFAFA")

    # Title at top, centred
    fig.text(
        0.5, 0.97, title,
        fontsize=12, fontweight="bold",
        color=PRIMARY_RED,
        ha="center", va="top",
    )

    # Explanation at very bottom
    fig.text(
        0.5, 0.01, explanation,
        fontsize=7, color="#666666", style="italic",
        ha="center", va="bottom", wrap=True,
    )

    # Axes fills the space between title and explanation
    ax = fig.add_axes([0.0, 0.08, 1.0, 0.86])
    ax.axis("off")

    headers = [_wrap(c["name"]) for c in col_config]

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=teams,
        colLabels=headers,
        cellColours=cell_colours,
        cellLoc="center",
        bbox=[0, 0, 1, 1],   # fill entire axes — eliminates whitespace
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Header row (white background with dark text)
    for c_idx in range(n_cols):
        cell = tbl[0, c_idx]
        cell.set_facecolor(_HEADER_BG)
        cell.set_text_props(color="#333333", fontweight="bold", fontsize=8)
        cell.set_edgecolor("#CCCCCC")

    # Row labels (white background with dark text)
    for r_idx in range(n_rows):
        cell = tbl[r_idx + 1, -1]
        cell.set_facecolor(_HEADER_BG)
        cell.set_text_props(color="#333333", fontweight="bold", fontsize=9)
        cell.set_edgecolor("#CCCCCC")

    # Data cells
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            cell = tbl[r_idx + 1, c_idx]
            bg = cell_colours[r_idx][c_idx]
            cell.set_text_props(color=_cell_text_colour(bg), fontsize=9)
            cell.set_edgecolor("#CCCCCC")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"    Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Helper: compute team stats dict
# ---------------------------------------------------------------------------

def _build_team_rows(df: pd.DataFrame, extractors: list[callable]) -> dict[str, list]:
    """
    extractors: list of functions f(tdf, gdf, all_df) → float
    where tdf = defending transitions, gdf = attacking transitions.
    """
    df = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]
    all_teams = set(df["losing_team_name"].unique()) | set(df["gaining_team_name"].unique())
    all_teams -= EXCLUDED_TEAMS
    result = {}
    for team in sorted(all_teams):
        tdf = df[df["losing_team_name"] == team]
        gdf = df[df["gaining_team_name"] == team]
        result[team] = [fn(tdf, gdf, df) for fn in extractors]
    return result


# ---------------------------------------------------------------------------
# Table 1 — Team Structure at Ball Lost
# ---------------------------------------------------------------------------

def plot_team_structure(df: pd.DataFrame, output_path: str | None = None) -> plt.Figure:
    def cm(col): return lambda tdf, g, a: tdf[col].mean() if len(tdf) else float("nan")

    rows = _build_team_rows(df, [
        cm("team_length_m_t0"), cm("team_compactness_t0"),
        cm("line_height_m_t0"), cm("players_behind_ball_t0"),
        cm("num_superiority_app1_t0"), cm("num_superiority_app2_t0"),
    ])
    cols = [
        {"name": "Team Length (m)",                           "fmt": ".2f", "hib": False},
        {"name": "Team Compactness (m)",                      "fmt": ".2f", "hib": False},
        {"name": "Defensive Line Height (m)",                 "fmt": ".2f", "hib": True,  "neutral": True},
        {"name": "Players Behind Ball",                       "fmt": ".1f", "hib": True},
        {"name": "Numerical Superiority App1 Rule-Based",     "fmt": ".1f", "hib": True},
        {"name": "Numerical Superiority App2 Clustering",     "fmt": ".1f", "hib": True},
    ]
    expl = (
        "Structural metrics measured at the moment of possession loss. Team Length = distance from rearmost to foremost player. "
        "Compactness = mean distance from team centre. Line Height = rearmost defensive cluster distance from own goal."
    )
    return _render_styled_table(rows, cols,
        title="Team Structure at Ball Lost", explanation=expl,
        sort_col_idx=4, output_path=output_path)


# ---------------------------------------------------------------------------
# Table 2 — Transition Dynamics (Positive) — Deltas
# ---------------------------------------------------------------------------

def plot_transition_dynamics_delta(df: pd.DataFrame, output_path: str | None = None) -> plt.Figure:
    def cm(col): return lambda tdf, g, a: tdf[col].mean() if len(tdf) else float("nan")
    def delta(c_end, c_start):
        return lambda tdf, g, a: (tdf[c_end] - tdf[c_start]).mean() if len(tdf) else float("nan")

    rows = _build_team_rows(df, [
        cm("centroid_advance_5s_m"), cm("centroid_advance_10s_m"),
        delta("team_length_m_t10", "team_length_m_t0"),
        delta("team_length_m_t20", "team_length_m_t0"),
        delta("team_compactness_t10", "team_compactness_t0"),
        delta("team_compactness_t20", "team_compactness_t0"),
    ])
    cols = [
        {"name": "Centroid Advance after 5s from ball lost (m)",       "fmt": ".2f", "hib": True},
        {"name": "Centroid Advance after 10s from ball lost (m)",      "fmt": ".2f", "hib": True},
        {"name": "Team Length Change after 5s from ball lost (m)",     "fmt": ".2f", "hib": False},
        {"name": "Team Length Change after 10s from ball lost (m)",    "fmt": ".2f", "hib": False},
        {"name": "Compactness Change after 5s from ball lost (m)",     "fmt": ".2f", "hib": False},
        {"name": "Compactness Change after 10s from ball lost (m)",    "fmt": ".2f", "hib": False},
    ]
    expl = (
        "Defensive recovery shape changes after possession loss. Centroid Advance = forward movement toward opponent goal. "
        "Length/Compactness Change = relative to moment of loss (negative = tightening)."
    )
    return _render_styled_table(rows, cols,
        title="Transition Dynamics — Defensive Recovery (Changes)", explanation=expl,
        sort_col_idx=0, output_path=output_path)


# ---------------------------------------------------------------------------
# Table 3 — Transition Dynamics (Positive) — Absolute values at 5s & 10s
# ---------------------------------------------------------------------------

def plot_transition_dynamics_absolute(df: pd.DataFrame, output_path: str | None = None) -> plt.Figure:
    def cm(col): return lambda tdf, g, a: tdf[col].mean() if len(tdf) else float("nan")

    rows = _build_team_rows(df, [
        cm("centroid_advance_5s_m"), cm("centroid_advance_10s_m"),
        cm("team_length_m_t10"), cm("team_length_m_t20"),
        cm("team_compactness_t10"), cm("team_compactness_t20"),
    ])
    cols = [
        {"name": "Centroid Advance after 5s from ball lost (m)",   "fmt": ".2f", "hib": True},
        {"name": "Centroid Advance after 10s from ball lost (m)",  "fmt": ".2f", "hib": True},
        {"name": "Team Length after 5s from ball lost (m)",        "fmt": ".2f", "hib": False},
        {"name": "Team Length after 10s from ball lost (m)",       "fmt": ".2f", "hib": False},
        {"name": "Team Compactness after 5s from ball lost (m)",   "fmt": ".2f", "hib": False},
        {"name": "Team Compactness after 10s from ball lost (m)",  "fmt": ".2f", "hib": False},
    ]
    expl = (
        "Defensive structure shape at 5s and 10s after possession loss. Centroid Advance = forward movement toward opponent goal. "
        "Lower Length/Compactness values = more compact, organised defensive block."
    )
    return _render_styled_table(rows, cols,
        title="Transition Dynamics — Defensive Recovery (Absolute Values)", explanation=expl,
        sort_col_idx=0, output_path=output_path)


# ---------------------------------------------------------------------------
# Table 4 — Attacking Transitions Quality
# ---------------------------------------------------------------------------

def plot_attacking_transitions(df: pd.DataFrame, output_path: str | None = None) -> plt.Figure:
    def pct_bool(col):
        return lambda tdf, gdf, a: gdf[col].astype(float).mean() * 100 if len(gdf) else float("nan")
    def pct_ratio(col):
        return lambda tdf, gdf, a: gdf[col].mean() * 100 if len(gdf) else float("nan")
    def pct_playmaker_either_pass():
        """Playmaker involved in 1st OR 2nd pass (union of dependencies)."""
        return lambda tdf, gdf, a: (
            (gdf["playmaker_dependency_1st"].astype(bool) | gdf["playmaker_dependency_2nd"].astype(bool))
            .mean() * 100 if len(gdf) else float("nan")
        )

    rows = _build_team_rows(df, [
        pct_bool("constructive_progression"),
        pct_bool("own_half_exit"),
        pct_ratio("productive_pass_ratio_45"),
        pct_ratio("productive_pass_ratio_90"),
        pct_bool("playmaker_dependency_1st"),
        pct_playmaker_either_pass(),
    ])
    cols = [
        {"name": "Constructive Progression (%)",        "fmt": ".1f", "hib": True},
        {"name": "Own Half Exit (%)",                   "fmt": ".1f", "hib": True},
        {"name": "Forward Pass Ratio 45° (%)",          "fmt": ".1f", "hib": True},
        {"name": "Forward Pass Ratio 90° (%)",          "fmt": ".1f", "hib": True},
        {"name": "Deep-Lying Playmaker Dependency 1st Pass (%)",   "fmt": ".1f", "hib": True},
        {"name": "Deep-Lying Playmaker Dependency in First 2 Passes (%)",   "fmt": ".1f", "hib": True},
    ]
    expl = (
        "Team attacking quality when gaining possession. Constructive Progression = ≥3 passes within 15s. "
        "Own Half Exit = advanced beyond own half. Forward Pass Ratio = % forward passes. "
        "Deep-Lying Playmaker Dependency = reliance on this key playmaker in 1st pass or in first 2 passes."
    )
    return _render_styled_table(rows, cols,
        title="Attacking Transitions — Team Quality", explanation=expl,
        sort_col_idx=0, output_path=output_path)


# ---------------------------------------------------------------------------
# Table 5 — Foul Analysis
# ---------------------------------------------------------------------------

def plot_foul_table(df: pd.DataFrame, output_path: str | None = None) -> plt.Figure:
    df_clean = df[~df["losing_team_name"].isin(EXCLUDED_TEAMS)]
    all_teams = set(df_clean["losing_team_name"].unique()) - EXCLUDED_TEAMS

    rows = {}
    for team in sorted(all_teams):
        tdf     = df_clean[df_clean["losing_team_name"] == team]
        foul_df = tdf[tdf["foul_committed"].astype(bool)]
        n       = len(tdf)
        n_f     = len(foul_df)
        foul_rate   = n_f / n * 100 if n else float("nan")

        # Skip teams with no fouls
        if foul_rate == 0.0:
            continue

        avg_time    = foul_df["foul_time_s"].mean() if n_f else float("nan")
        avg_loc     = foul_df["foul_x_m"].mean()    if n_f else float("nan")
        bad_pct     = (foul_df["foul_superiority_rating"] == "Bad").sum()  / n_f * 100 if n_f else float("nan")
        okay_pct    = (foul_df["foul_superiority_rating"] == "Okay").sum() / n_f * 100 if n_f else float("nan")
        rows[team] = [foul_rate, avg_time, avg_loc, bad_pct, okay_pct]

    cols = [
        {"name": "Foul Interruption Rate (%)",               "fmt": ".1f", "hib": True,  "neutral": True},
        {"name": "Avg Time to Foul (s)",                     "fmt": ".1f", "hib": False},
        {"name": "Avg Foul Location — Distance from Own Goal (m)", "fmt": ".1f", "hib": True},
        {"name": "Fouls in Numerical Superiority — Bad (%)", "fmt": ".0f", "hib": False},
        {"name": "Fouls in Equality / Inferiority — Okay (%)","fmt": ".0f", "hib": True},
    ]
    expl = (
        "Defending team's foul patterns when committing fouls. Rate = % of transitions interrupted by foul. "
        "Bad fouls = committed in numerical superiority (proactive). Okay fouls = equality/inferiority (tactical save)."
    )
    return _render_styled_table(rows, cols,
        title="Foul Analysis", explanation=expl,
        sort_col_idx=0, output_path=output_path)
