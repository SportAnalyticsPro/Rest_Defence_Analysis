"""
src/visualisations/__init__.py
------------------------------
Shared constants and utilities for the visualization suite.
All charts read from the all_transitions.csv produced by main.py.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Brand palette & primary colors
# ---------------------------------------------------------------------------

BRAND_PALETTE = ["#E22C0F", "#FEBF10", "#602100", "#800080"]  # Red, Yellow, Brown, Purple

# Primary colors used consistently across all visualizations
PRIMARY_RED    = "#BE0000"
PRIMARY_ORANGE = "#FFA100"
PRIMARY_YELLOW = "#FEBF10"

RATING_COLOURS = {
    "Best": "#2E7D32",
    "Good": "#81C784",
    "Okay": "#FEBF10",
    "Bad":  "#E22C0F",
}

HEADER_COLOUR  = "#602100"   # brown — used for table headers
HEADER_TEXT    = "#FFFFFF"

# Teams excluded from all visualisations (incomplete data or non-league teams)
EXCLUDED_TEAMS = {"Borussia Dortmund"}


def get_brand_colour(idx: int) -> str:
    """Return brand colour at position idx, cycling through the palette."""
    return BRAND_PALETTE[idx % len(BRAND_PALETTE)]


# ---------------------------------------------------------------------------
# SPE helper (mirrors report_generator._spe_from_csv, avoids import cycle)
# ---------------------------------------------------------------------------

def spe_from_csv(df: pd.DataFrame, team_name: str) -> tuple[float, float]:
    tdf = df[df["losing_team_name"] == team_name]
    v15 = tdf[tdf["has_15s_window"].astype(bool)]
    v20 = tdf[tdf["has_20s_window"].astype(bool)]
    spe_15 = (1.0 - v15["ball_reached_third_15s"].astype(float).mean()) * 100
    spe_20 = (1.0 - v20["ball_reached_third_20s"].astype(float).mean()) * 100
    return spe_15, spe_20


# ---------------------------------------------------------------------------
# Chart orchestrator
# ---------------------------------------------------------------------------

def generate_all_charts(df: pd.DataFrame, charts_dir: Path) -> None:
    """Generate all 6 chart types and save to charts_dir."""
    import matplotlib
    matplotlib.use("Agg")

    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    from src.visualizations.ratings_bar      import plot_ratings_bar
    from src.visualizations.line_plots        import plot_metric_evolution
    from src.visualizations.scatterplots      import plot_compactness_vs_length, plot_lineheight_vs_cadv
    from src.visualizations.structural_table  import plot_structural_table
    from src.visualizations.spider_plots      import plot_spider_absolute, plot_spider_single_team
    from src.visualizations.pitch_block_plot  import plot_pitch_block
    from src.visualizations.pizza_plots       import (
        plot_pizza_defending, plot_pizza_attacking, plot_pizza_foul,
    )

    print("  Generating Chart 1 — Ratings bar ...")
    plot_ratings_bar(df, output_path=str(charts_dir / "ratings_bar.png"))

    print("  Generating Chart 2 — Evolution line plots ...")
    for metric in ("team_length_m", "team_compactness", "team_centroid_x_norm"):
        plot_metric_evolution(df, metric, output_path=str(charts_dir / f"evolution_{metric}.png"))

    print("  Generating Chart 3 — Scatterplots ...")
    plot_compactness_vs_length(df, output_path=str(charts_dir / "scatter_compactness_length.png"))
    plot_lineheight_vs_cadv(df, output_path=str(charts_dir / "scatter_lineheight_cadv.png"))

    print("  Generating Chart 4 — Structural table ...")
    plot_structural_table(df, output_path=str(charts_dir / "structural_table.png"))

    print("  Generating Chart 5 — Radar plots ...")
    plot_spider_absolute(df, output_path=str(charts_dir / "radar_absolute.png"))

    # Single-team radar plots with league mean comparison
    for team in sorted(df["gaining_team_name"].unique()):
        if team not in EXCLUDED_TEAMS:
            safe = team.lower().replace(" ", "_")
            plot_spider_single_team(df, team, output_path=str(charts_dir / f"radar_single_{safe}.png"))

    print("  Generating Chart 6 — Pitch block plots ...")
    for team in sorted(df["losing_team_name"].unique()):
        safe = team.lower().replace(" ", "_")
        plot_pitch_block(df, team, output_path=str(charts_dir / f"pitch_block_{safe}.png"))

    print("  Generating Chart 7 — Pizza plots (defending & attacking transitions) ...")
    test_teams = ["Como", "Hellas Verona", "Juventus"]
    for team in test_teams:
        safe = team.lower().replace(" ", "_")
        plot_pizza_defending(df, team, output_path=str(charts_dir / f"pizza_defending_{safe}.png"))
        plot_pizza_attacking(df, team, output_path=str(charts_dir / f"pizza_attacking_{safe}.png"))

    print("  Generating Chart 8 — Pizza plots (foul analysis) ...")
    for team in test_teams:
        safe = team.lower().replace(" ", "_")
        plot_pizza_foul(df, team, output_path=str(charts_dir / f"pizza_foul_{safe}.png"))

    print(f"  All charts saved to {charts_dir}")
