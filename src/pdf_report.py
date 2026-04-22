"""
pdf_report.py
-------------
Generates a 7-page A4 portrait PDF multi-match report focused on Juventus,
mirroring Report Multi Match_v3.pptx.

Usage:
    python3 -m src.pdf_report \
        --csv output/master-0.1.3/all_transitions.csv \
        --output output/report.pdf \
        [--meta out_rest_defence_20260421_meta.json] \
        [--focus-team "Juventus"]
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
from PIL import Image

from src.visualizations import (
    EXCLUDED_TEAMS, PRIMARY_RED, PRIMARY_ORANGE, PRIMARY_YELLOW,
    spe_from_csv,
)
from src.logos import get_logo_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGE_W    = 8.27    # A4 portrait width (inches)
_PAGE_H    = 11.69   # A4 portrait height (inches)
_BG        = "#FAFAFA"
_EMBED_DPI = 200
_LOGO_PATH = Path(__file__).parent / "images" / "SportAnalytics-logo.png"

_COMPARISON_TEAMS = ["Como", "Inter Milan", "Juventus"]

import textwrap as _tw


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_EMBED_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return np.array(Image.open(buf))


def _embed(ax: plt.Axes, fig: plt.Figure) -> None:
    """Embed fig at natural proportions — no resizing, no stretching."""
    img = _fig_to_image(fig)
    plt.close(fig)
    ax.imshow(img, interpolation="none")  # 'none' = no additional resampling blur
    ax.set_facecolor(_BG)
    ax.axis("off")


def _question_ax(ax: plt.Axes, text: str, fontsize: int = 10) -> None:
    ax.set_facecolor(_BG)
    ax.axis("off")
    ax.text(0.01, 0.5, _tw.fill(text, width=95),
            transform=ax.transAxes, fontsize=fontsize, fontstyle="italic",
            va="center", ha="left", color="#222222", multialignment="left")


def _title_ax(ax: plt.Axes, text: str, fontsize: int = 11) -> None:
    ax.set_facecolor(_BG)
    ax.axis("off")
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="center", ha="center", color=PRIMARY_RED)


class _Layout:
    """Top-to-bottom absolute layout on an A4 figure.

    All heights are in figure-fraction units (0–1).
    A4 = 11.69", so fraction 0.20 ≈ 2.34" — a comfortable chart height.
    """
    L    = 0.04          # left margin
    R    = 0.96          # right margin
    W    = R - L         # full chart width fraction
    HALF = (R - L - 0.02) / 2   # half-width (pair layout)
    MID  = L + HALF + 0.02      # x start of right panel in pair

    T_H  = 0.042   # title block height
    Q_H  = 0.030   # question text height
    C_H  = 0.200   # standard chart height  (≈ 2.34")
    P_H  = 0.260   # tall chart (pizza)    (≈ 3.04")
    GAP  = 0.007   # gap between elements

    def __init__(self, fig: plt.Figure):
        self.fig = fig
        self.y   = 0.97   # current top position, moving downward

    def _ax(self, bottom, height, left=None, width=None):
        l = left  if left  is not None else self.L
        w = width if width is not None else self.W
        return self.fig.add_axes([l, bottom, w, height])

    def title(self, text, fontsize=11):
        ax = self._ax(self.y - self.T_H, self.T_H)
        _title_ax(ax, text, fontsize=fontsize)
        self.y -= self.T_H + self.GAP

    def question(self, text, fontsize=10):
        ax = self._ax(self.y - self.Q_H, self.Q_H)
        _question_ax(ax, text, fontsize=fontsize)
        self.y -= self.Q_H + self.GAP

    def question_pair(self, t1, t2, fontsize=9):
        ax1 = self._ax(self.y - self.Q_H, self.Q_H, self.L,   self.HALF)
        ax2 = self._ax(self.y - self.Q_H, self.Q_H, self.MID, self.HALF)
        _question_ax(ax1, t1, fontsize=fontsize)
        _question_ax(ax2, t2, fontsize=fontsize)
        self.y -= self.Q_H + self.GAP

    def chart(self, fig, height=None, width=None):
        """Place a chart.
        width  = fraction of page width  → slot height auto-calculated from aspect ratio (no letterboxing)
        height = fraction of page height → slot width fixed at self.W (may letterbox)
        """
        if width is not None:
            img = _fig_to_image(fig)
            plt.close(fig)
            ih, iw = img.shape[:2]
            slot_h = (width * _PAGE_W) / (iw / ih) / _PAGE_H
            left = (1.0 - width) / 2
            ax = self.fig.add_axes([left, self.y - slot_h, width, slot_h])
            ax.imshow(img, interpolation="none")
            ax.set_facecolor(_BG)
            ax.axis("off")
            h = slot_h
        else:
            h = height if height is not None else self.C_H
            ax = self._ax(self.y - h, h)
            _embed(ax, fig)
        self.y -= h + self.GAP

    def chart_pair(self, fig1, fig2, height=None, width=None):
        """Place two charts side by side.
        width  = fraction of page width for each chart (height auto from aspect ratio)
        height = fraction of page height for each slot
        """
        if width is not None:
            img1 = _fig_to_image(fig1); plt.close(fig1)
            img2 = _fig_to_image(fig2); plt.close(fig2)
            h1 = (width * _PAGE_W) / (img1.shape[1] / img1.shape[0]) / _PAGE_H
            h2 = (width * _PAGE_W) / (img2.shape[1] / img2.shape[0]) / _PAGE_H
            h = max(h1, h2)
            gap = 0.01
            ax1 = self.fig.add_axes([self.L, self.y - h, width, h])
            ax2 = self.fig.add_axes([self.L + width + gap, self.y - h, width, h])
            for ax, img in ((ax1, img1), (ax2, img2)):
                ax.imshow(img, interpolation="none")
                ax.set_facecolor(_BG)
                ax.axis("off")
        else:
            h = height if height is not None else self.P_H
            ax1 = self._ax(self.y - h, h, self.L,   self.HALF)
            ax2 = self._ax(self.y - h, h, self.MID, self.HALF)
            _embed(ax1, fig1)
            _embed(ax2, fig2)
        self.y -= h + self.GAP


def _load_meta(meta_path) -> dict | None:
    if meta_path is None:
        return None
    with open(meta_path) as f:
        matches = json.load(f)
    dates, names = [], []
    for m in matches:
        name = m.get("match_name", "")
        names.append(name)
        parts = name.split("_")
        if parts:
            dates.append(parts[0])
    dates.sort()
    return {
        "matches": names,
        "date_first": dates[0] if dates else None,
        "date_last":  dates[-1] if dates else None,
    }


def _new_page(pdf: PdfPages) -> plt.Figure:
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    fig.patch.set_facecolor(_BG)
    return fig


def _save(fig: plt.Figure, pdf: PdfPages) -> None:
    pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 1 — Cover
# ---------------------------------------------------------------------------

def _page_cover(df: pd.DataFrame, meta: dict | None, focus_team: str,
                pdf: PdfPages) -> None:
    fig = _new_page(pdf)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_BG)
    ax.axis("off")

    logo_top = 0.78

    if _LOGO_PATH.exists():
        logo_img = mpimg.imread(str(_LOGO_PATH))
        logo_h_pts = logo_img.shape[0]
        logo_w_pts = logo_img.shape[1]
        aspect = logo_w_pts / logo_h_pts
        logo_h_fig = 0.14
        logo_w_fig = logo_h_fig * aspect * (_PAGE_H / _PAGE_W)
        logo_ax = fig.add_axes([0.5 - logo_w_fig / 2, logo_top,
                                 logo_w_fig, logo_h_fig])
        logo_ax.imshow(logo_img)
        logo_ax.axis("off")
        logo_top -= 0.02

    fig.text(0.5, logo_top - 0.02, "Multi-Match Report",
             fontsize=32, fontweight="bold", ha="center", va="top",
             color="#222222")
    fig.text(0.5, logo_top - 0.10,
             "Rest Defence Analysis — Defensive and Attacking Transitions",
             fontsize=15, ha="center", va="top",
             fontstyle="italic", color="#666666")
    fig.text(0.5, logo_top - 0.155,
             f"Focus team: {focus_team}",
             fontsize=13, ha="center", va="top", color=PRIMARY_RED,
             fontweight="bold")

    if meta:
        date_str = (f"{meta['date_first']}  —  {meta['date_last']}"
                    if meta["date_first"] else "")
        if date_str:
            fig.text(0.5, logo_top - 0.21, date_str,
                     fontsize=13, ha="center", va="top", color="#444444")

    # Footer bar
    bar_ax = fig.add_axes([0, 0, 1, 0.03])
    bar_ax.set_facecolor(PRIMARY_RED)
    bar_ax.axis("off")

    _save(fig, pdf)


# ---------------------------------------------------------------------------
# Page 2 — Ball Regain Effectiveness
# ---------------------------------------------------------------------------

def _page_ball_regain(df: pd.DataFrame, focus_team: str, pdf: PdfPages) -> None:
    from src.visualizations.ratings_bar import plot_ratings_bar
    from src.visualizations.spe_bar import _plot_spe_bars_single
    from src.visualizations.pizza_plots import plot_pizza_defending

    fig = _new_page(pdf)
    lay = _Layout(fig)
    lay.title("Ball Regain Effectiveness in Defensive Transitions After Final Third Ball Loss")
    lay.question("How effective are we at controlling opponent transitions compared to the league?")
    lay.question("What proportion of our defensive transitions lead to dangerous outcomes?")
    lay.chart(plot_ratings_bar(df), width=0.88)
    lay.question_pair(f"Ball Regain Effectiveness — {focus_team} Overview",
                      "SPE 20s — Team Comparison")
    lay.chart_pair(
        plot_pizza_defending(df, focus_team),
        _plot_spe_bars_single(df, "spe_20", "Structural Prevention Efficiency (20s Window)"),
        width=0.43,
    )
    _save(fig, pdf)


# ---------------------------------------------------------------------------
# Page 3 — Team Structure & Organization
# ---------------------------------------------------------------------------

def _page_team_structure(df: pd.DataFrame, focus_team: str, pdf: PdfPages) -> None:
    from src.visualizations.scatterplots import plot_lineheight_vs_cadv, plot_compactness_vs_length
    from src.visualizations.line_plots import plot_metric_evolution

    fig = _new_page(pdf)
    lay = _Layout(fig)
    lay.title("Team Structure & Organization in Defensive Transitions After Ball Loss in the Final Third")
    lay.question("How does the team's rest defence structure translate into recovery behaviour "
                 "after losing possession?")
    lay.chart(plot_lineheight_vs_cadv(df), width=0.88)
    lay.question("How compact is our team compared to others?")
    lay.chart(plot_compactness_vs_length(df), width=0.88)
    lay.question("Do we maintain a controlled defensive line during transition?")
    lay.chart(plot_metric_evolution(df, "team_centroid_x_norm", teams=_COMPARISON_TEAMS), width=0.88)
    _save(fig, pdf)


# ---------------------------------------------------------------------------
# Page 4 — Foul-Based Transition Control
# ---------------------------------------------------------------------------

def _page_foul_control(df: pd.DataFrame, focus_team: str, pdf: PdfPages) -> None:
    from src.visualizations.structural_table import plot_foul_table
    from src.visualizations.scatterplots import plot_foul_time_vs_location
    from src.visualizations.line_plots import plot_metric_evolution

    fig = _new_page(pdf)
    lay = _Layout(fig)
    lay.title("Foul-Based Transition Control — How We Stop Counterattacks")
    lay.question("How efficient are we at interrupting transitions through fouls? "
                 "Are we fouling under control (numerical superiority) or in emergency situations?")
    lay.chart(plot_foul_table(df), width=0.88)
    lay.question("When do we choose to foul during defensive transitions?")
    lay.chart(plot_foul_time_vs_location(df), width=0.88)
    lay.question("How aggressive is the team immediately after transition, "
                 "and how does this evolve over time?")
    lay.chart(plot_metric_evolution(df, "n_pressing_team", teams=_COMPARISON_TEAMS,
                                    offsets=("t10", "t50", "t100"),
                                    labels=["After 1s", "After 5s", "After 10s"]), width=0.88)
    _save(fig, pdf)


# ---------------------------------------------------------------------------
# Page 5 — Attacking Transition Profile
# ---------------------------------------------------------------------------

def _page_attacking(df: pd.DataFrame, focus_team: str, pdf: PdfPages) -> None:
    from src.visualizations.structural_table import plot_attacking_transitions
    from src.visualizations.pizza_plots import plot_pizza_attacking

    fig = _new_page(pdf)
    lay = _Layout(fig)
    lay.title("Attacking Transition Profile — From Recovery on Our Own 3rd to Progression")
    lay.question("Which teams are most effective at turning ball recoveries into forward progression?")
    lay.chart(plot_attacking_transitions(df), width=0.88)
    lay.question(f"Is {focus_team} more direct, structured, or dependent on specific players "
                 "in attacking transitions?")
    lay.chart(plot_pizza_attacking(df, focus_team), width=0.70)
    _save(fig, pdf)


# ---------------------------------------------------------------------------
# Page 6 — Rankings
# ---------------------------------------------------------------------------

_RANK_METRICS = {
    "Ball Regain Effectiveness & Team Structure": [
        [
            ("SPE (20s)",
             lambda df, t: spe_from_csv(df, t)[1],
             True),
            ("Team Length",
             lambda df, t: df[df["losing_team_name"] == t]["team_length_m_t0"].mean(),
             False),
            ("Line Height",
             lambda df, t: df[df["losing_team_name"] == t]["line_height_m_t0"].mean(),
             True),
        ],
        [
            ("Players Behind Ball",
             lambda df, t: df[df["losing_team_name"] == t]["players_behind_ball_t0"].mean(),
             True),
            ("Num. Superiority",
             lambda df, t: df[df["losing_team_name"] == t]["num_superiority_app1_t0"].mean(),
             True),
            ("Compactness Ratio",
             lambda df, t: df[df["losing_team_name"] == t]["team_compactness_t0"].mean(),
             False),
        ],
    ],
    "Foul-Based Transition Control": [
        [
            ("Foul Interruption Rate",
             lambda df, t: (
                 df[df["losing_team_name"] == t]["foul_committed"].astype(bool).sum()
                 / max(len(df[df["losing_team_name"] == t]), 1) * 100
             ),
             True),
            ("Team Press t1",
             lambda df, t: df[df["losing_team_name"] == t]["n_pressing_team_t10"].mean(),
             True),
            ("Team Press t10",
             lambda df, t: df[df["losing_team_name"] == t]["n_pressing_team_t100"].mean(),
             True),
        ],
    ],
    "Attacking Transition Profile": [
        [
            ("Constructive Prog %",
             lambda df, t: df[df["gaining_team_name"] == t]["constructive_progression"]
                            .astype(float).mean() * 100,
             True),
            ("Own Half Exit %",
             lambda df, t: df[df["gaining_team_name"] == t]["own_half_exit"]
                            .astype(float).mean() * 100,
             True),
            ("Productive Pass 45°",
             lambda df, t: df[df["gaining_team_name"] == t]["productive_pass_ratio_45"]
                            .mean() * 100,
             True),
        ],
    ],
}


def _draw_ranking_column(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    extractor,
    higher_better: bool,
) -> None:
    """Full-page ranking column: all teams, logo + name + value per row."""
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    all_teams = [
        t for t in
        set(df["losing_team_name"].unique()) | set(df["gaining_team_name"].unique())
        if t not in EXCLUDED_TEAMS
    ]
    vals = {t: float(extractor(df, t)) for t in all_teams}
    vals = {t: v for t, v in vals.items() if not np.isnan(v)}
    ranked = sorted(vals.items(), key=lambda x: x[1], reverse=higher_better)
    n = len(ranked)
    if n == 0:
        return

    # Metric label header
    ax.text(0.5, 0.985, label,
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="center", va="top", color=PRIMARY_RED)
    ax.axhline(y=0.965, xmin=0.01, xmax=0.99, color="#CCCCCC", linewidth=0.8)

    row_h = 0.955 / n
    row_colors = ["#FFFFFF", "#F2F2F2"]

    for i, (team, val) in enumerate(ranked):
        y_bot = 0.960 - (i + 1) * row_h
        y_mid = y_bot + row_h / 2

        # Row background
        rect = mpatches.FancyBboxPatch(
            (0.01, y_bot), 0.98, row_h,
            boxstyle="square,pad=0",
            facecolor=row_colors[i % 2],
            edgecolor="#E8E8E8", linewidth=0.3,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)

        # Rank number
        ax.text(0.04, y_mid, f"{i + 1}.",
                transform=ax.transAxes, fontsize=7.5,
                va="center", ha="left", color="#888888")

        # Logo — zoom fixed so logo ≈ 75% of row physical height
        # row physical height in inches = row_h * axes_h_in (0.910 * 11.69 = 10.64")
        # target logo ≈ 0.35": zoom = 0.35 / (64px / 72dpi) = 0.35 / 0.889 ≈ 0.39
        logo = get_logo_image(team, size=64)
        if logo is not None:
            zoom = min(0.10, row_h * 10.64 * 0.75 / (256 / 72))
            img = OffsetImage(logo, zoom=zoom)
            ab = AnnotationBbox(
                img, (0.18, y_mid),
                frameon=False, xycoords="axes fraction",
                box_alignment=(0.5, 0.5), pad=0, zorder=4,
            )
            ax.add_artist(ab)

        # Team name
        ax.text(0.30, y_mid, team,
                transform=ax.transAxes, fontsize=8,
                va="center", ha="left", color="#222222")

        # Value
        ax.text(0.97, y_mid, f"{val:.1f}",
                transform=ax.transAxes, fontsize=8,
                va="center", ha="right", color="#444444",
                fontweight="bold")


def _page_rankings(df: pd.DataFrame, pdf: PdfPages) -> None:
    """One page per metric row — 4 pages total, 3 full rankings per page."""
    # Flatten into pages: each page = one metric row with its section label
    pages = []
    for section_name, rows in _RANK_METRICS.items():
        for row in rows:
            pages.append((section_name, row))

    for section_name, metrics in pages:
        fig = _new_page(pdf)

        # Page title
        fig.text(0.5, 0.975, "RANKINGS",
                 fontsize=18, fontweight="bold", ha="center", va="top",
                 color="#222222")
        fig.text(0.5, 0.950, section_name,
                 fontsize=10, fontweight="bold", ha="center", va="top",
                 color=PRIMARY_RED)

        # Separator under subtitle
        sep_ax = fig.add_axes([0.04, 0.938, 0.92, 0.001])
        sep_ax.set_facecolor(PRIMARY_RED)
        sep_ax.axis("off")

        # 3 ranking columns, equal width
        col_w = 0.285
        col_gap = 0.025
        col_left = [0.04, 0.04 + col_w + col_gap, 0.04 + 2 * (col_w + col_gap)]

        for col_idx, (label, extractor, hib) in enumerate(metrics):
            ax = fig.add_axes([col_left[col_idx], 0.02, col_w, 0.910])
            _draw_ranking_column(ax, df, label, extractor, hib)

        _save(fig, pdf)


# ---------------------------------------------------------------------------
# Page 7 — Glossary
# ---------------------------------------------------------------------------

_GLOSSARY = [
    ("Transition Rating",
     "Overall quality classification of the defending team's response: "
     "Best / Good / Okay / Bad."),
    ("SPE (20s)",
     "Structural Prevention Efficiency (20s window): % of transitions where the "
     "defending team prevents the opponent from reaching the defensive third within 20s "
     "of losing possession."),
    ("Team Length",
     "Distance (m) from the most forward to the deepest outfield player at transition "
     "start (t0). Lower values indicate a more compact vertical shape."),
    ("Line Height",
     "Average distance (m) of the defensive line from the team's own goal at t0. "
     "Higher values reflect a higher, more proactive defensive block."),
    ("Players Behind Ball",
     "Number of defending players positioned between the ball and their own goal at t0."),
    ("Numerical Superiority",
     "Player count advantage of the defending team in the defensive zone at t0 "
     "(App1 rule-based zone). Positive values mean more defenders than attackers."),
    ("Compactness Ratio",
     "Horizontal width of the defending team (m) at t0. Lower values indicate "
     "a tighter, more organised defensive block."),
    ("Centroid Advance",
     "Forward displacement (m) of the defending team's centroid in the first 5s "
     "after ball loss. Positive = team moves toward opponent goal (pressing)."),
    ("Foul Interruption Rate",
     "% of defensive transitions in which the defending team commits a foul to stop "
     "the opponent's progression."),
    ("Team Press t1 / t10",
     "Number of outfield players actively pressing (marked with p_ > 0) at 1s (t1) "
     "and 10s (t10) after the transition. Tracks how pressing intensity evolves."),
    ("Constructive Progression",
     "% of attacking transitions in which the gaining team completes ≥3 forward-oriented "
     "passes within 15s, indicating structured forward play."),
    ("Own Half Exit",
     "% of attacking transitions where the gaining team advances the ball beyond "
     "the halfway line."),
    ("Productive Pass Ratio 45°",
     "% of passes played within a ±45° forward cone during the transition. "
     "Higher values indicate more direct, vertical progression."),
    ("Playmaker Independence",
     "Inverse of reliance on the deep-lying playmaker: (1 − dependency) × 100. "
     "Higher values mean more distributed ball circulation across the team."),
    ("Transition Speed",
     "Average duration (s) of the attacking transition from ball gain to "
     "end of sequence. Lower values indicate faster counters."),
]


def _page_glossary(pdf: PdfPages) -> None:
    fig = _new_page(pdf)
    gs = gridspec.GridSpec(
        1, 1,
        figure=fig,
        left=0.06, right=0.94, top=0.95, bottom=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(_BG)
    ax.axis("off")

    fig.text(0.5, 0.975, "Glossary — Metric Definitions",
             fontsize=14, fontweight="bold",
             ha="center", va="top", color=PRIMARY_RED)

    # Draw alternating rows — start below the title with extra gap
    n = len(_GLOSSARY)
    table_top = 0.91   # axes fraction — leaves space below title
    row_h = table_top / n

    for i, (term, defn) in enumerate(_GLOSSARY):
        y_bot = table_top - (i + 1) * row_h
        bg = "#F5F5F5" if i % 2 == 0 else "#FFFFFF"
        rect = mpatches.FancyBboxPatch(
            (0.0, y_bot), 1.0, row_h,
            boxstyle="square,pad=0",
            facecolor=bg, edgecolor="#E0E0E0", linewidth=0.4,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)

        y_mid = y_bot + row_h / 2

        # Term (indented from left edge)
        ax.text(0.02, y_mid, term,
                transform=ax.transAxes,
                fontsize=7.5, fontweight="bold",
                va="center", ha="left", color=PRIMARY_RED,
                clip_on=False)

        # Definition — wrapped, starting after metric name column
        wrapped = _tw.fill(defn, width=85)
        ax.text(0.29, y_mid, wrapped,
                transform=ax.transAxes,
                fontsize=7, va="center", ha="left",
                color="#333333", clip_on=False)

    _save(fig, pdf)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def generate_pdf_report(
    df: pd.DataFrame,
    output_path: str | Path,
    meta_path: str | Path | None = None,
    focus_team: str = "Juventus",
) -> None:
    """Generate 7-page A4 portrait PDF report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(meta_path)

    print(f"Generating PDF report (focus: {focus_team})...")
    with PdfPages(str(output_path)) as pdf:
        print("  Page 1 — Cover ...")
        _page_cover(df, meta, focus_team, pdf)

        print("  Page 2 — Ball Regain Effectiveness ...")
        _page_ball_regain(df, focus_team, pdf)

        print("  Page 3 — Team Structure & Organization ...")
        _page_team_structure(df, focus_team, pdf)

        print("  Page 4 — Foul-Based Transition Control ...")
        _page_foul_control(df, focus_team, pdf)

        print("  Page 5 — Attacking Transition Profile ...")
        _page_attacking(df, focus_team, pdf)

        print("  Page 6 — Rankings ...")
        _page_rankings(df, pdf)

        print("  Page 7 — Glossary ...")
        _page_glossary(pdf)

    print(f"\nReport saved: {output_path}")


def _load_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV with auto-detected separator (comma or semicolon)."""
    with open(csv_path) as f:
        first = f.readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(csv_path, sep=sep)


def _resolve_team_names(df: pd.DataFrame, meta_path: str | Path) -> pd.DataFrame:
    """Map Team_ID placeholders to real team names using the meta JSON."""
    with open(meta_path) as f:
        matches = json.load(f)
    team_map: dict[str, str] = {}
    for m in matches:
        for side in ("home_team", "away_team"):
            t = m[side]
            team_map[f"Team_{t['team_id']}"] = t["team_name"]
    df = df.copy()
    df["losing_team_name"]  = df["losing_team_name"].map(team_map).fillna(df["losing_team_name"])
    df["gaining_team_name"] = df["gaining_team_name"].map(team_map).fillna(df["gaining_team_name"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-match PDF report")
    parser.add_argument("--csv",         required=True,             help="Path to transitions CSV")
    parser.add_argument("--output",      required=True,             help="Output PDF path")
    parser.add_argument("--meta",        default=None,              help="Meta JSON (required for team name resolution)")
    parser.add_argument("--focus-team",  default="Juventus",        help="Team to spotlight (default: Juventus)")
    args = parser.parse_args()

    df = _load_csv(args.csv)
    if args.meta:
        df = _resolve_team_names(df, args.meta)
    generate_pdf_report(df, args.output, args.meta, args.focus_team)


if __name__ == "__main__":
    main()
