"""
report_generator.py
-------------------
Standalone report generator. Reads only from a metrics CSV produced by main.py.
No raw tracking data required.

Usage:
  python report_generator.py --input all_transitions.csv --output-dir output/
  python report_generator.py --input all_transitions.csv --match-id 7418 --output-dir output/
  python report_generator.py --input all_transitions.csv --teams "Juventus,Parma" --output-dir output/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.helpers import (
    format_value, col_mean, col_delta_mean, pct_bool, pct_delta, jersey_str,
)


# ---------------------------------------------------------------------------
# SPE from CSV flags (no raw data required)
# ---------------------------------------------------------------------------

def _spe_from_csv(df: pd.DataFrame, team_name: str) -> tuple[float, float]:
    """Compute SPE(15s) and SPE(20s) from per-transition CSV flags."""
    tdf = df[df["losing_team_name"] == team_name]
    valid_15 = tdf[tdf["has_15s_window"].astype(bool)]
    valid_20 = tdf[tdf["has_20s_window"].astype(bool)]
    spe_15 = (1.0 - valid_15["ball_reached_third_15s"].astype(float).mean()) * 100
    spe_20 = (1.0 - valid_20["ball_reached_third_20s"].astype(float).mean()) * 100
    return spe_15, spe_20


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Glossaries
# ---------------------------------------------------------------------------

_SINGLE_MATCH_GLOSSARY = """
---

## Metric Glossary

### Time Offsets
| Label | Meaning |
|---|---|
| **t0** | Exact frame when possession is lost (transition moment) |
| **t0+1s** | 1 second after t0. Press values are meaningful here; at t0 the team was in possession so p_ = 0 by definition |
| **t0+5s** | 5 seconds after t0 |
| **t0+10s** | 10 seconds after t0 |

### Negative Transition Metrics (defending team quality)

| Metric | Abbreviation / Column | Interpretation |
|---|---|---|
| SPE (15s) | Structural Prevention Efficiency | % of transitions where ball does NOT enter the defensive third within 15 s. **Higher = better defensive organisation.** |
| SPE (20s) | Structural Prevention Efficiency (20s) | Same metric with 20-second window. |
| Team Length | `team_length_m` | Distance (m) from rearmost to foremost outfield player along the pitch axis. **Lower = more compact block.** |
| Line Height | `line_height_m` | Distance (m) from the rearmost defensive line centroid to the team's own goal line. **Higher = more aggressive pressing line; lower = deeper, more cautious.** |
| Players Behind Ball | `players_behind_ball` | Count of outfield defenders with position behind the ball at t0. **More = better immediate cover.** |
| NumSup RD App1 (Rule-Based) / App2 (Clustering) | `num_superiority_app1/2` | Defenders minus attackers in the App1 / App2 rest-defence zones. App1 uses rule-based zone construction; App2 uses k-means clustering. **Positive = numerical advantage; negative = gap in zone.** |
| Team Compactness (m) | `team_compactness` | Mean Euclidean distance (m) of outfield defenders from their team centre. **Lower = more compact shape; higher = more spread.** |
| Zone Press | `zone_press_app1/2` | Mean pressing intensity (`p_` column, 0–100 scale) of defending players inside the App1/App2 zone. **0 = no pressing, 100 = maximum pressing.** Reported as percentage change (Δ%) from t0+1s baseline. Negative Δ = pressing harder over time (good). |
| Team Press | `team_press` | Mean pressing intensity of ALL defending outfield players. Same inverted 0–100 scale. **Negative Δ = team increases pressing over time.** |
| Centroid Advance | `centroid_advance_5s_m` / `_10s_m` | Forward movement (m) of the defending team's centroid toward the opponent goal. **Positive = recovering/advancing shape; negative = retreating.** |

### Transition Rating
| Rating | Meaning |
|---|---|
| **Best** | Defending team regains possession within 5 seconds |
| **Bad** | Shot conceded within 15 s, OR ball enters defensive third within 5 s, OR opponent plays an in-behind pass (StartX > 750) within 15 s, OR gaining team was fouled with ≥2 player numerical advantage for defending team |
| **Good** | Ball goes out of play within 15 s, OR opponent commits a foul (defending team wins free kick), OR defending team regains 5–15 s without dangerous outcome |
| **Okay** | Gaining team was fouled with ≤1 player advantage for defending team, OR attack persists beyond 15 s without matching Bad/Good conditions |


### Positive Transition Metrics (gaining team attack quality)

| Metric | Column | Interpretation |
|---|---|---|
| Constructive Progression | `constructive_progression` | % of transitions where the gaining team records a possession phase with ≥ 3 passes (PassCount ≥ 3) within 15 s. **Higher = opponent builds play more often.** |
| Own Half Exit | `own_half_exit` | % of transitions where the gaining team has a possession phase starting from their own half (StartX ≤ 500 in Wyscout 0–1000 coords) within 15 s. **Higher = opponent involves deeper players.** |
| Forward Pass Ratio (45°) | `productive_pass_ratio_45` | % of gaining team's passes in 15 s that are strict forward passes (within 45° of attack direction). **Higher = opponent attacks more directly.** |
| Forward Pass Ratio (90°) | `productive_pass_ratio_90` | % of gaining team's passes in 15 s that are forward or sideways (within 90°, excludes only backward passes). **Higher = opponent tends to advance, not play backward.** |
| Playmaker Dep. (1st pass) | `playmaker_dependency_1st` | % of transitions where the 1st post-transition pass targets the gaining team's auto-identified Deep-Lying Playmaker. **Higher = opponent's first instinct is to find the playmaker.** |
| Playmaker Dep. (2nd pass) | `playmaker_dependency_2nd` | % of transitions where the 2nd pass targets the playmaker. Together with the 1st-pass metric, shows how quickly the team routes through their key player. |
"""

_MULTI_MATCH_GLOSSARY = """
---

## Metric Glossary

### Column Abbreviations

| Column | Full Name | Interpretation |
|---|---|---|
| **N** | Transitions | Number of rest-defence transitions analysed for this team |
| **SPE (15s)** | Structural Prevention Efficiency | % of transitions where ball does NOT reach defensive third in 15 s. Higher = better. |
| **SPE (20s)** | Structural Prevention Efficiency (20s) | Same metric with 20-second window. |
| **% Best/Good/Okay/Bad** | Transition Rating | Distribution of qualitative transition outcomes. Best = regain ≤5s; Good = ball dead or opponent fouls; Okay = opponent fouled with small advantage or prolonged attack; Bad = shot/penetration/dangerous play conceded. |
| **TeamLen(m)** | Team Length | Distance (m) from rearmost to foremost outfield player at t0. Lower = more compact. |
| **LineHt(m)** | Line Height | Distance (m) from rearmost defensive line to own goal at t0. Higher = higher, more aggressive line. |
| **BehindBall** | Players Behind Ball | Count of outfield players positioned behind the ball at t0. More = better cover. |
| **NumSup RD App1 (Rule-Based) / App2 (Clustering)** | Numerical Superiority Rest Defence | Defenders minus attackers in App1/App2 zone at t0. App1 = rule-based; App2 = clustering. Positive = advantage; negative = gap. |
| **TeamComp(m)** | Team Compactness | Mean Euclidean distance (m) of outfield defenders from team centre at t0. Lower = more compact shape. |
| **NumSup App1 (5s)** | Numerical Superiority App1 at t0+5s | Defenders minus attackers in App1 (rule-based) zone 5 s after transition. Positive = defending team advantage. |
| **NumSup App1 (10s)** | Numerical Superiority App1 at t0+10s | Same metric 10 s after transition. |
| **NumSup App2 (5s)** | Numerical Superiority App2 at t0+5s | Defenders minus attackers in App2 (clustering) zone 5 s after transition. Positive = defending team advantage. |
| **NumSup App2 (10s)** | Numerical Superiority App2 at t0+10s | Same metric 10 s after transition. |
| **CompΔ(5s)** | Compactness Delta at 5 s | Change in Team Compactness (m) from t0 to t0+5s. Negative = team tightened up (good). |
| **ZPress1(t1s)** | Zone Press App1 at t0+1s | Mean pressing intensity (`p_` column, 0–100 scale) of defenders inside App1 zone, 1 s after transition. 0 = no pressing, 100 = maximum pressing. |
| **ZPress1Δ%(5s)** | Zone Press App1 Δ% 1→5s | Percentage change in zone pressing intensity from t0+1s to t0+5s. **Positive = team pressed harder.** |
| **ZPress1Δ%(10s)** | Zone Press App1 Δ% 1→10s | Percentage change from t0+1s to t0+10s. Positive = increased pressing. |
| **ZPress2Δ%(5s)** | Zone Press App2 Δ% 1→5s | Same for App2 (clustering) zone approach. |
| **TmPress(t1s)** | Team Press at t0+1s | Mean pressing intensity of all outfield defenders, 1 s after transition. Same 0–100 scale. |
| **TmPressΔ%(5s)** | Team Press Δ% 1→5s | Percentage change in overall team pressing from t0+1s to t0+5s. Positive = pressed harder. |
| **TmPressΔ%(10s)** | Team Press Δ% 1→10s | Percentage change from t0+1s to t0+10s. |
| **EscZ-Δ%(5s)** | Escape Pressure Zone Δ% 1→5s | Percentage change in pressure received (`ps_` column, 0–100 scale) by gaining-team players inside zone. Positive = more pressure (not escaping). Computed from transitions where team was gaining. |
| **EscZ-Δ%(10s)** | Escape Pressure Zone Δ% 1→10s | Same from t0+1s to t0+10s. |
| **EscT-Δ%(5s)** | Escape Pressure Team Δ% 1→5s | Percentage change in pressure received across all gaining-team outfield players. |
| **EscT-Δ%(10s)** | Escape Pressure Team Δ% 1→10s | Same from t0+1s to t0+10s. |
| **CAdv5s(m)** | Centroid Advance 5 s | Forward movement (m) of the defending team centroid in 5 s. Positive = recovering shape. |
| **CAdv10s(m)** | Centroid Advance 10 s | Forward movement (m) of the defending team centroid in 10 s. |
| **ConstrProg%** | Constructive Progression % | % of transitions where **team made ≥ 3 passes within 15 s when attacking**. Higher = more structured build-up. Computed from team's gaining transitions. |
| **OwnHalfExit%** | Own Half Exit % | % of transitions where **team controlled ball in their own half within 15 s when attacking**. Higher = ball retained in safe area. |
| **ProdPass(45°)%** | Forward Pass Ratio (45°) | % of **team's passes in 15 s that go strictly forward** (within 45° of attack direction) when attacking. |
| **ProdPass(90°)%** | Forward Pass Ratio (90°) | % of **team's passes in 15 s that go forward or sideways** (within 90°) when attacking. |
| **PM Dep (1st)%** | Playmaker Dep. 1st pass | % of transitions where **team's 1st pass targets the Deep-Lying Playmaker** when attacking. |
| **PM Dep (2nd)%** | Playmaker Dep. 2nd pass | % of transitions where **team's 2nd pass targets the Deep-Lying Playmaker** when attacking. |

### Pressure Scale Notes
- **Pressing (`p_` columns)**: 0 = no pressing, 100 = maximum pressing intensity. Reported as **percentage change (Δ%) from t0+1s baseline**. Positive Δ% = team pressed harder.
- **Escape Pressure (`ps_` columns)**: 0 = no pressure received, 100 = maximum pressure received. Reported as **percentage change (Δ%) from t0+1s baseline**. Positive Δ% = more pressure received (unsuccessful escape); negative Δ% = pressure relieved (successful escape).

### Transition Types

| Type | Definition | Metrics |
|---|---|---|
| **Positive Transition** | When team **loses possession** and transitions to defending. Measures how the team responds structurally and tactically. | CAdv5s/10s, TeamLen Δ, Compact Δ |
| **Negative Transition** | When team **gains possession** and transitions to attacking. Measures how effectively the team builds play and progresses the ball. | ConstrProg%, OwnHalfExit%, ProdPass, PM Dep |
"""


# ---------------------------------------------------------------------------
# Per-match report (reads from CSV, no raw data)
# ---------------------------------------------------------------------------

def generate_match_report(
    df: pd.DataFrame,
    match_id: str,
    out_dir: Path,
    playmakers: dict | None = None,
    jersey_map: dict | None = None,
) -> None:
    """Generate match_<id>_summary.md from a metrics DataFrame (CSV-sourced)."""
    mdf = df[df["match_id"].astype(str) == str(match_id)] if "match_id" in df.columns else df
    if mdf.empty:
        return

    offset_keys   = [0, 2, 10, 20]
    offset_labels = ["t0", "t0+1s", "t0+5s", "t0+10s"]
    md_lines: list[str] = [f"# Match {match_id} — Rest Defence Summary\n"]

    for team_name in sorted(mdf["losing_team_name"].unique()):
        tdf = mdf[mdf["losing_team_name"] == team_name]
        gdf = mdf[mdf["gaining_team_name"] == team_name]
        n   = len(tdf)

        spe_15, spe_20 = _spe_from_csv(mdf, team_name)
        spe_15_str = format(spe_15, ".1f") + "%" if not np.isnan(spe_15) else "—"
        spe_20_str = format(spe_20, ".1f") + "%" if not np.isnan(spe_20) else "—"

        rts = tdf["transition_rating"] if "transition_rating" in tdf.columns else pd.Series(dtype=str)
        pct = lambda r: f"{(rts == r).sum() / n * 100:.0f}%" if n > 0 else "—"

        # Resolve playmaker for this team's gaining transitions
        team_tid = None
        if len(gdf) > 0 and "gaining_team_id" in gdf.columns:
            mode = gdf["gaining_team_id"].mode()
            if len(mode) > 0:
                team_tid = int(mode.iloc[0])
        pm_id  = (playmakers or {}).get((str(match_id), team_tid)) if team_tid else None
        pm_str = jersey_str(pm_id, jersey_map or {})

        md_lines += [
            f"## {team_name} (defending) — {n} transitions\n",
            "### Overview\n",
            _md_table(
                ["Metric", "Value", "Notes"],
                [
                    ["SPE (15s)",  spe_15_str,
                     "% transitions where ball didn't reach defensive third in 15 s — higher is better"],
                    ["SPE (20s)",  spe_20_str,
                     "% transitions where ball didn't reach defensive third in 20 s — higher is better"],
                    ["% Best",  pct("Best"),  "Possession regained within 5 s"],
                    ["% Good",  pct("Good"),  "Ball out of play or foul won within 15 s"],
                    ["% Okay",  pct("Okay"),  "Foul committed or attack delayed past 15 s"],
                    ["% Bad",   pct("Bad"),   "Shot, penetration, or in-behind pass conceded within 15 s"],
                ],
            ),
            "",
            "### Structural Metrics (mean)\n",
            "> Lower Team Length and Team Compactness = more compact/organised block. "
            "Higher Line Height = more aggressive pressing line position.\n",
            _md_table(
                ["Metric"] + offset_labels,
                [
                    ["Team Length (m)"]
                    + [format_value(col_mean(tdf, f"team_length_m_t{k}")) for k in offset_keys],
                    ["Line Height (m)"]
                    + [format_value(col_mean(tdf, f"line_height_m_t{k}")) for k in offset_keys],
                    ["Players Behind Ball"]
                    + [format_value(col_mean(tdf, f"players_behind_ball_t{k}"), ".1f") for k in offset_keys],
                    ["NumSup RD App1 (Rule-Based)"]
                    + [format_value(col_mean(tdf, f"num_superiority_app1_t{k}"), ".1f") for k in offset_keys],
                    ["NumSup RD App2 (Clustering)"]
                    + [format_value(col_mean(tdf, f"num_superiority_app2_t{k}"), ".1f") for k in offset_keys],
                    ["Team Compactness (m)"]
                    + [format_value(col_mean(tdf, f"team_compactness_t{k}"), ".2f") for k in offset_keys],
                ],
            ),
            "",
            "### Pressing & Escape Pressure (mean)\n",
            "> **Press scale:** 0 = no pressing, 100 = maximum pressing. Δ values show change from t0+1s. Negative Δ = team pressed harder. "
            "> **Escape scale:** 0 = no pressure, 100 = maximum pressure received. Negative Δ = team escaped pressure successfully. "
            "t0 is omitted because the losing team was in possession (press = 0 by definition). "
            "Negative Δ = team is pressing harder later in the transition.\n",
            _md_table(
                ["Metric", "t0+1s", "t0+5s", "t0+10s", "Δ (1s→5s)", "Δ (1s→10s)"],
                [
                    ["Zone Press App1",
                     format_value(col_mean(tdf, "zone_press_app1_t2")),
                     format_value(col_mean(tdf, "zone_press_app1_t10")),
                     format_value(col_mean(tdf, "zone_press_app1_t20")),
                     format_value(col_delta_mean(tdf, "zone_press_app1_t10", "zone_press_app1_t2")),
                     format_value(col_delta_mean(tdf, "zone_press_app1_t20", "zone_press_app1_t2"))],
                    ["Zone Press App2",
                     format_value(col_mean(tdf, "zone_press_app2_t2")),
                     format_value(col_mean(tdf, "zone_press_app2_t10")),
                     format_value(col_mean(tdf, "zone_press_app2_t20")),
                     format_value(col_delta_mean(tdf, "zone_press_app2_t10", "zone_press_app2_t2")),
                     format_value(col_delta_mean(tdf, "zone_press_app2_t20", "zone_press_app2_t2"))],
                    ["Team Press (all players)",
                     format_value(col_mean(tdf, "team_press_t2")),
                     format_value(col_mean(tdf, "team_press_t10")),
                     format_value(col_mean(tdf, "team_press_t20")),
                     format_value(col_delta_mean(tdf, "team_press_t10", "team_press_t2")),
                     format_value(col_delta_mean(tdf, "team_press_t20", "team_press_t2"))],
                    ["Zone Esc.Press (App1)",
                     format_value(col_mean(gdf, "gaining_ps_zone_t2")),
                     format_value(col_mean(gdf, "gaining_ps_zone_t10")),
                     format_value(col_mean(gdf, "gaining_ps_zone_t20")),
                     format_value(col_delta_mean(gdf, "gaining_ps_zone_t10", "gaining_ps_zone_t2")),
                     format_value(col_delta_mean(gdf, "gaining_ps_zone_t20", "gaining_ps_zone_t2"))],
                    ["Team Esc.Press (all players)",
                     format_value(col_mean(gdf, "gaining_ps_mean_t2")),
                     format_value(col_mean(gdf, "gaining_ps_mean_t10")),
                     format_value(col_mean(gdf, "gaining_ps_mean_t20")),
                     format_value(col_delta_mean(gdf, "gaining_ps_mean_t10", "gaining_ps_mean_t2")),
                     format_value(col_delta_mean(gdf, "gaining_ps_mean_t20", "gaining_ps_mean_t2"))],
                ],
            ),
            "",
            "### Transition Dynamics (Positive Transition)\n",
            "> **Positive Transition:** Metrics when team loses possession and must defend. "
            "CAdv = centroid advance (positive = recovering shape). "
            "TeamLen/Compact Δ = change in structural metrics at 5s and 10s.\n",
            _md_table(
                ["Metric", "Value", "Notes"],
                [
                    ["Centroid Advance 5s (m)",
                     format_value(col_mean(tdf, "centroid_advance_5s_m")),
                     "Positive = team centroid moved forward (recovering shape)"],
                    ["Centroid Advance 10s (m)",
                     format_value(col_mean(tdf, "centroid_advance_10s_m")),
                     "Positive = continued advance after 10 s"],
                    ["Team Length Δ 5s (m)",
                     format_value(col_mean(tdf, "team_length_m_t10") - col_mean(tdf, "team_length_m_t0"), ".2f"),
                     "Change in team length from t0 to 5s. Negative = more compact."],
                    ["Team Length Δ 10s (m)",
                     format_value(col_mean(tdf, "team_length_m_t20") - col_mean(tdf, "team_length_m_t0"), ".2f"),
                     "Change in team length from t0 to 10s."],
                    ["Compactness Δ 5s (m)",
                     format_value(col_mean(tdf, "team_compactness_t10") - col_mean(tdf, "team_compactness_t0"), ".2f"),
                     "Change in team compactness from t0 to 5s. Negative = tighter shape."],
                    ["Compactness Δ 10s (m)",
                     format_value(col_mean(tdf, "team_compactness_t20") - col_mean(tdf, "team_compactness_t0"), ".2f"),
                     "Change in team compactness from t0 to 10s. Negative = tighter shape."],
                ],
            ),
            "",
            "### Transition Dynamics (Negative Transition)\n",
            "> **Negative Transition:** Metrics show team's own attacking performance when they gain possession. "
            "ConstrProg/OwnHalfExit/ProdPass/PM Dep = quality of possessions when team has the ball.\n",
            f"> Gaining Team Playmaker (Deep-Lying, composite score): **{pm_str}**\n",
            "",
            _md_table(
                ["Metric", "Value", "Notes"],
                [
                    ["Constructive Progression (PassCount ≥ 3)",
                     f"{format_value(pct_bool(tdf, 'constructive_progression'), '.1f')}%",
                     "% transitions where gaining team made ≥3 passes in 15 s"],
                    ["Own Half Exit (StartX ≤ 500)",
                     f"{format_value(pct_bool(tdf, 'own_half_exit'), '.1f')}%",
                     "% transitions where gaining team had possession in own half within 15 s"],
                    ["Forward Pass Ratio (45°)",
                     f"{format_value(col_mean(tdf, 'productive_pass_ratio_45') * 100, '.1f')}%",
                     "% of gaining team passes in 15 s that are strictly forward (within 45° of attack dir)"],
                    ["Forward Pass Ratio (90°)",
                     f"{format_value(col_mean(tdf, 'productive_pass_ratio_90') * 100, '.1f')}%",
                     "% of gaining team passes in 15 s that are forward or sideways (within 90°)"],
                    ["Playmaker Dep. (1st pass → PM)",
                     f"{format_value(pct_bool(tdf, 'playmaker_dependency_1st'), '.1f')}%",
                     f"1st pass targets playmaker ({pm_str})"],
                    ["Playmaker Dep. (2nd pass → PM)",
                     f"{format_value(pct_bool(tdf, 'playmaker_dependency_2nd'), '.1f')}%",
                     f"2nd pass targets playmaker ({pm_str})"],
                ],
            ),
            "",
        ]

    md_lines.append(_SINGLE_MATCH_GLOSSARY)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"match_{match_id}_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"  Report saved: {md_path}")


# ---------------------------------------------------------------------------
# Multi-match comparison report (reads from CSV, no raw data)
# ---------------------------------------------------------------------------

def _build_comparison_sections(
    df: pd.DataFrame,
    comparison_teams: set[str],
) -> tuple:
    """Build sec1–sec4b DataFrames from a metrics DataFrame."""
    comp_df = df[
        df["losing_team_name"].isin(comparison_teams) |
        df["gaining_team_name"].isin(comparison_teams)
    ].copy()

    team_rows = []
    for team_name in sorted(comparison_teams):
        tdf = comp_df[comp_df["losing_team_name"] == team_name]
        gdf = comp_df[comp_df["gaining_team_name"] == team_name]
        if tdf.empty:
            continue

        n   = len(tdf)
        n_m = tdf["match_id"].nunique() if "match_id" in tdf.columns else "—"
        rts = tdf["transition_rating"] if "transition_rating" in tdf.columns else pd.Series(dtype=str)
        pct = lambda r: f"{(rts == r).sum() / n * 100:.0f}%" if n > 0 else "—"

        spe_15, spe_20 = _spe_from_csv(comp_df, team_name)

        team_rows.append({
            "team":  team_name,
            "n":     n,
            "n_m":   n_m,
            "spe_15": spe_15,
            "spe_20": spe_20,
            "pct_best": pct("Best"),
            "pct_good": pct("Good"),
            "pct_okay": pct("Okay"),
            "pct_bad":  pct("Bad"),
            # Section 2
            "team_len":    format_value(col_mean(tdf, "team_length_m_t0")),
            "line_ht":     format_value(col_mean(tdf, "line_height_m_t0")),
            "behind_ball": format_value(col_mean(tdf, "players_behind_ball_t0"), ".1f"),
            "numsup1":     format_value(col_mean(tdf, "num_superiority_app1_t0"), ".1f"),
            "numsup2":     format_value(col_mean(tdf, "num_superiority_app2_t0"), ".1f"),
            "compact":     format_value(col_mean(tdf, "team_compactness_t0"), ".2f"),
            # Section 3
            "numsup1_5s":  format_value(col_mean(tdf, "num_superiority_app1_t10"), ".1f"),
            "numsup1_10s": format_value(col_mean(tdf, "num_superiority_app1_t20"), ".1f"),
            "numsup2_5s":  format_value(col_mean(tdf, "num_superiority_app2_t10"), ".1f"),
            "numsup2_10s": format_value(col_mean(tdf, "num_superiority_app2_t20"), ".1f"),
            "compact_d5s": format_value(col_mean(tdf, "team_compactness_t10") - col_mean(tdf, "team_compactness_t0"), ".2f"),
            "zp1_1s":  format_value(col_mean(tdf, "zone_press_app1_t2")),
            "zp1_d5":  format_value(pct_delta(tdf, "zone_press_app1", 2, 10, col_mean)),
            "zp1_d10": format_value(pct_delta(tdf, "zone_press_app1", 2, 20, col_mean)),
            "zp2_d5":  format_value(pct_delta(tdf, "zone_press_app2", 2, 10, col_mean)),
            "tp_1s":   format_value(col_mean(tdf, "team_press_t2")),
            "tp_d5":   format_value(pct_delta(tdf, "team_press", 2, 10, col_mean)),
            "tp_d10":  format_value(pct_delta(tdf, "team_press", 2, 20, col_mean)),
            "escz_d5":  format_value(pct_delta(gdf, "gaining_ps_zone", 2, 10, col_mean)),
            "escz_d10": format_value(pct_delta(gdf, "gaining_ps_zone", 2, 20, col_mean)),
            "esct_d5":  format_value(pct_delta(gdf, "gaining_ps_mean", 2, 10, col_mean)),
            "esct_d10": format_value(pct_delta(gdf, "gaining_ps_mean", 2, 20, col_mean)),
            # Section 4a
            "cadv5":        format_value(col_mean(tdf, "centroid_advance_5s_m")),
            "cadv10":       format_value(col_mean(tdf, "centroid_advance_10s_m")),
            "team_len_d5s": format_value(col_mean(tdf, "team_length_m_t10") - col_mean(tdf, "team_length_m_t0"), ".2f"),
            "team_len_d10s":format_value(col_mean(tdf, "team_length_m_t20") - col_mean(tdf, "team_length_m_t0"), ".2f"),
            "compact_d10s": format_value(col_mean(tdf, "team_compactness_t20") - col_mean(tdf, "team_compactness_t0"), ".2f"),
            # Section 4b
            "cp_pct_own":   f"{format_value(pct_bool(gdf, 'constructive_progression'), '.1f')}%" if not gdf.empty else "—",
            "ohe_pct_own":  f"{format_value(pct_bool(gdf, 'own_half_exit'), '.1f')}%" if not gdf.empty else "—",
            "ppr45_pct_own":f"{format_value(col_mean(gdf, 'productive_pass_ratio_45') * 100 if not gdf.empty else float('nan'), '.1f')}%" if not gdf.empty else "—",
            "ppr90_pct_own":f"{format_value(col_mean(gdf, 'productive_pass_ratio_90') * 100 if not gdf.empty else float('nan'), '.1f')}%" if not gdf.empty else "—",
            "pmd1_pct_own": f"{format_value(pct_bool(gdf, 'playmaker_dependency_1st'), '.1f')}%" if not gdf.empty else "—",
            "pmd2_pct_own": f"{format_value(pct_bool(gdf, 'playmaker_dependency_2nd'), '.1f')}%" if not gdf.empty else "—",
        })

    team_rows.sort(
        key=lambda r: r["spe_15"] if not (isinstance(r["spe_15"], float) and np.isnan(r["spe_15"])) else -1,
        reverse=True,
    )

    def _build_sec(rows, field_map):
        return pd.DataFrame([
            {"Team": r["team"], **{hdr: r[key] for hdr, key in field_map.items()}}
            for r in rows
        ])

    sec1 = _build_sec(team_rows, {
        "N": "n", "Matches": "n_m",
        "SPE (15s)": "spe_15", "SPE (20s)": "spe_20",
        "% Best": "pct_best", "% Good": "pct_good",
        "% Okay": "pct_okay", "% Bad": "pct_bad",
    })
    sec1["SPE (15s)"] = sec1["SPE (15s)"].apply(
        lambda v: format(v, ".1f") + "%" if isinstance(v, float) and not np.isnan(v) else "—"
    )
    sec1["SPE (20s)"] = sec1["SPE (20s)"].apply(
        lambda v: format(v, ".1f") + "%" if isinstance(v, float) and not np.isnan(v) else "—"
    )

    sec2 = _build_sec(team_rows, {
        "TeamLen(m)": "team_len", "LineHt(m)": "line_ht", "BehindBall": "behind_ball",
        "NumSup RD App1 (Rule-Based)": "numsup1", "NumSup RD App2 (Clustering)": "numsup2",
        "TeamComp(m)": "compact",
    })
    sec3 = _build_sec(team_rows, {
        "NumSup App1 (5s)": "numsup1_5s", "NumSup App1 (10s)": "numsup1_10s",
        "NumSup App2 (5s)": "numsup2_5s", "NumSup App2 (10s)": "numsup2_10s",
        "CompΔ(5s)": "compact_d5s",
        "ZPress1(t1s)": "zp1_1s", "ZPress1Δ%(5s)": "zp1_d5", "ZPress1Δ%(10s)": "zp1_d10",
        "ZPress2Δ%(5s)": "zp2_d5",
        "TmPress(t1s)": "tp_1s", "TmPressΔ%(5s)": "tp_d5", "TmPressΔ%(10s)": "tp_d10",
        "EscZ-Δ%(5s)": "escz_d5", "EscZ-Δ%(10s)": "escz_d10",
        "EscT-Δ%(5s)": "esct_d5", "EscT-Δ%(10s)": "esct_d10",
    })
    sec4a = _build_sec(team_rows, {
        "CAdv5s(m)": "cadv5", "CAdv10s(m)": "cadv10",
        "TeamLen Δ(5s)": "team_len_d5s", "TeamLen Δ(10s)": "team_len_d10s",
        "Compact Δ(5s)": "compact_d5s", "Compact Δ(10s)": "compact_d10s",
    })
    sec4b = _build_sec(team_rows, {
        "ConstrProg%": "cp_pct_own", "OwnHalfExit%": "ohe_pct_own",
        "ProdPass(45°)%": "ppr45_pct_own", "ProdPass(90°)%": "ppr90_pct_own",
        "PM Dep(1st)%": "pmd1_pct_own", "PM Dep(2nd)%": "pmd2_pct_own",
    })

    return sec1, sec2, sec3, sec4a, sec4b, comp_df


def _write_comparison_md(
    path: Path,
    sec1: pd.DataFrame,
    sec2: pd.DataFrame,
    sec3: pd.DataFrame,
    sec4a: pd.DataFrame,
    sec4b: pd.DataFrame,
) -> None:
    def _df_to_md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
        data_rows = [
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, sep] + data_rows)

    lines = [
        "# Rest Defence — Multi-Match Comparison\n",
        "_Sorted by SPE (Structural Prevention Efficiency, 15-second window), descending._\n",
        "",
        "## Section 1 — Ratings & SPE (15s & 20s windows)\n",
        "_Higher SPE = better at keeping the ball out of the defensive third after losing possession._\n",
        _df_to_md(sec1), "",
        "## Section 2 — Structural Metrics at t0\n",
        "_Lower Team Length and Team Compactness = more compact block. "
        "Higher Line Height = more aggressive position. "
        "Positive Num.Sup = defenders outnumber attackers in zone._\n",
        _df_to_md(sec2), "",
        "## Section 3 — Pressing & Escape Pressure\n",
        "_**Press scale:** 0 = no pressing, 100 = maximum pressing. Δ% reported as percentage change from t0+1s baseline. Positive Δ% = team pressed harder. "
        "**Escape scale:** 0 = no pressure, 100 = maximum pressure. Positive Δ% = more pressure received (not escaping); negative Δ% = pressure relieved._\n",
        _df_to_md(sec3), "",
        "## Section 4a — Transition Dynamics (Positive Transition)\n",
        "_**Positive Transition:** Metrics when team loses possession and must defend. CAdv = centroid advance (positive = recovering shape). "
        "TeamLen/Compact Δ = change in structural metrics at 5s and 10s after possession loss._\n",
        _df_to_md(sec4a), "",
        "## Section 4b — Transition Dynamics (Negative Transition)\n",
        "_**Negative Transition:** Metrics show team's own attacking performance when they gain possession. "
        "ConstrProg/OwnHalfExit/ProdPass/PM Dep = quality of possessions when team has the ball._\n",
        _df_to_md(sec4b), "",
        _MULTI_MATCH_GLOSSARY,
    ]
    path.write_text("\n".join(lines))


def generate_comparison_report(
    df: pd.DataFrame,
    out_dir: Path,
    teams: set[str] | None = None,
) -> None:
    """Generate team_comparison.md and team_comparison.csv from a metrics DataFrame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_teams = teams if teams else (
        set(df["losing_team_name"].unique()) | set(df["gaining_team_name"].unique())
    )

    sec1, sec2, sec3, sec4a, sec4b, comp_df = _build_comparison_sections(df, comparison_teams)

    if comp_df.empty:
        print("  No transitions found for comparison teams.")
        return

    md_path = out_dir / "team_comparison.md"
    _write_comparison_md(md_path, sec1, sec2, sec3, sec4a, sec4b)
    print(f"  Comparison markdown saved: {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reports from a transitions metrics CSV."
    )
    parser.add_argument("--input",      required=True,
                        help="Path to the metrics CSV (produced by main.py)")
    parser.add_argument("--match-id",   type=str, nargs="+", default=None,
                        help="One or more match IDs. One ID → single-match report. "
                             "Multiple IDs → per-match reports + comparison for those matches.")
    parser.add_argument("--teams",      type=str, default=None,
                        help="Comma-separated team names to filter comparison (default: all teams in CSV)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same directory as --input)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent

    if args.match_id and len(args.match_id) == 1:
        # Single match: generate match report only
        generate_match_report(df, args.match_id[0], out_dir)
    elif args.match_id and len(args.match_id) > 1:
        # Multiple matches: per-match reports + comparison restricted to those matches
        match_ids = [str(m) for m in args.match_id]
        for mid in match_ids:
            generate_match_report(df, mid, out_dir)
        filtered = df[df["match_id"].astype(str).isin(match_ids)]
        generate_comparison_report(filtered, out_dir, teams=None)
    else:
        # No match-id: comparison across all (or filtered) teams
        teams = set(t.strip() for t in args.teams.split(",")) if args.teams else None
        generate_comparison_report(df, out_dir, teams=teams)


if __name__ == "__main__":
    main()
