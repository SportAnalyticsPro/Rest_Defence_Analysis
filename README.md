# Transition Control

A Python analytics pipeline for detecting and analysing **rest-defence transitions** in football tracking data. It computes structural, spatial, and pressing metrics for every transition event across a full season, and produces a branded multi-page PDF report.

---

## Project Structure

```
transition_control/
├── main.py                     # Data extraction engine
├── src/
│   ├── pdf_report.py           # PDF report generator
│   ├── logos.py                # Team logo loader
│   ├── images/
│   │   ├── SportAnalytics-logo.png
│   │   └── logos/              # 20 Serie A team logos (256×256)
│   ├── visualizations/         # Chart functions (bar, scatter, pizza, table …)
│   ├── data_loading.py
│   ├── transition_detection.py
│   ├── rest_defence_area.py
│   └── helpers.py
├── data/                       # Raw input files (not committed)
├── output/                     # Generated CSVs and PDFs
└── legacy/
    └── report_generator.py     # Deprecated MD-based report (superseded by pdf_report.py)
```

---

## Requirements

Python 3.11+. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 1 — Input Data

Place the following files in the `data/` folder before running the pipeline:

| File | Description |
|---|---|
| `ih_raw_data.csv` | Frame-by-frame tracking data (player positions at every frame) |
| `ih_action_data.csv` | Event actions (passes, shots, duels, …) with frame references |
| `ih_events.csv` | High-level match events |
| `matchesList_analisi_transizioni.csv` | Match metadata: IDs, team IDs, date, competition |

> **Note:** The pipeline also accepts a pre-computed transitions CSV (`out_rest_defence_*.csv`) + a meta JSON (`out_rest_defence_*_meta.json`) to skip re-extraction and go directly to the PDF report.

---

## Step 2 — Extract Transition Metrics (`main.py`)

`main.py` reads the raw tracking data, detects all final-third ball-loss transitions, and computes ~100 structural, pressing, and attacking metrics per event.

### Common usage

```bash
# Full season — outputs all_transitions.csv for all available matches
python main.py --output-dir output/season/

# Specific matches
python main.py --match-id 7418 7754 --output-dir output/

# Single team — all their matches
python main.py --team-id 95 --output-dir output/

# Generate per-transition PNGs alongside the CSV
python main.py --match-id 7418 --output-dir output/

# Generate MP4 videos instead of PNGs
python main.py --match-id 7418 --video --output-dir output/

# Limit to first N transitions (useful for quick checks)
python main.py --match-id 7418 --n 5 --output-dir output/
```

### Outputs

| File | Description |
|---|---|
| `output/<dir>/all_transitions.csv` | One row per transition, all metrics |
| `output/<dir>/<match_id>_transitions.csv` | Per-match subset |
| `output/<dir>/transition_<id>.png` | Optional per-transition pitch frame |

### Key metric groups

| Group | Columns | Description |
|---|---|---|
| Structural (t0…t150) | `team_length_m_t*`, `line_height_m_t*`, `team_compactness_t*`, … | Defending team shape at 5 time offsets |
| Pressing | `n_pressing_team_t*`, `n_pressing_zone_*` | Pressing player counts |
| SPE | `ball_reached_third_15s`, `has_15s_window`, … | Whether the opponent reached the defensive third |
| Transition rating | `transition_rating` | Best / Good / Okay / Bad |
| Fouls | `foul_committed`, `foul_time_s`, `foul_x_m`, `foul_superiority_rating` | Foul context during the transition |
| Attacking | `constructive_progression`, `own_half_exit`, `productive_pass_ratio_45`, … | Gaining team attack quality |

> **Time offsets:** `t0` = transition start, `t10` = +1 s, `t50` = +5 s, `t100` = +10 s, `t150` = +15 s (at 10 fps tracking data).

---

## Step 3 — Generate the PDF Report (`src/pdf_report.py`)

The PDF report takes the transitions CSV (and optionally the meta JSON for cover-page dates) and produces a **10-page A4 portrait report** focused on a chosen team.

### Usage

```bash
# With pre-computed final_data (semicolon-delimited, team IDs resolved via meta JSON)
python3 -m src.pdf_report \
    --csv  output/final_data/out_rest_defence_20260421.csv \
    --meta output/final_data/out_rest_defence_20260421_meta.json \
    --output output/report.pdf

# With a comma-delimited all_transitions.csv (team names already resolved)
python3 -m src.pdf_report \
    --csv output/season/all_transitions.csv \
    --output output/report.pdf

# Change the focus team (default: Juventus)
python3 -m src.pdf_report \
    --csv  output/final_data/out_rest_defence_20260421.csv \
    --meta output/final_data/out_rest_defence_20260421_meta.json \
    --output output/report.pdf \
    --focus-team "Inter Milan"
```

> The script **auto-detects** the CSV separator (`,` or `;`) and **resolves `Team_ID` placeholders** to real team names using the meta JSON when provided.

### Report pages

| Page | Content |
|---|---|
| 1 | Cover — logo, title, date range |
| 2 | Ball Regain Effectiveness — rating distribution, SPE 20s bar, defending pizza (focus team) |
| 3 | Team Structure & Organization — scatter plots, centroid evolution line chart |
| 4 | Foul-Based Transition Control — foul table, foul scatter, pressing evolution |
| 5 | Attacking Transition Profile — attacking metrics table, attacking pizza (focus team) |
| 6–9 | Rankings — all teams ranked per metric (one page per metric group) |
| last | Glossary — definition of every metric used |

---

## Adding a New Team Logo

1. Add a `256x256` PNG to `src/images/logos/` named `<slug>.football-logos.cc.png`
2. If the team name doesn't convert cleanly to a slug (e.g. spaces → hyphens), add an entry to `_NAME_TO_SLUG` in `src/logos.py`:

```python
_NAME_TO_SLUG = {
    "as roma":   "roma",      # "as-roma" would be wrong
    "inter milan": "inter",
    ...
}
```

---

## Notes

- `legacy/report_generator.py` — an older Markdown-based report tool, superseded by the PDF generator. Kept for reference.
- `plot_animation.py` — standalone script for rendering transition animations to video.
