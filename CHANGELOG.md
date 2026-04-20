# Changelog

## [0.0.8] - 20/04/2026

* [feature] Remove match list dependency: team 'a'/'b' labels now derived from raw_df.team_owner column (0=home='a', 1=away='b') via action_id cross-reference when matchesList CSV is absent
* [feature] Optional teams_metadata.csv support for team name/colour overrides when match list unavailable
* [refactor] CSV column cleanup: removed pitch_control, coverage_ratio (all variants), compactness_delta, zone_press_app3, num_superiority_app3, pitch_control_delta, pressure_delta — ~30 columns removed (99→67)
* [refactor] Compactness delta in reports now computed from team_compactness values directly rather than stored separately
* [perf] pitch_control_snapshot() no longer called per-frame in prevention metrics (significant CPU saving)
* [feature] --match-id now accepts multiple IDs (space-separated); automatically generates a cross-match comparison when ≥2 IDs are given
* [feature] --team-id added: resolves all match IDs for a team from action data (works with and without match list), then runs per-match analysis and comparison

## [0.0.7] - 19/04/2026

* [feature] File-based logging: all log output now written to both stdout and <output_dir>/run.log
* [feature] Fatal exceptions captured with full traceback in run.log via try/except in main()
* [refactor] _log() now routes through Python logging module instead of print()

## [0.0.6] - 19/04/2026

* [fix] Lazy imports: moved sklearn, mplsoccer, matplotlib.animation to inside the functions that need them
* [fix] `python main.py --help` now responds in ~0.5s instead of several seconds
* Affected files: main.py, src/rest_defence_area.py, src/metrics/prevention.py, src/visualisation.py, src/video.py

## [0.0.5] - 19/04/2026

* [fix] Round all float columns in exported CSVs to 4 decimal places (was up to 16 digits, e.g. coverage_ratio)
* Applied to all three CSV save sites: single-match summary, multi-match comparison, and --export-csv path

## [0.0.4] - 17/04/2026

* [feature] Press & Escape Pressure metrics: added percentage change (Δ%) reporting from t0+1s baseline for pressing intensity and escape pressure
* [feature] Escape pressure metrics for gaining team: gaining_ps_zone and gaining_ps_mean track pressure received when team gains possession
* [refactor] Section 4 reorganized into Positive Transition (defensive response) and Negative Transition (attacking performance) with clear definitions
* [refactor] Added structural metric deltas (Team Length, Compactness) at 5s and 10s to transition dynamics section
* [fix] Playmaker identification corrected: now shows gaining team's playmaker (not opponent's) in Gaining Team Analysis section
* [fix] Scale descriptions corrected from inverted (0=max, 100=none) to normal (0=none, 100=max) for all pressing metrics
* [refactor] Visualization grid layout adjusted: increased table row height ratio to accommodate expanded metrics (figsize 22x26, height_ratios 2.0:2.0:4.0)
* [feature] Extended time_offsets to include t30 (15s) for complete 15-second window analysis in all prevention metrics

## [0.0.3] - 16/04/2026

* [feature] Constructive Progression Rate refined with event chain inspection (clearances & long passes in first 3 passes)
* [refactor] Improved check_event_chain() to identify structured vs. defensive build-up
* [refactor] Restructure transition_rating() with worst-case conflict resolution logic
* [fix] Implement multi-condition analysis: check all Bad/Good/Okay outcomes before returning
* [fix] Foul detection via action_df: distinguish "Foul suffered" (losing team fouled) from "Foul conceded" (gaining team fouled)
* [fix] Foul suffered with player count analysis: Okay if ≤1 player advantage, Bad if ≥2 players

## [0.0.2] - 16/04/2026

* [refactor] Move utility functions to src/helpers.py
* [feature] Add check_event_chain() helper for event inspection via action_id

## [0.0.1] - 16/04/2026

* First commit: baseline of Transition Control project
* [setup] Configured VERSION, CHANGELOG and automated commit script
