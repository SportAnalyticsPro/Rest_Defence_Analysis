# Changelog

## [0.1.1] - 20/04/2026

* [feature] Foul analysis: 6 new per-transition CSV columns — foul_committed, foul_time_s, foul_x_m, foul_defenders_behind_ball, foul_attackers_behind_ball, foul_superiority_rating
* [feature] foul_x_m converted from Wyscout 0-1000 to metres (0=own goal, 105=opponent goal) for human-readable pitch location
* [feature] Foul detection has no time limit — captures fouls until the losing team naturally regains possession (previously capped at 15 s)
* [feature] _compute_foul_context() extracted as shared helper used by both transition_rating() and compute_transition_metrics()
* [feature] Foul Analysis section added to per-match MD report (via report_generator.py) per defending team
* [feature] Section 5 — Foul Analysis added to multi-match comparison MD and console output
* [feature] gaining_team_playmaker_id and gaining_team_playmaker_jersey stored in CSV so report_generator.py can display full playmaker string (#N player_id=X) without raw data
* [fix] Playmaker name/jersey now resolves correctly in standalone report_generator.py from CSV-only input

## [0.1.0] - 20/04/2026

* [feature] New report_generator.py: standalone report engine that reads only from the metrics CSV — no raw tracking data required
* [feature] report_generator.py generates single-match MD reports (generate_match_report) and multi-match comparison reports (generate_comparison_report) with full glossaries
* [feature] SPE now computed from per-transition CSV flags (ball_reached_third_15s, has_15s_window) in report_generator — eliminates raw data dependency in all report generation
* [refactor] main.py: _save_match_summary() reduced to CSV-save + delegate to report_generator; multi_match_comparison() delegates MD generation to report_generator
* [feature] multi_match_comparison() now saves all_transitions.csv to output root (replaces team_comparison.csv as the primary data export)
* [refactor] Removed _md_table(), _SINGLE_MATCH_GLOSSARY, _MULTI_MATCH_GLOSSARY, _write_comparison_md() from main.py — all live in report_generator.py
* [refactor] Moved jersey_str() to src/helpers.py so both main.py and report_generator.py can import it
* [fix] _print_match_summary() no longer requires raw_df/lmap/direction_df — SPE computed from metrics_df CSV flags directly

## [0.0.9] - 20/04/2026

* [feature] Add ball_reached_third_15s and ball_reached_third_20s per-transition SPE flags to CSV — report_generator can now compute SPE as a simple column mean without accessing raw data
* [feature] Add has_15s_window and has_20s_window flags to correctly exclude end-of-half transitions from SPE computation
* [feature] Add losing_team_attacks_right boolean to CSV for direction-aware chart rendering
* [feature] Add team_centroid_x_norm at t0, t0+1s, t0+5s, t0+10s — distance of losing team centroid from own goal (metres, direction-independent) — needed for pitch block plots and evolution charts
* [fix] SPE flags and centroid columns now computed in both visualise_match() and compute_all_metrics() paths, ensuring CSV is complete regardless of execution mode

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
