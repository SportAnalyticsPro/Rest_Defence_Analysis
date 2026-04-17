# Changelog

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
