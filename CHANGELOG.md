# Changelog

## [0.0.3] - 16/04/2026

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
