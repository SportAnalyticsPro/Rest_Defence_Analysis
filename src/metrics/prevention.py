"""
metrics/prevention.py
---------------------
Structural / prevention metrics 1-7, computed at t0, t0+1s, t0+5s, t0+10s.
Zones are recomputed fresh at each time offset (players move).

All positions are in absolute centimetres (origin at pitch centre).

Metric 5 — Team Compactness:
  Mean Euclidean distance (m) of each outfield player from the team centre.
  Lower = more compact; higher = more spread out.

Metric 7 — Pressing Intensity (replaces old Pressure Snapshot/Window):
  zone_press_app{1,2,3} : mean p_{losing_team}_{i} for defenders inside zone
  team_press : mean p_{losing_team}_{i} for ALL outfield defenders
  p_ column scale: 0 = no pressing, 100 = maximum pressing intensity.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data_loading import (
    PITCH_HALF_LENGTH_CM,
    get_player_positions,
    get_frame,
)
from ..rest_defence_area import (
    RestDefenceZone,
    build_zones,
    KMEANS_K_OPTIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attacks_right(losing_team: str, team_a_attacks_right: bool) -> bool:
    return (losing_team == "a") == team_a_attacks_right


# ---------------------------------------------------------------------------
# Metric 1 — Team Length
# ---------------------------------------------------------------------------

def team_length_cm(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("nan")
    return float(positions[:, 0].max() - positions[:, 0].min())


# ---------------------------------------------------------------------------
# Metric 2 — Rest Defence Line Height
# ---------------------------------------------------------------------------

def rest_defence_line_height_cm(
    positions: np.ndarray,
    losing_team: str,
    team_a_attacks_right: bool,
    k_options: tuple[int, ...] = KMEANS_K_OPTIONS,
) -> float:
    """
    1D k-means (best k by silhouette) on outfield x.
    Height = distance from rearmost cluster centroid to own goal line.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(positions) < 3:
        return float("nan")

    ar = _attacks_right(losing_team, team_a_attacks_right)
    own_goal_x = -PITCH_HALF_LENGTH_CM if ar else PITCH_HALF_LENGTH_CM

    xs = positions[:, 0].reshape(-1, 1)
    best_k, best_score, best_labels = k_options[0], -1.0, None

    for k in k_options:
        if k >= len(positions):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(xs)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(xs, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels.copy()

    if best_labels is None:
        rearmost_x = xs.min() if ar else xs.max()
        return float(abs(float(rearmost_x) - own_goal_x))

    cluster_ids = np.unique(best_labels)
    centroids_x = {cid: float(positions[best_labels == cid, 0].mean()) for cid in cluster_ids}
    rearmost_centroid_x = min(centroids_x.values()) if ar else max(centroids_x.values())
    return float(abs(rearmost_centroid_x - own_goal_x))


# ---------------------------------------------------------------------------
# Metric 3 — Players Behind Ball Line
# ---------------------------------------------------------------------------

def players_behind_ball_line(
    positions: np.ndarray,
    ball_x: float,
    losing_team: str,
    team_a_attacks_right: bool,
) -> int:
    if len(positions) == 0:
        return 0
    ar = _attacks_right(losing_team, team_a_attacks_right)
    xs = positions[:, 0]
    return int(np.sum(xs < ball_x) if ar else np.sum(xs > ball_x))


# ---------------------------------------------------------------------------
# Metric 4 — Numerical Superiority in Zone
# ---------------------------------------------------------------------------

def numerical_superiority(
    zone: RestDefenceZone,
    losing_positions: np.ndarray,
    gaining_positions: np.ndarray,
) -> int:
    defenders_in = int(np.sum(zone.contains_array(losing_positions))) if len(losing_positions) else 0
    attackers_in = int(np.sum(zone.contains_array(gaining_positions))) if len(gaining_positions) else 0
    return defenders_in - attackers_in


# ---------------------------------------------------------------------------
# Metric 5 — Team Compactness
# ---------------------------------------------------------------------------

def team_compactness(positions: np.ndarray) -> float:
    """
    Mean Euclidean distance (m) of each outfield player from the team centre.
    Team centre = mean (x, y) of all outfield players.
    Lower = more compact; higher = more spread out.

    Platform definition: standard deviation of player distribution on the field —
    implemented here as mean distance from team centre (equivalent interpretation).

    positions: (N, 2) array in absolute cm (outfield players only, no GK).
    """
    if len(positions) < 2:
        return float("nan")
    centre = positions.mean(axis=0)
    distances = np.sqrt(np.sum((positions - centre) ** 2, axis=1))
    return float(distances.mean() / 100)   # cm → m


# ---------------------------------------------------------------------------
# Metric 6 — Pitch Control Snapshot
# ---------------------------------------------------------------------------

def pitch_control_snapshot(
    frame_row: pd.Series,
    losing_team: str,
    zone: RestDefenceZone | None = None,
) -> dict:
    gaining_team = "b" if losing_team == "a" else "a"

    def _team_cov(team: str) -> float:
        return sum(
            float(frame_row.get(f"c_{team}_{i}", 0) or 0) for i in range(1, 12)
        )

    c_l = _team_cov(losing_team)
    c_g = _team_cov(gaining_team)
    total = c_l + c_g
    coverage_ratio = c_l / total if total > 0 else float("nan")

    pitch_control_zone = float("nan")
    coverage_ratio_zone = float("nan")
    if zone is not None:
        def _team_zone_cov(team: str) -> float:
            cov = 0.0
            for i in range(1, 12):
                x = frame_row.get(f"x_{team}_{i}")
                y = frame_row.get(f"y_{team}_{i}")
                c = frame_row.get(f"c_{team}_{i}", 0)
                if pd.notna(x) and pd.notna(y) and pd.notna(c):
                    if zone.contains(float(x), float(y)):
                        cov += float(c)
            return cov

        cz_l, cz_g = _team_zone_cov(losing_team), _team_zone_cov(gaining_team)
        pitch_control_zone = cz_l   # raw zone coverage
        zt = cz_l + cz_g
        if zt > 0:
            coverage_ratio_zone = cz_l / zt

    return {
        "pitch_control":      c_l,             # raw absolute coverage
        "coverage_ratio":     coverage_ratio,  # relative to opponent
        "pitch_control_zone": pitch_control_zone,
        "coverage_ratio_zone": coverage_ratio_zone,
    }


# ---------------------------------------------------------------------------
# Metric 7 — Pressing Intensity (zone and team level)
# ---------------------------------------------------------------------------

def zone_press_intensity(
    frame_row: pd.Series,
    losing_team: str,
    zone: RestDefenceZone,
) -> float:
    """
    Mean p_{losing_team}_{i} for losing-team outfield players inside zone.
    Scale: 0 = no pressing, 100 = maximum pressing intensity.
    Returns NaN if no qualifying players inside zone.
    """
    values = []
    for i in range(2, 12):   # outfield only (skip GK slot 1)
        x = frame_row.get(f"x_{losing_team}_{i}")
        y = frame_row.get(f"y_{losing_team}_{i}")
        p = frame_row.get(f"p_{losing_team}_{i}")
        if pd.notna(x) and pd.notna(y) and pd.notna(p):
            if zone.contains(float(x), float(y)):
                values.append(float(p))
    return float(np.mean(values)) if values else float("nan")


def team_press_intensity(
    frame_row: pd.Series,
    losing_team: str,
) -> float:
    """
    Mean p_{losing_team}_{i} for ALL outfield players of the losing team.
    Tracks overall team pressing effort regardless of zone position.
    Scale: 0 = no pressing, 100 = maximum pressing intensity.
    """
    values = []
    for i in range(2, 12):
        p = frame_row.get(f"p_{losing_team}_{i}")
        if pd.notna(p):
            values.append(float(p))
    return float(np.mean(values)) if values else float("nan")


def gaining_zone_escape_pressure(
    frame_row: pd.Series,
    gaining_team: str,
    zone: RestDefenceZone,
) -> float:
    """
    Mean ps_{gaining_team}_{i} for gaining-team outfield players inside zone.
    Scale: 100 = maximum pressure received, 0 = no pressure.
    Returns NaN if no qualifying players inside zone.
    """
    values = []
    for i in range(2, 12):
        x = frame_row.get(f"x_{gaining_team}_{i}")
        y = frame_row.get(f"y_{gaining_team}_{i}")
        ps = frame_row.get(f"ps_{gaining_team}_{i}")
        if pd.notna(x) and pd.notna(y) and pd.notna(ps):
            if zone.contains(float(x), float(y)):
                values.append(float(ps))
    return float(np.mean(values)) if values else float("nan")


def gaining_team_escape_pressure(
    frame_row: pd.Series,
    gaining_team: str,
) -> float:
    """
    Mean ps_{gaining_team}_{i} across ALL outfield players of the gaining team.
    Scale: 100 = maximum pressure received, 0 = no pressure.
    """
    values = []
    for i in range(2, 12):
        ps = frame_row.get(f"ps_{gaining_team}_{i}")
        if pd.notna(ps):
            values.append(float(ps))
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_prevention_metrics(
    transition_row: pd.Series,
    raw_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    losing_team_label: str,
    time_offsets: tuple[int, ...] = (0, 2, 10, 20, 30),
) -> dict:
    """
    Compute prevention metrics 1-7 at each time offset.
    Zones are recomputed fresh at every offset.

    Returns
    -------
    Nested dict: { offset_frames: { metric_name: value } }
    Spread deltas are stored at the offset where they apply.
    """
    match_id     = transition_row["match_id"]
    period       = int(transition_row["period"])
    losing_team  = losing_team_label
    gaining_team = "b" if losing_team == "a" else "a"
    t0_frame     = int(transition_row["t0_frame"])

    dir_row = direction_df.loc[(str(match_id), period)]
    team_a_attacks_right = bool(dir_row["team_a_attacks_right"])

    ar = (losing_team == "a") == team_a_attacks_right
    own_goal_x_cm = -PITCH_HALF_LENGTH_CM if ar else PITCH_HALF_LENGTH_CM

    results: dict[int, dict] = {}

    # First pass: compute per-offset metrics

    for offset in time_offsets:
        frame_num = t0_frame + offset
        frame_row = get_frame(raw_df, match_id, frame_num)

        if frame_row is None:
            results[offset] = _nan_prevention_metrics()
            continue

        ball_x = float(frame_row.get("x_ball", float("nan")) or float("nan"))

        losing_pos  = get_player_positions(frame_row, losing_team,  include_gk=False)
        gaining_pos = get_player_positions(frame_row, gaining_team, include_gk=False)

        # Recompute zones at this offset
        zone_app1, zone_app2, zone_app3, best_k, _, _ = build_zones(
            frame_row, losing_team, team_a_attacks_right
        )

        # Team compactness (entire team, not zone-filtered)
        compactness = team_compactness(losing_pos)

        centroid_x_norm = (
            abs(float(losing_pos[:, 0].mean()) - own_goal_x_cm) / 100
            if len(losing_pos) > 0 else float("nan")
        )

        results[offset] = {
            # Metric 1
            "team_length_m": team_length_cm(losing_pos) / 100,
            # Metric 2
            "line_height_m": rest_defence_line_height_cm(
                losing_pos, losing_team, team_a_attacks_right
            ) / 100,
            # Metric 3
            "players_behind_ball": players_behind_ball_line(
                losing_pos, ball_x, losing_team, team_a_attacks_right
            ),
            # Metric 4
            "num_superiority_app1": numerical_superiority(zone_app1, losing_pos, gaining_pos),
            "num_superiority_app2": numerical_superiority(zone_app2, losing_pos, gaining_pos),
            # Metric 5 — team compactness + centroid position
            "team_compactness":    compactness,
            "team_centroid_x_norm": centroid_x_norm,
            # Metric 7 — pressing intensity
            "zone_press_app1": zone_press_intensity(frame_row, losing_team, zone_app1),
            "zone_press_app2": zone_press_intensity(frame_row, losing_team, zone_app2),
            "team_press": team_press_intensity(frame_row, losing_team),
            # Metric 8 — escape pressure (gaining team under pressure)
            "gaining_ps_zone": gaining_zone_escape_pressure(frame_row, gaining_team, zone_app1),
            "gaining_ps_mean": gaining_team_escape_pressure(frame_row, gaining_team),
        }

    return results


def _nan_prevention_metrics() -> dict:
    return {
        "team_length_m":        float("nan"),
        "line_height_m":        float("nan"),
        "players_behind_ball":  float("nan"),
        "num_superiority_app1": float("nan"),
        "num_superiority_app2": float("nan"),
        "team_compactness":      float("nan"),
        "team_centroid_x_norm":  float("nan"),
        "zone_press_app1":       float("nan"),
        "zone_press_app2":      float("nan"),
        "team_press":           float("nan"),
        "gaining_ps_zone":      float("nan"),
        "gaining_ps_mean":      float("nan"),
    }
