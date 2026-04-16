"""
rest_defence_area.py
--------------------
Three zone definitions for the Rest Defence Area:

  Approach 1 — Rule-Based with KMeans fallback
    Rectangle from last defender → second-third boundary (min 10 m depth),
    but replaced by Approach 2 zone if that is larger.

  Approach 2 — Pure KMeans (rearmost cluster only)
    1D k-means (k ∈ {2,3,4}, best by silhouette) on losing team outfield x.
    Zone = bounding box of rearmost cluster + buffer.

  Approach 3 — Adaptive KMeans (conditional second line)
    Same as Approach 2, but if the rearmost cluster is still in the team's own
    half, the second-rearmost cluster is included in the zone too.

All coordinates are absolute centimetres (origin at pitch centre).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .data_loading import (
    PITCH_HALF_LENGTH_CM,
    PITCH_HALF_WIDTH_CM,
    THIRD_BOUNDARY_CM,
    get_player_positions,
    last_defender_x,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KMEANS_BUFFER_CM         = 500       # fallback symmetric buffer (cm)
KMEANS_K_OPTIONS         = (3, 4)   # losing team: k=2 excluded (always ≥3 lines)
KMEANS_K_OPTIONS_GAINING = (3, 4) # gaining team: k=2 excluded (always ≥3 lines)
KMEANS_RANDOM_STATE      = 42
MIN_ZONE_DEPTH_CM        = 1000     # minimum rectangle longitudinal depth (10 m)


# ---------------------------------------------------------------------------
# RestDefenceZone value object
# ---------------------------------------------------------------------------

@dataclass
class RestDefenceZone:
    """
    Axis-aligned rectangle in absolute cm coordinates.

    x_min, x_max : longitudinal extent
    y_min, y_max : lateral extent (full pitch width by default)
    method       : 'app1' | 'app2' | 'app3'
    """
    x_min: float
    x_max: float
    y_min: float = -PITCH_HALF_WIDTH_CM
    y_max: float =  PITCH_HALF_WIDTH_CM
    method: str  = "app1"

    def contains(self, x: float, y: float) -> bool:
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def contains_array(self, positions: np.ndarray) -> np.ndarray:
        if len(positions) == 0:
            return np.array([], dtype=bool)
        xs, ys = positions[:, 0], positions[:, 1]
        return (
            (xs >= self.x_min) & (xs <= self.x_max)
            & (ys >= self.y_min) & (ys <= self.y_max)
        )

    def area_m2(self) -> float:
        return (abs(self.x_max - self.x_min) / 100) * (abs(self.y_max - self.y_min) / 100)

    def overlap_coefficient(self, other: "RestDefenceZone") -> float:
        ix_min = max(self.x_min, other.x_min)
        ix_max = min(self.x_max, other.x_max)
        iy_min = max(self.y_min, other.y_min)
        iy_max = min(self.y_max, other.y_max)
        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0
        intersection = (ix_max - ix_min) * (iy_max - iy_min)
        a1 = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        a2 = (other.x_max - other.x_min) * (other.y_max - other.y_min)
        union = a1 + a2 - intersection
        return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Internal: 1D KMeans on x-positions
# ---------------------------------------------------------------------------

def _kmeans_1d(
    positions: np.ndarray,             # shape (N, 2) absolute cm
    k_options: tuple[int, ...] = KMEANS_K_OPTIONS,
) -> tuple[np.ndarray, int]:
    """
    Run 1D k-means on x-coordinates, choose best k by silhouette score.
    Returns (labels, best_k).
    Falls back to all-zeros if fewer than 3 players or silhouette fails.
    """
    if len(positions) < 3:
        return np.zeros(len(positions), dtype=int), 1

    xs = positions[:, 0].reshape(-1, 1)
    best_k, best_score, best_labels = k_options[0], -1.0, None

    for k in k_options:
        if k >= len(positions):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
            labels = km.fit_predict(xs)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(xs, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels.copy()

    if best_labels is None:
        best_labels = np.zeros(len(positions), dtype=int)
    return best_labels, best_k


def _sorted_cluster_ids_by_depth(
    positions: np.ndarray,
    labels: np.ndarray,
    attacks_right: bool,
) -> list[int]:
    """
    Return cluster ids sorted from rearmost (closest to own goal) to frontmost.
    attacks_right=True → own goal at low x → rearmost = lowest centroid x.
    """
    cluster_ids = list(np.unique(labels))
    centroids_x = {cid: float(positions[labels == cid, 0].mean()) for cid in cluster_ids}
    return sorted(cluster_ids, key=lambda c: centroids_x[c], reverse=not attacks_right)


def _offside_aware_buffer_cm(
    last_def_x: float,
    attacks_right: bool,
) -> tuple[float, float]:
    """
    Return (forward_cm, back_cm) based on where the most rearward player of
    the rearmost cluster sits relative to the halfway line.

    The reference is the last defender's position — not the cluster centroid —
    because offside is judged from the most rearward defending player.

    attacks_right=True  → own goal x=−5250, forward = +x direction
    attacks_right=False → own goal x=+5250, forward = −x direction

    Tiers:
      In own half (last_def_x < 0 if ar, > 0 if not ar):
          forward=1000, back=0  — full 10 m depth forward
      0–5 m past halfway:
          back = dist_past; forward = 1000 − dist_past
      > 5 m past halfway:
          forward=500, back=500  — standard symmetric
    """
    dist_past_cm = max(0.0, last_def_x if attacks_right else -last_def_x)
    back_cm    = min(dist_past_cm, 500.0)
    forward_cm = 1000.0 - back_cm
    return forward_cm, back_cm


def _bbox_from_positions(
    positions: np.ndarray,
    forward_cm: float = KMEANS_BUFFER_CM,
    back_cm: float    = KMEANS_BUFFER_CM,
    attacks_right: bool = True,
) -> tuple[float, float]:
    """
    Return (x_min, x_max) with asymmetric forward/back buffers, clipped to pitch.

    forward_cm : buffer toward opponent goal
    back_cm    : buffer toward own goal
    """
    x_min = float(positions[:, 0].min())
    x_max = float(positions[:, 0].max())
    if attacks_right:
        x_min -= back_cm
        x_max += forward_cm
    else:
        x_min -= forward_cm
        x_max += back_cm
    return max(x_min, -PITCH_HALF_LENGTH_CM), min(x_max, PITCH_HALF_LENGTH_CM)


# ---------------------------------------------------------------------------
# Approach 2 — Pure KMeans (rearmost cluster, offside-aware buffer)
# ---------------------------------------------------------------------------

def zone_approach2(
    frame_row,
    losing_team: str,
    team_a_attacks_right: bool,
) -> tuple[RestDefenceZone, int, np.ndarray]:
    """
    1D k-means on losing team outfield x. Zone = rearmost cluster bbox with
    offside-aware asymmetric buffer (reference = most rearward player).

    Returns
    -------
    zone         : RestDefenceZone (method='app2')
    best_k       : chosen k
    losing_labels: cluster label per outfield player (for visualisation)
    """
    positions = get_player_positions(frame_row, losing_team, include_gk=False)
    attacks_right = (losing_team == "a") == team_a_attacks_right

    labels, best_k = _kmeans_1d(positions)
    sorted_ids = _sorted_cluster_ids_by_depth(positions, labels, attacks_right)
    rearmost_id = sorted_ids[0]
    cluster_pos = positions[labels == rearmost_id]

    # Use the most rearward player position as the offside reference
    last_def_x = (float(cluster_pos[:, 0].min()) if attacks_right
                  else float(cluster_pos[:, 0].max()))
    forward_cm, back_cm = _offside_aware_buffer_cm(last_def_x, attacks_right)
    x_min, x_max = _bbox_from_positions(cluster_pos, forward_cm, back_cm, attacks_right)

    return (
        RestDefenceZone(x_min=x_min, x_max=x_max, method="app2"),
        best_k,
        labels,
    )


# ---------------------------------------------------------------------------
# Approach 3 — Adaptive KMeans (conditional second line)
# DISABLED: commented out for now; zone_app3 = zone_app2 in build_zones().
# Re-enable by removing comments and restoring the call in build_zones.
# ---------------------------------------------------------------------------

# def zone_approach3(
#     positions: np.ndarray,
#     labels: np.ndarray,
#     losing_team: str,
#     team_a_attacks_right: bool,
# ) -> RestDefenceZone:
#     """
#     Reuses already-computed 1D k-means labels from Approach 2.
#     If the rearmost cluster centroid is in the team's own half, include the
#     second-rearmost cluster too (back 4 + midfield block scenario).
#     """
#     attacks_right = (losing_team == "a") == team_a_attacks_right
#     sorted_ids = _sorted_cluster_ids_by_depth(positions, labels, attacks_right)
#     rearmost_id = sorted_ids[0]
#     rearmost_centroid_x = float(positions[labels == rearmost_id, 0].mean())
#     in_own_half = (
#         (attacks_right and rearmost_centroid_x < 0)
#         or (not attacks_right and rearmost_centroid_x > 0)
#     )
#     if in_own_half and len(sorted_ids) >= 2:
#         second_id = sorted_ids[1]
#         included = positions[np.isin(labels, [rearmost_id, second_id])]
#     else:
#         included = positions[labels == rearmost_id]
#     last_def_x = (float(included[:, 0].min()) if attacks_right
#                   else float(included[:, 0].max()))
#     forward_cm, back_cm = _offside_aware_buffer_cm(last_def_x, attacks_right)
#     x_min, x_max = _bbox_from_positions(included, forward_cm, back_cm, attacks_right)
#     return RestDefenceZone(x_min=x_min, x_max=x_max, method="app3")


# ---------------------------------------------------------------------------
# Approach 1 — Rule-Based Rectangle with KMeans fallback
# ---------------------------------------------------------------------------

def zone_approach1(
    frame_row,
    losing_team: str,
    team_a_attacks_right: bool,
    kmeans_zone: RestDefenceZone | None = None,
) -> RestDefenceZone:
    """
    Rectangle from last defender → second-third boundary, min 10 m depth.
    Falls back to KMeans zone if that is larger.

    Parameters
    ----------
    kmeans_zone : pre-computed Approach 2 zone (passed in to avoid recomputation)
    """
    attacks_right = (losing_team == "a") == team_a_attacks_right
    ld_x = last_defender_x(frame_row, losing_team, team_a_attacks_right)

    if attacks_right:
        # Own goal at x = -5250; zone spans [x_min, +1750]
        x_min = min(ld_x, THIRD_BOUNDARY_CM - MIN_ZONE_DEPTH_CM)
        x_max = THIRD_BOUNDARY_CM
    else:
        # Own goal at x = +5250; zone spans [-1750, x_max]
        x_min = -THIRD_BOUNDARY_CM
        x_max = max(ld_x, -THIRD_BOUNDARY_CM + MIN_ZONE_DEPTH_CM)

    rect = RestDefenceZone(x_min=x_min, x_max=x_max, method="app1")

    # Use KMeans zone if it covers more area (fallback)
    if kmeans_zone is not None and kmeans_zone.area_m2() > rect.area_m2():
        return RestDefenceZone(
            x_min=kmeans_zone.x_min, x_max=kmeans_zone.x_max, method="app1"
        )
    return rect


# ---------------------------------------------------------------------------
# Gaining team cluster (visualisation / future Approach 4)
# ---------------------------------------------------------------------------

def cluster_gaining_team(
    frame_row,
    gaining_team: str,
    k_options: tuple[int, ...] = KMEANS_K_OPTIONS_GAINING,
) -> np.ndarray:
    """
    1D k-means on gaining team outfield x-positions.
    Returns labels array (shape N outfield players).
    """
    positions = get_player_positions(frame_row, gaining_team, include_gk=False)
    if len(positions) < 3:
        return np.zeros(max(len(positions), 0), dtype=int)
    labels, _ = _kmeans_1d(positions, k_options=k_options)
    return labels


# ---------------------------------------------------------------------------
# Main dispatcher — called at each time offset
# ---------------------------------------------------------------------------

def build_zones(
    frame_row,
    losing_team: str,
    team_a_attacks_right: bool,
) -> tuple[RestDefenceZone, RestDefenceZone, RestDefenceZone, int, np.ndarray, np.ndarray]:
    """
    Build all three zones for a single frame.

    Returns
    -------
    zone_app1     : Rule-Based (rect or KMeans fallback)
    zone_app2     : Pure KMeans (rearmost cluster)
    zone_app3     : Adaptive KMeans (conditional second line)
    best_k        : k chosen for losing team clustering
    losing_labels : 1D cluster labels for losing team outfield players
    gaining_labels: 1D cluster labels for gaining team outfield players
    """
    gaining_team = "b" if losing_team == "a" else "a"

    # Approach 2: pure k-means with offside-aware buffer
    zone_app2, best_k, losing_labels = zone_approach2(
        frame_row, losing_team, team_a_attacks_right
    )

    # Approach 3 is disabled — use App2 as placeholder (zone_approach3 commented out above)
    zone_app3 = zone_app2

    # Approach 1: rule-based rectangle with App2 as fallback if larger
    zone_app1 = zone_approach1(
        frame_row, losing_team, team_a_attacks_right, kmeans_zone=zone_app2
    )

    gaining_labels = cluster_gaining_team(frame_row, gaining_team)

    return zone_app1, zone_app2, zone_app3, best_k, losing_labels, gaining_labels
