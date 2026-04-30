"""
baselines.py — Phase 2: Baseline Policies

Two reference policies for evaluating the Q-learning agent against
simple heuristics.  Both implement a select_action(state) → int interface
that is compatible with CityGridSimulator.step().

GreedyNearestNeighborPolicy
    Always moves toward the closest undelivered dropoff by straight-line
    distance.  When battery drops below BATTERY_THRESHOLD (20 %), diverts
    to the nearest charging station first.

RandomPolicy
    Picks a uniformly random valid action at each timestep (including stay).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from simulator import DIRECTIONS, STAY


# ---------------------------------------------------------------------------
# Shared geometry helper
# ---------------------------------------------------------------------------

def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _best_valid_direction(
    pos: Tuple[int, int],
    target: Tuple[int, int],
    state: Dict[str, Any],
) -> int:
    """Return the action index that moves *pos* most directly toward *target*
    while staying within navigable cells.

    Navigability excludes: out-of-bounds, no-fly zones, grid obstacles,
    and current moving-obstacle positions.  Falls back to STAY if the drone
    is already at the target or every direction is blocked.
    """
    if pos == target:
        return STAY

    grid     = state["grid"]
    no_fly   = state["no_fly_zones"]
    obs_set  = set(state["moving_obstacles"])
    n        = state["grid_size"]

    dr_t = target[0] - pos[0]
    dc_t = target[1] - pos[1]
    mag = math.sqrt(dr_t ** 2 + dc_t ** 2)
    dr_n, dc_n = dr_t / mag, dc_t / mag

    best_action = STAY
    best_dot    = -math.inf

    for i, (dr, dc) in enumerate(DIRECTIONS):
        nr, nc = pos[0] + dr, pos[1] + dc
        if (
            0 <= nr < n and 0 <= nc < n
            and grid[nr, nc] == 0
            and (nr, nc) not in no_fly
            and (nr, nc) not in obs_set
        ):
            d_mag = math.sqrt(dr ** 2 + dc ** 2)
            dot = (dr / d_mag) * dr_n + (dc / d_mag) * dc_n
            if dot > best_dot:
                best_dot    = dot
                best_action = i

    return best_action


# ---------------------------------------------------------------------------
# Greedy nearest-neighbour policy
# ---------------------------------------------------------------------------

class GreedyNearestNeighborPolicy:
    """Dispatch to the closest undelivered dropoff; recharge when battery < 20 %.

    Decision priority (evaluated every timestep):
      1. Battery < BATTERY_THRESHOLD → head to nearest charging station.
      2. Otherwise → head to the dropoff of the nearest pending delivery
         (Euclidean straight-line distance).
      3. No pending deliveries → stay.
    """

    BATTERY_THRESHOLD: float = 20.0

    def select_action(self, state: Dict[str, Any]) -> int:
        pos      = state["drone_pos"]
        battery  = state["battery_level"]
        pending  = [d for d in state["deliveries"] if d.status == "pending"]
        chargers = state["charging_stations"]

        # --- Low battery: divert to nearest charging station ---
        if battery < self.BATTERY_THRESHOLD and chargers:
            nearest_charger = min(chargers, key=lambda s: _euclidean(pos, s))
            return _best_valid_direction(pos, nearest_charger, state)

        # --- Move toward nearest pending dropoff ---
        if not pending:
            return STAY

        nearest = min(pending, key=lambda d: _euclidean(pos, d.dropoff))
        return _best_valid_direction(pos, nearest.dropoff, state)


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Select a uniformly random valid action at each timestep.

    Valid actions are directional moves that land on a navigable cell
    (in-bounds, not a grid obstacle, not a no-fly zone, not a moving
    obstacle) plus the STAY action which is always valid.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: Dict[str, Any]) -> int:
        pos     = state["drone_pos"]
        grid    = state["grid"]
        no_fly  = state["no_fly_zones"]
        obs_set = set(state["moving_obstacles"])
        n       = state["grid_size"]

        valid: List[int] = [STAY]
        r, c = pos
        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < n and 0 <= nc < n
                and grid[nr, nc] == 0
                and (nr, nc) not in no_fly
                and (nr, nc) not in obs_set
            ):
                valid.append(i)

        return int(self._rng.choice(valid))
