"""
mdp.py — Phase 3: MDP Formulation

Defines the Markov Decision Process that governs the Q-learning agent's
high-level dispatch decisions.

State  S : (drone_pos, battery_bin, remaining_deliveries, time_bin)
           ─ drone_pos              : (row, col) grid position
           ─ battery_bin            : int in [0, N_BATTERY_BINS-1]
                                      each bin covers 10 percentage points
           ─ remaining_deliveries   : frozenset of pending delivery IDs
           ─ time_bin               : int in [0, N_TIME_BINS-1]
                                      discretised fraction of max_steps elapsed

Actions A : (action_type, delivery_id)
           ─ (ACTION_TYPE_DELIVER, i)  → fly to delivery i's dropoff via A*
           ─ (ACTION_TYPE_CHARGE,  -1) → fly to nearest charging station via A*
           ─ (ACTION_TYPE_SKIP,    i)  → abandon delivery i (mark failed, no penalty)

Transition T(s, a, s'):
    Wind conditions stored in the simulator's wind_map introduce non-determinism:
    the same path costs a stochastically varying amount of battery each episode
    because the wind map is re-sampled from N(1.0, 0.2) on each reset().

Reward R(s, a):
    Accumulated over all simulator steps taken to execute the high-level action:
      +100  per on-time delivery
       −50  for battery depletion or obstacle collision
        −1  per timestep elapsed

Discount factor γ = 0.95
"""

from __future__ import annotations

import math
from typing import Any, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# MDP constants
# ---------------------------------------------------------------------------

GAMMA: float = 0.95         # discount factor
N_BATTERY_BINS: int = 10    # battery [0-9%, 10-19%, …, 90-100%]
N_TIME_BINS:    int = 10    # elapsed-time fraction, 10 equal slices

# High-level action types
ACTION_TYPE_DELIVER: int = 0   # fly_to_delivery_i
ACTION_TYPE_CHARGE:  int = 1   # fly_to_nearest_charging_station
ACTION_TYPE_SKIP:    int = 2   # skip_delivery_i_and_reorder

# Hashable type aliases used by the Q-table
Action   = Tuple[int, int]                               # (action_type, delivery_id)
MDPState = Tuple[Tuple[int, int], int, FrozenSet[int], int]


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(
    drone_pos:     Tuple[int, int],
    battery_level: float,
    deliveries:    List[Any],
    timestep:      int,
    max_steps:     int,
) -> MDPState:
    """Convert the simulator's continuous state into a discrete MDP state tuple.

    The returned tuple is fully hashable and suitable as a Q-table key.
    """
    battery_bin = min(int(battery_level / 10.0), N_BATTERY_BINS - 1)
    time_bin    = min(
        int(timestep * N_TIME_BINS / max(max_steps, 1)),
        N_TIME_BINS - 1,
    )
    remaining = frozenset(d.id for d in deliveries if d.status == "pending")
    return (drone_pos, battery_bin, remaining, time_bin)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def get_valid_actions(
    state:            MDPState,
    deliveries:       List[Any],
    charging_stations: Set[Tuple[int, int]],
) -> List[Action]:
    """Return all high-level actions valid at the given MDP state.

    Valid actions are:
      * fly_to_delivery_i   for every delivery still in the pending set
      * fly_to_charging_station  whenever at least one charger exists
      * skip_delivery_i     for every delivery still in the pending set
    """
    _, _, remaining_ids, _ = state
    actions: List[Action] = []

    for d in deliveries:
        if d.id in remaining_ids:
            actions.append((ACTION_TYPE_DELIVER, d.id))

    if charging_stations:
        actions.append((ACTION_TYPE_CHARGE, -1))

    for d in deliveries:
        if d.id in remaining_ids:
            actions.append((ACTION_TYPE_SKIP, d.id))

    return actions


def action_target(
    action:            Action,
    deliveries:        List[Any],
    charging_stations: Set[Tuple[int, int]],
    drone_pos:         Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Resolve a high-level action to a destination grid cell.

    Returns None for SKIP actions (no physical movement required).
    """
    action_type, delivery_id = action

    if action_type == ACTION_TYPE_DELIVER:
        for d in deliveries:
            if d.id == delivery_id:
                return d.dropoff
        return None

    if action_type == ACTION_TYPE_CHARGE:
        if not charging_stations:
            return None
        return min(
            charging_stations,
            key=lambda s: math.sqrt(
                (s[0] - drone_pos[0]) ** 2 + (s[1] - drone_pos[1]) ** 2
            ),
        )

    # ACTION_TYPE_SKIP — no grid target
    return None
