"""
simulator.py — Phase 2: City Grid Simulation Environment

Provides CityGridSimulator, a discrete-time environment for evaluating
drone delivery policies on procedurally generated city maps.

Key design points
-----------------
* Grid: 2-D numpy array; 0 = open cell.  No-fly zones are hard-blocked
  overlays (not baked into the grid) so A* can treat them separately.
* Wind map: per-cell Gaussian(1.0, 0.2) factor, clipped to [0.5, 2.0].
  Stored as a 2-D numpy array compatible with astar.find_path().
* Actions: 0–7 = 8-directional moves (same order as astar.DIRECTIONS);
  8 = stay.
* Battery: starts at 100.0, consumed as distance × wind_factor per step.
  Charging stations restore battery to 100 on arrival.
* Deliveries: drone must reach a dropoff cell within the time window to
  earn the +100 reward.
* Moving obstacles: each timestep, every obstacle moves one cell in a
  random valid direction and kills the drone on collision (−50).
* Reward: −1/timestep, +100/on-time delivery, −50/crash.
* Benchmarks: generate_benchmarks() writes 100 reproducible configs to
  benchmarks.pkl (34 easy, 33 medium, 33 hard).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# 8-directional moves (N, NE, E, SE, S, SW, W, NW) — matches astar.DIRECTIONS
DIRECTIONS: List[Tuple[int, int]] = [
    (-1,  0),
    (-1,  1),
    ( 0,  1),
    ( 1,  1),
    ( 1,  0),
    ( 1, -1),
    ( 0, -1),
    (-1, -1),
]
STAY: int = 8

# Canonical config for each difficulty tier
TIER_CONFIGS: Dict[str, Dict[str, int]] = {
    "easy": {
        "grid_size": 10,
        "num_deliveries": 5,
        "num_no_fly_zones": 5,
        "num_charging_stations": 2,
        "num_moving_obstacles": 3,
    },
    "medium": {
        "grid_size": 20,
        "num_deliveries": 10,
        "num_no_fly_zones": 15,
        "num_charging_stations": 4,
        "num_moving_obstacles": 6,
    },
    "hard": {
        "grid_size": 50,
        "num_deliveries": 20,
        "num_no_fly_zones": 50,
        "num_charging_stations": 10,
        "num_moving_obstacles": 15,
    },
}


@dataclass
class Delivery:
    """A single delivery job."""
    id: int
    pickup: Tuple[int, int]
    dropoff: Tuple[int, int]
    time_window: Tuple[int, int]   # (earliest_step, latest_step)
    status: str = "pending"        # "pending" | "delivered" | "failed"


State = Dict[str, Any]  # type alias for the dict returned by step/reset


class CityGridSimulator:
    """Discrete-time city grid simulator for drone delivery routing.

    Parameters
    ----------
    grid_size:
        Side length of the square city grid.
    num_deliveries:
        Number of delivery jobs per episode.
    num_no_fly_zones:
        Number of cells permanently blocked from flight.
    num_charging_stations:
        Number of cells that refill battery to 100 on arrival.
    num_moving_obstacles:
        Number of obstacles that move one cell per timestep.
    seed:
        RNG seed for full reproducibility.
    """

    def __init__(
        self,
        grid_size: int = 10,
        num_deliveries: int = 5,
        num_no_fly_zones: int = 5,
        num_charging_stations: int = 2,
        num_moving_obstacles: int = 3,
        seed: int = 42,
    ) -> None:
        required = (
            1                       # drone start
            + num_no_fly_zones
            + num_charging_stations
            + 2 * num_deliveries    # pickup + dropoff per job
            + num_moving_obstacles
        )
        if required >= grid_size * grid_size:
            raise ValueError(
                f"Too many entities ({required}) for a {grid_size}×{grid_size} grid."
            )

        self.grid_size = grid_size
        self.num_deliveries = num_deliveries
        self.num_no_fly_zones = num_no_fly_zones
        self.num_charging_stations = num_charging_stations
        self.num_moving_obstacles = num_moving_obstacles
        self.seed = seed
        self.max_steps: int = grid_size * num_deliveries * 4

        # Runtime state (populated by reset())
        self.grid: np.ndarray = np.zeros((grid_size, grid_size), dtype=int)
        self.wind_map: np.ndarray = np.ones((grid_size, grid_size), dtype=float)
        self.drone_pos: Tuple[int, int] = (0, 0)
        self.battery: float = 100.0
        self.deliveries: List[Delivery] = []
        self.no_fly_zones: Set[Tuple[int, int]] = set()
        self.charging_stations: Set[Tuple[int, int]] = set()
        self.moving_obstacles: List[List[int]] = []
        self.timestep: int = 0
        self.done: bool = False

        # Preserved RNG for deterministic obstacle movement within an episode
        self._obs_rng: np.random.Generator = np.random.default_rng(seed)

        # Pygame display handle (lazy init)
        self._screen: Optional[Any] = None
        self._pygame_initialized: bool = False

        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> State:
        """Reinitialise the episode from the configured seed.

        Returns the initial state dict.
        """
        rng = np.random.default_rng(self.seed)
        n = self.grid_size

        self.grid = np.zeros((n, n), dtype=int)
        self.wind_map = np.clip(
            rng.normal(1.0, 0.2, (n, n)), 0.5, 2.0
        ).astype(float)

        occupied: Set[Tuple[int, int]] = set()

        def rand_empty() -> Tuple[int, int]:
            while True:
                r = int(rng.integers(0, n))
                c = int(rng.integers(0, n))
                if (r, c) not in occupied:
                    return (r, c)

        self.drone_pos = rand_empty()
        occupied.add(self.drone_pos)

        self.no_fly_zones = set()
        for _ in range(self.num_no_fly_zones):
            pos = rand_empty()
            self.no_fly_zones.add(pos)
            occupied.add(pos)

        self.charging_stations = set()
        for _ in range(self.num_charging_stations):
            pos = rand_empty()
            self.charging_stations.add(pos)
            occupied.add(pos)

        self.deliveries = []
        for i in range(self.num_deliveries):
            pickup = rand_empty()
            occupied.add(pickup)
            dropoff = rand_empty()
            occupied.add(dropoff)
            latest = int(rng.integers(self.max_steps // 2, self.max_steps + 1))
            self.deliveries.append(
                Delivery(id=i, pickup=pickup, dropoff=dropoff,
                         time_window=(0, latest))
            )

        self.moving_obstacles = []
        for _ in range(self.num_moving_obstacles):
            pos = rand_empty()
            occupied.add(pos)
            self.moving_obstacles.append(list(pos))

        # Fork a fresh RNG for obstacle movement, seeded deterministically
        self._obs_rng = np.random.default_rng(self.seed + 99_999)

        self.battery = 100.0
        self.timestep = 0
        self.done = False

        return self._make_state()

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Advance the simulation by one timestep.

        Parameters
        ----------
        action:
            Integer 0–7 for a directional move, or 8 (STAY).

        Returns
        -------
        state:   New environment state.
        reward:  Scalar reward for this transition.
        done:    True when the episode has ended.
        info:    Diagnostic dict (keys: 'crash', 'charged', 'delivered',
                 'delivery_failed', 'invalid_move', 'timeout',
                 'episode_complete').
        """
        if self.done:
            raise RuntimeError("Episode is done — call reset() first.")

        reward: float = -1.0
        info: Dict[str, Any] = {}

        # --- Directional move ---
        if action < STAY:
            dr, dc = DIRECTIONS[action]
            nr = self.drone_pos[0] + dr
            nc = self.drone_pos[1] + dc

            if self._cell_navigable(nr, nc):
                dist = float(np.sqrt(dr * dr + dc * dc))
                cost = dist * float(self.wind_map[nr, nc])

                if self.battery - cost <= 0.0:
                    self.battery = 0.0
                    self.done = True
                    reward -= 50.0
                    info["crash"] = "battery_depleted"
                    return self._make_state(), reward, self.done, info

                self.battery -= cost
                self.drone_pos = (nr, nc)
            else:
                info["invalid_move"] = True
        # action == STAY: drone holds position; no battery consumed

        # --- Charging ---
        if self.drone_pos in self.charging_stations:
            self.battery = 100.0
            info["charged"] = True

        # --- Delivery check at current cell ---
        for d in self.deliveries:
            if d.status == "pending" and self.drone_pos == d.dropoff:
                if self.timestep <= d.time_window[1]:
                    d.status = "delivered"
                    reward += 100.0
                    info.setdefault("delivered", []).append(d.id)
                else:
                    d.status = "failed"
                    info.setdefault("delivery_failed", []).append(d.id)

        # --- Advance clock ---
        self.timestep += 1

        # Expire deliveries whose windows have closed
        for d in self.deliveries:
            if d.status == "pending" and self.timestep > d.time_window[1]:
                d.status = "failed"
                info.setdefault("delivery_failed", []).append(d.id)

        # --- Move obstacles ---
        self._move_obstacles()

        # --- Collision with moving obstacle ---
        obs_set = {tuple(o) for o in self.moving_obstacles}
        if self.drone_pos in obs_set:
            self.done = True
            reward -= 50.0
            info["crash"] = "obstacle_collision"
            return self._make_state(), reward, self.done, info

        # --- Terminal conditions ---
        if self.timestep >= self.max_steps:
            self.done = True
            info["timeout"] = True
        elif all(d.status != "pending" for d in self.deliveries):
            self.done = True
            info["episode_complete"] = True

        return self._make_state(), reward, self.done, info

    def render(self) -> None:
        """Visualise the current state with Pygame.

        Legend
        ------
        Blue   — drone
        Red    — no-fly zone
        Green  — charging station
        Yellow — pending delivery dropoff
        Orange — moving obstacle
        """
        try:
            import pygame
        except ImportError:
            print("[render] pygame not installed — skipping visualisation.")
            return

        WINDOW = 600
        cell = max(2, WINDOW // self.grid_size)
        W = cell * self.grid_size
        H = cell * self.grid_size

        if not self._pygame_initialized:
            pygame.init()
            self._screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Delivery Drone Route Planner")
            self._pygame_initialized = True

        BG         = (210, 210, 210)
        GRID_LINE  = (160, 160, 160)
        C_NFZ      = (220,  50,  50)
        C_CHARGER  = ( 50, 200,  50)
        C_DELIVERY = (230, 200,   0)
        C_OBSTACLE = (230, 130,   0)
        C_DRONE    = (  0,   0, 220)

        self._screen.fill(BG)
        for i in range(self.grid_size + 1):
            pygame.draw.line(self._screen, GRID_LINE, (i * cell, 0), (i * cell, H))
            pygame.draw.line(self._screen, GRID_LINE, (0, i * cell), (W, i * cell))

        def fill(r: int, c: int, color: tuple) -> None:
            pygame.draw.rect(
                self._screen, color,
                (c * cell + 1, r * cell + 1, cell - 2, cell - 2)
            )

        for r, c in self.no_fly_zones:
            fill(r, c, C_NFZ)
        for r, c in self.charging_stations:
            fill(r, c, C_CHARGER)
        for d in self.deliveries:
            if d.status == "pending":
                fill(*d.dropoff, C_DELIVERY)
        for obs in self.moving_obstacles:
            fill(obs[0], obs[1], C_OBSTACLE)
        fill(*self.drone_pos, C_DRONE)

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False

    def get_valid_actions(self) -> List[int]:
        """Return all actions reachable from the current drone position."""
        valid = [STAY]
        r, c = self.drone_pos
        for i, (dr, dc) in enumerate(DIRECTIONS):
            if self._cell_navigable(r + dr, c + dc):
                valid.append(i)
        return valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cell_navigable(self, r: int, c: int) -> bool:
        """True if (r, c) is in-bounds, not a grid obstacle, and not a no-fly zone."""
        return (
            0 <= r < self.grid_size
            and 0 <= c < self.grid_size
            and self.grid[r, c] == 0
            and (r, c) not in self.no_fly_zones
        )

    def _move_obstacles(self) -> None:
        """Move each obstacle one cell in a random valid direction."""
        rng = self._obs_rng
        occupied = {tuple(o) for o in self.moving_obstacles}
        new_positions: List[List[int]] = []

        for obs in self.moving_obstacles:
            r, c = obs
            current = (r, c)
            # Free current cell so others (and itself) can target it
            occupied.discard(current)

            candidates = []
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                cand = (nr, nc)
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and self.grid[nr, nc] == 0
                    and cand not in self.no_fly_zones
                    and cand not in occupied
                ):
                    candidates.append(cand)

            if candidates:
                chosen = candidates[int(rng.integers(0, len(candidates)))]
                occupied.add(chosen)
                new_positions.append(list(chosen))
            else:
                occupied.add(current)
                new_positions.append(obs)

        self.moving_obstacles = new_positions

    def _make_state(self) -> State:
        return {
            "drone_pos":          self.drone_pos,
            "battery_level":      self.battery,
            "deliveries":         self.deliveries,
            "timestep":           self.timestep,
            "no_fly_zones":       self.no_fly_zones,
            "charging_stations":  self.charging_stations,
            "moving_obstacles":   [tuple(o) for o in self.moving_obstacles],
            "wind_map":           self.wind_map,
            "grid":               self.grid,
            "grid_size":          self.grid_size,
            "max_steps":          self.max_steps,
        }


# ---------------------------------------------------------------------------
# Benchmark suite generation
# ---------------------------------------------------------------------------

def generate_benchmarks(save_path: str = "benchmarks.pkl") -> None:
    """Create 100 reproducible test scenarios and serialise to disk.

    Distribution: 34 easy, 33 medium, 33 hard (= 100 total).
    Each entry is a dict with keys 'tier', 'seed', and 'config' — pass
    config as **kwargs to CityGridSimulator to reproduce the scenario.
    """
    scenarios: List[Dict[str, Any]] = []
    tier_counts = {"easy": 34, "medium": 33, "hard": 33}
    seed_base = 0

    for tier, count in tier_counts.items():
        base_cfg = TIER_CONFIGS[tier]
        for i in range(count):
            s = seed_base + i
            scenarios.append(
                {"tier": tier, "seed": s, "config": {**base_cfg, "seed": s}}
            )
        seed_base += count

    with open(save_path, "wb") as fh:
        pickle.dump(scenarios, fh)

    print(f"Saved {len(scenarios)} benchmark scenarios → {save_path}")
    for tier, count in tier_counts.items():
        cfg = TIER_CONFIGS[tier]
        print(
            f"  {tier:6s}: {count} scenarios  "
            f"(grid={cfg['grid_size']}×{cfg['grid_size']}, "
            f"deliveries={cfg['num_deliveries']})"
        )


if __name__ == "__main__":
    generate_benchmarks()
