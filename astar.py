"""
astar.py — Phase 1: A* Pathfinder with Dynamic Replanning

Low-level pathfinding engine for the Delivery Drone Route Planner.
Represents the city as a 2D grid with 8-directional movement, Euclidean
distance heuristic, wind-penalized edge costs, no-fly zone enforcement,
and mid-flight dynamic replanning when new obstacles are detected.
"""

import heapq
import math
import numpy as np
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

# Type aliases
Node = Tuple[int, int]          # (row, col)
Path = List[Node]
WindMap = Union[np.ndarray, Dict[Tuple[Node, Node], float]]

# 8-directional moves: (delta_row, delta_col)
DIRECTIONS: List[Tuple[int, int]] = [
    (-1,  0),  # N
    (-1,  1),  # NE
    ( 0,  1),  # E
    ( 1,  1),  # SE
    ( 1,  0),  # S
    ( 1, -1),  # SW
    ( 0, -1),  # W
    (-1, -1),  # NW
]


def _euclidean(a: Node, b: Node) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _wind_factor(wind_map: Optional[WindMap], from_node: Node, to_node: Node) -> float:
    """Return the wind factor for traversing an edge (from_node -> to_node).

    Supports two wind_map formats:
      - numpy 2D array: wind_map[row][col] is the factor at the destination cell
      - dict: keyed by (from_node, to_node) pairs
      - None: no wind, factor = 1.0
    """
    if wind_map is None:
        return 1.0
    if isinstance(wind_map, dict):
        return float(wind_map.get((from_node, to_node), 1.0))
    # numpy array — factor at destination cell
    return float(wind_map[to_node[0]][to_node[1]])


def _is_valid(grid: np.ndarray, node: Node, no_fly_zones: Set[Node]) -> bool:
    """Return True if node is within bounds, not an obstacle, and not a no-fly zone."""
    r, c = node
    rows, cols = grid.shape
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    if grid[r, c] != 0:
        return False
    if node in no_fly_zones:
        return False
    return True


def find_path(
    grid: np.ndarray,
    start: Node,
    goal: Node,
    no_fly_zones: Set[Node],
    wind_map: Optional[WindMap],
    battery_remaining: float,
) -> Optional[Tuple[Path, float]]:
    """Find the lowest-cost path from start to goal using A* search.

    Edge cost = Euclidean distance × wind_factor (battery consumption model).
    Nodes in no_fly_zones or grid obstacles are hard-blocked.
    Paths whose cumulative cost would exceed battery_remaining are pruned.

    Args:
        grid:               2D numpy array; 0 = open, non-zero = obstacle.
        start:              (row, col) origin.
        goal:               (row, col) destination.
        no_fly_zones:       Set of (row, col) positions that must not be entered.
        wind_map:           Per-edge wind factors (see _wind_factor).
        battery_remaining:  Maximum allowable total edge cost.

    Returns:
        (path, estimated_battery_cost) where path is an ordered list of nodes
        from start to goal, or None if the goal is unreachable within constraints.
    """
    if not _is_valid(grid, start, no_fly_zones):
        return None
    if not _is_valid(grid, goal, no_fly_zones):
        return None

    if start == goal:
        return [start], 0.0

    # Min-heap entries: (f_score, tie_break_counter, g_score, node)
    # tie_break_counter keeps heapq stable when f scores are equal.
    counter = 0
    h0 = _euclidean(start, goal)
    open_heap: List[Tuple[float, int, float, Node]] = [(h0, counter, 0.0, start)]

    g_score: Dict[Node, float] = {start: 0.0}
    came_from: Dict[Node, Optional[Node]] = {start: None}

    while open_heap:
        f, _, g, current = heapq.heappop(open_heap)

        # Stale entry — a better path to `current` was already found.
        if g > g_score.get(current, math.inf):
            continue

        if current == goal:
            return _reconstruct_path(came_from, current), g

        for dr, dc in DIRECTIONS:
            neighbor = (current[0] + dr, current[1] + dc)

            if not _is_valid(grid, neighbor, no_fly_zones):
                continue

            base_dist = _euclidean(current, neighbor)  # 1.0 or √2
            edge_cost = base_dist * _wind_factor(wind_map, current, neighbor)
            tentative_g = g + edge_cost

            # Battery constraint: prune branch if already over budget.
            if tentative_g > battery_remaining:
                continue

            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = _euclidean(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h, counter, tentative_g, neighbor))

    return None  # Goal unreachable within battery constraints


def _reconstruct_path(came_from: Dict[Node, Optional[Node]], current: Node) -> Path:
    path: Path = []
    node: Optional[Node] = current
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def replan(
    grid: np.ndarray,
    current_pos: Node,
    goal: Node,
    no_fly_zones: Set[Node],
    wind_map: Optional[WindMap],
    battery_remaining: float,
    planned_path: Path,
) -> Optional[Tuple[Path, float]]:
    """Detect whether the planned path is still valid and replan if needed.

    Called mid-flight when the grid or no-fly zones may have changed (e.g., a
    moving obstacle enters the drone's intended route).  If any node in the
    remaining portion of planned_path is now blocked, A* is rerun from
    current_pos to the original goal.

    Args:
        grid:           Current grid state (may include newly appeared obstacles).
        current_pos:    Drone's current position.
        goal:           Original destination.
        no_fly_zones:   Current no-fly zone set.
        wind_map:       Current wind conditions.
        battery_remaining: Remaining battery from current_pos.
        planned_path:   Previously computed path (full, from original start).

    Returns:
        (path, estimated_battery_cost) starting from current_pos, or None if
        the goal is no longer reachable.
    """
    # Locate current position in the existing plan.
    try:
        idx = planned_path.index(current_pos)
    except ValueError:
        # Drone is off-plan (e.g., forced deviation) — replan from scratch.
        return find_path(grid, current_pos, goal, no_fly_zones, wind_map, battery_remaining)

    remaining = planned_path[idx:]

    # Check every upcoming node for newly blocked cells.
    needs_replan = any(not _is_valid(grid, node, no_fly_zones) for node in remaining)

    if not needs_replan:
        # Path still clear — compute cost of remaining segment and return it.
        cost = sum(
            _euclidean(remaining[i], remaining[i + 1])
            * _wind_factor(wind_map, remaining[i], remaining[i + 1])
            for i in range(len(remaining) - 1)
        )
        return remaining, cost

    # New obstacle detected on planned route — replan from current position.
    return find_path(grid, current_pos, goal, no_fly_zones, wind_map, battery_remaining)
