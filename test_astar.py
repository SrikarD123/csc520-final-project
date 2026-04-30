"""
test_astar.py — Unit tests for Phase 1 A* pathfinder.

Covers:
  - Shortest path on open grid (cardinal and diagonal)
  - Obstacle rerouting
  - No-fly zone avoidance and goal-is-no-fly-zone rejection
  - Battery-constrained failure and success
  - Wind factor effect on battery cost
  - Wind-induced battery failure
  - Zero-cost start == goal
  - Fully-walled-off goal
  - Dynamic replanning (replan): clear path, new grid obstacle, new no-fly zone
"""

import math
import numpy as np
import pytest

from astar import find_path, replan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_grid(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=int)


def grid_with_obstacles(rows: int, cols: int, obstacles) -> np.ndarray:
    g = open_grid(rows, cols)
    for r, c in obstacles:
        g[r, c] = 1
    return g


def uniform_wind(rows: int, cols: int, factor: float = 1.0) -> np.ndarray:
    return np.full((rows, cols), factor, dtype=float)


# ---------------------------------------------------------------------------
# find_path tests
# ---------------------------------------------------------------------------

class TestFindPathOpenGrid:
    def test_straight_east_path(self):
        """Cardinal path: (0,0) → (0,4) should be 5 nodes and cost 4.0."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (0, 4), set(), wind, 100.0)

        assert result is not None
        path, cost = result
        assert path[0] == (0, 0)
        assert path[-1] == (0, 4)
        assert len(path) == 5
        assert cost == pytest.approx(4.0)

    def test_straight_south_path(self):
        """Cardinal path: (0,0) → (4,0) should be 5 nodes and cost 4.0."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (4, 0), set(), wind, 100.0)

        assert result is not None
        path, cost = result
        assert path[0] == (0, 0)
        assert path[-1] == (4, 0)
        assert len(path) == 5
        assert cost == pytest.approx(4.0)

    def test_diagonal_path(self):
        """Diagonal path: (0,0) → (4,4) should be 5 nodes and cost 4√2."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (4, 4), set(), wind, 100.0)

        assert result is not None
        path, cost = result
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        assert len(path) == 5
        assert cost == pytest.approx(4 * math.sqrt(2))

    def test_start_equals_goal(self):
        """Zero-movement: start == goal returns single-node path with cost 0."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (2, 2), (2, 2), set(), wind, 100.0)

        assert result is not None
        path, cost = result
        assert path == [(2, 2)]
        assert cost == pytest.approx(0.0)

    def test_adjacent_nodes(self):
        """Single step: adjacent nodes produce correct single-edge cost."""
        grid = open_grid(3, 3)
        wind = uniform_wind(3, 3)
        result = find_path(grid, (1, 1), (1, 2), set(), wind, 10.0)

        assert result is not None
        path, cost = result
        assert path == [(1, 1), (1, 2)]
        assert cost == pytest.approx(1.0)


class TestFindPathObstacles:
    def test_vertical_wall_rerouting(self):
        """A* navigates around a vertical wall blocking the direct east route."""
        # Block column 2 rows 0–3; row 4 gap forces detour south then east
        obstacles = [(r, 2) for r in range(4)]
        grid = grid_with_obstacles(5, 5, obstacles)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (0, 4), set(), wind, 100.0)

        assert result is not None
        path, _ = result
        assert path[0] == (0, 0)
        assert path[-1] == (0, 4)
        # Must not pass through the blocked cells
        for r, c in path:
            assert not (c == 2 and r < 4), f"Path used obstacle cell ({r},{c})"

    def test_goal_surrounded_unreachable(self):
        """Return None when goal column is fully walled off (no diagonal bypass)."""
        # Blocking the entire column 3 cuts off column 4 even with 8-dir movement
        obstacles = [(r, 3) for r in range(5)]
        grid = grid_with_obstacles(5, 5, obstacles)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (2, 4), set(), wind, 100.0)

        assert result is None

    def test_maze_with_single_opening(self):
        """A* finds the only opening through a near-complete horizontal wall."""
        # Row 2 is a wall except column 4
        obstacles = [(2, c) for c in range(4)]  # cols 0-3 blocked, col 4 open
        grid = grid_with_obstacles(5, 5, obstacles)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 2), (4, 2), set(), wind, 100.0)

        assert result is not None
        path, _ = result
        assert path[0] == (0, 2)
        assert path[-1] == (4, 2)
        # Every node must be unblocked
        for r, c in path:
            assert grid[r, c] == 0, f"Path used obstacle at ({r},{c})"


class TestFindPathNoFlyZones:
    def test_no_fly_zone_avoidance(self):
        """Path never passes through any no-fly zone cell."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        no_fly = {(0, 1), (0, 2), (0, 3)}
        result = find_path(grid, (0, 0), (0, 4), no_fly, wind, 100.0)

        assert result is not None
        path, _ = result
        assert path[0] == (0, 0)
        assert path[-1] == (0, 4)
        for node in path:
            assert node not in no_fly, f"Path entered no-fly zone at {node}"

    def test_no_fly_goal_is_none(self):
        """Return None when the goal itself is in a no-fly zone."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (4, 4), {(4, 4)}, wind, 100.0)

        assert result is None

    def test_no_fly_start_is_none(self):
        """Return None when the start itself is in a no-fly zone."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (4, 4), {(0, 0)}, wind, 100.0)

        assert result is None

    def test_no_fly_zone_completely_seals_goal(self):
        """No-fly zone ring around goal makes it unreachable."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        # Surround (2,2) with a ring of no-fly zones
        ring = {
            (1, 1), (1, 2), (1, 3),
            (2, 1),         (2, 3),
            (3, 1), (3, 2), (3, 3),
        }
        result = find_path(grid, (0, 0), (2, 2), ring, wind, 100.0)

        assert result is None


class TestFindPathBattery:
    def test_battery_too_low_for_goal(self):
        """Return None when battery cannot cover the minimum path cost."""
        grid = open_grid(10, 10)
        wind = uniform_wind(10, 10)
        # Minimum diagonal path (0,0)→(9,9): 9 * √2 ≈ 12.73
        result = find_path(grid, (0, 0), (9, 9), set(), wind, 5.0)

        assert result is None

    def test_battery_exactly_sufficient(self):
        """Path found when battery is just enough to cover the diagonal."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        # Optimal diagonal (0,0)→(4,4): 4√2 ≈ 5.657
        result = find_path(grid, (0, 0), (4, 4), set(), wind, 6.0)

        assert result is not None

    def test_battery_slightly_under_sufficient(self):
        """Return None when battery is marginally below minimum path cost."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        # 4√2 ≈ 5.657; budget of 5.0 should fail
        result = find_path(grid, (0, 0), (4, 4), set(), wind, 5.0)

        assert result is None

    def test_wind_doubles_cost(self):
        """Wind factor of 2.0 doubles the battery cost compared to no wind."""
        grid = open_grid(5, 5)
        low_wind = uniform_wind(5, 5, 1.0)
        high_wind = uniform_wind(5, 5, 2.0)

        r1 = find_path(grid, (0, 0), (0, 4), set(), low_wind, 100.0)
        r2 = find_path(grid, (0, 0), (0, 4), set(), high_wind, 100.0)

        assert r1 is not None and r2 is not None
        assert r2[1] == pytest.approx(r1[1] * 2.0)

    def test_wind_causes_battery_failure(self):
        """High wind turns an otherwise-reachable path into a battery failure."""
        grid = open_grid(5, 5)
        # Without wind: 4 steps east = 4.0 battery
        # With wind=3.0: 12.0 battery needed; budget = 10.0 → fail
        wind = uniform_wind(5, 5, 3.0)
        result = find_path(grid, (0, 0), (0, 4), set(), wind, 10.0)

        assert result is None

    def test_wind_none_defaults_to_one(self):
        """wind_map=None is treated as all-1.0 wind factor."""
        grid = open_grid(5, 5)
        r_none = find_path(grid, (0, 0), (0, 4), set(), None, 100.0)
        r_unit = find_path(grid, (0, 0), (0, 4), set(), uniform_wind(5, 5, 1.0), 100.0)

        assert r_none is not None and r_unit is not None
        assert r_none[1] == pytest.approx(r_unit[1])

    def test_wind_dict_format(self):
        """Dict-format wind_map produces correct per-edge costs."""
        grid = open_grid(3, 3)
        # Only the edge (1,0)→(1,1) has high wind; rest default to 1.0
        wind_dict = {((1, 0), (1, 1)): 5.0}
        # Optimal path (1,0)→(1,2): could go direct (cost=1*5+1*1=6) or
        # route diagonally — let A* pick the cheapest.
        result = find_path(grid, (1, 0), (1, 2), set(), wind_dict, 100.0)

        assert result is not None
        # Verify the returned cost matches manual edge summation
        path, cost = result
        manual_cost = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            dist = math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
            wf = wind_dict.get((a, b), 1.0)
            manual_cost += dist * wf
        assert cost == pytest.approx(manual_cost)


class TestFindPathReturnedPath:
    def test_path_is_contiguous(self):
        """Every consecutive pair in the returned path must be adjacent (≤ √2 apart)."""
        grid = open_grid(8, 8)
        wind = uniform_wind(8, 8)
        result = find_path(grid, (0, 0), (7, 7), set(), wind, 100.0)

        assert result is not None
        path, _ = result
        for i in range(len(path) - 1):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            dist = math.sqrt((r1-r0)**2 + (c1-c0)**2)
            assert dist <= math.sqrt(2) + 1e-9, (
                f"Non-adjacent step in path: {path[i]} → {path[i+1]}"
            )

    def test_path_nodes_are_valid(self):
        """All returned path nodes must be valid (in-bounds, not obstacle/no-fly)."""
        obstacles = [(2, c) for c in range(4)]
        grid = grid_with_obstacles(6, 6, obstacles)
        wind = uniform_wind(6, 6)
        no_fly = {(4, 4)}
        result = find_path(grid, (0, 0), (5, 5), no_fly, wind, 100.0)

        assert result is not None
        path, _ = result
        rows, cols = grid.shape
        for r, c in path:
            assert 0 <= r < rows and 0 <= c < cols
            assert grid[r, c] == 0
            assert (r, c) not in no_fly


# ---------------------------------------------------------------------------
# replan tests
# ---------------------------------------------------------------------------

class TestReplan:
    def test_no_replan_needed_returns_remaining_path(self):
        """When the remaining path is still clear, replan returns it unchanged."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        result = find_path(grid, (0, 0), (0, 4), set(), wind, 100.0)
        assert result is not None
        path, _ = result  # [(0,0),(0,1),(0,2),(0,3),(0,4)]

        replanned = replan(grid, (0, 2), (0, 4), set(), wind, 100.0, path)
        assert replanned is not None
        new_path, _ = replanned
        assert new_path[0] == (0, 2)
        assert new_path[-1] == (0, 4)

    def test_replan_triggered_by_new_grid_obstacle(self):
        """New obstacle on planned path triggers replan that avoids it."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        planned = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

        # Block (0,3) — on the remaining path from (0,1)
        new_grid = grid_with_obstacles(5, 5, [(0, 3)])
        result = replan(new_grid, (0, 1), (0, 4), set(), wind, 100.0, planned)

        assert result is not None
        new_path, _ = result
        assert new_path[0] == (0, 1)
        assert new_path[-1] == (0, 4)
        assert (0, 3) not in new_path

    def test_replan_triggered_by_new_no_fly_zone(self):
        """New no-fly zone on planned path triggers replan that avoids it."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        planned = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

        new_no_fly = {(0, 2), (0, 3)}
        result = replan(grid, (0, 1), (0, 4), new_no_fly, wind, 100.0, planned)

        assert result is not None
        new_path, _ = result
        assert new_path[0] == (0, 1)
        assert new_path[-1] == (0, 4)
        for node in new_path:
            assert node not in new_no_fly

    def test_replan_returns_none_when_unreachable(self):
        """Replan returns None when new obstacles make goal unreachable."""
        # Full column-3 wall plus column-4 goal — goal cut off
        obstacles = [(r, 3) for r in range(5)]
        grid = grid_with_obstacles(5, 5, obstacles)
        wind = uniform_wind(5, 5)
        planned = [(0, 0), (0, 1), (0, 2), (0, 3), (2, 4)]

        result = replan(grid, (0, 1), (2, 4), set(), wind, 100.0, planned)
        assert result is None

    def test_replan_when_current_pos_off_plan(self):
        """Drone off planned route triggers full replan from current position."""
        grid = open_grid(5, 5)
        wind = uniform_wind(5, 5)
        planned = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

        # Drone drifted to (1,0) — not in planned path
        result = replan(grid, (1, 0), (0, 4), set(), wind, 100.0, planned)

        assert result is not None
        new_path, _ = result
        assert new_path[0] == (1, 0)
        assert new_path[-1] == (0, 4)

    def test_replan_cost_consistent_with_find_path(self):
        """Replan on a clear remaining path returns same cost as find_path direct call."""
        grid = open_grid(6, 6)
        wind = uniform_wind(6, 6)
        planned = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

        r_replan = replan(grid, (2, 2), (5, 5), set(), wind, 100.0, planned)
        r_direct = find_path(grid, (2, 2), (5, 5), set(), wind, 100.0)

        assert r_replan is not None and r_direct is not None
        _, cost_replan = r_replan
        _, cost_direct = r_direct
        assert cost_replan == pytest.approx(cost_direct)

    def test_replan_battery_respected(self):
        """Replanned path still respects battery constraints."""
        grid = open_grid(10, 10)
        wind = uniform_wind(10, 10)
        planned = [(0, 0), (0, 1), (0, 2), (9, 9)]

        # Force replan: block (0,2)
        new_grid = grid_with_obstacles(10, 10, [(0, 2)])
        # Replanning (0,1)→(9,9) needs at minimum 9√2 ≈ 12.73; budget 5.0 → fail
        result = replan(new_grid, (0, 1), (9, 9), set(), wind, 5.0, planned)
        assert result is None
