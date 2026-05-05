"""
qlearning.py — Phase 3: Tabular Q-Learning Agent

Implements the high-level dispatch policy using the standard Q-learning update:

    Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]

where each (s, a) pair corresponds to an MDP state (see mdp.py) and a
high-level action.  Low-level movement for DELIVER and CHARGE actions is
executed by A* (astar.find_path / astar.replan), giving the two-layer
architecture described in the project spec.

Training details
----------------
* ε-greedy exploration: ε decays linearly from 1.0 → 0.05 over training.
* Each training episode uses a distinct seed (base + episode index) so the
  agent encounters varied city layouts and wind maps, improving generalisation.
* The Q-table is a nested defaultdict — only visited (state, action) pairs
  occupy memory, keeping the footprint manageable for tabular learning.
* After training, the Q-table and final ε are pickled to qtable.pkl and the
  rolling-average reward curve is saved to training_curve.png.

CLI
---
    python qlearning.py [--tier easy|medium|hard] [--episodes N] [--save PATH]
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless-safe — no display required
import matplotlib.pyplot as plt
import numpy as np

from astar import find_path, replan
from mdp import (
    GAMMA,
    ACTION_TYPE_CHARGE,
    ACTION_TYPE_DELIVER,
    ACTION_TYPE_SKIP,
    Action,
    MDPState,
    action_target,
    encode_state,
    get_valid_actions,
)
from simulator import CityGridSimulator, DIRECTIONS, STAY


class QLearningAgent:
    """Tabular Q-learning agent for the two-layer drone delivery system.

    Parameters
    ----------
    alpha:          Learning rate (default 0.1).
    gamma:          Discount factor (default 0.95, matches MDP spec).
    epsilon_start:  Initial exploration probability (default 1.0).
    epsilon_end:    Final exploration probability (default 0.05).
    """

    def __init__(
        self,
        alpha:         float = 0.1,
        gamma:         float = GAMMA,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.05,
    ) -> None:
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon       = epsilon_start

        # Q[state][action] = Q-value; unvisited pairs default to 0.0
        self.Q: Dict[MDPState, Dict[Action, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state:         MDPState,
        valid_actions: List[Action],
        exploit:       bool = False,
    ) -> Action:
        """ε-greedy selection.  exploit=True forces the greedy choice."""
        if not valid_actions:
            raise ValueError("select_action called with empty action list.")

        if not exploit and np.random.random() < self.epsilon:
            return valid_actions[int(np.random.randint(len(valid_actions)))]

        q_row = self.Q[state]
        return max(valid_actions, key=lambda a: q_row[a])

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------

    def update(
        self,
        state:       MDPState,
        action:      Action,
        reward:      float,
        next_state:  MDPState,
        next_valid:  List[Action],
        done:        bool,
    ) -> None:
        """Apply one Q-learning update.

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
        """
        current_q = self.Q[state][action]
        if done or not next_valid:
            target = reward
        else:
            next_q_row  = self.Q[next_state]
            best_next_q = max(next_q_row[a] for a in next_valid)
            target = reward + self.gamma * best_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # High-level action execution
    # ------------------------------------------------------------------

    def execute_action(self, sim: CityGridSimulator, action: Action) -> float:
        """Execute a high-level action, returning the cumulative simulator reward.

        DELIVER / CHARGE: A* finds the path; drone follows it step-by-step
            with dynamic replanning when moving obstacles appear.
            The A* estimated battery cost feeds into the next MDP state via
            the updated sim.battery after execution.
        SKIP: immediately marks the delivery as failed (no movement).
        """
        action_type, delivery_id = action

        if action_type == ACTION_TYPE_SKIP:
            for d in sim.deliveries:
                if d.id == delivery_id:
                    d.status = "failed"
            return 0.0

        target = action_target(action, sim.deliveries, sim.charging_stations, sim.drone_pos)
        if target is None:
            return 0.0

        return self._execute_to_target(sim, target)

    def _execute_to_target(
        self, sim: CityGridSimulator, target: Tuple[int, int]
    ) -> float:
        """Follow an A*-planned path to target, dynamically replanning as needed.

        On the first call, A* computes the full path.  After each simulator
        step, replan() checks whether the remaining path is still obstacle-free;
        if a moving obstacle has entered the route, A* reruns from the current
        position.  This implements the Phase 1 dynamic-replanning requirement
        in the context of the two-layer agent.

        Returns the cumulative reward collected across all steps.
        """
        total_reward = 0.0
        planned_path: Optional[List[Tuple[int, int]]] = None

        while sim.drone_pos != target and not sim.done:

            if planned_path is None:
                # Initial plan
                result = find_path(
                    sim.grid, sim.drone_pos, target,
                    sim.no_fly_zones, sim.wind_map, sim.battery,
                )
                if result is None:
                    # Target unreachable — fast-forward episode with STAY to
                    # avoid calling A* 200× more in the outer training loop.
                    while not sim.done:
                        _, r, _, _ = sim.step(STAY)
                        total_reward += r
                    break
                planned_path, _ = result
            else:
                # Incremental validity check; replan only if blocked
                result = replan(
                    sim.grid, sim.drone_pos, target,
                    sim.no_fly_zones, sim.wind_map, sim.battery,
                    planned_path,
                )
                if result is None:
                    while not sim.done:
                        _, r, _, _ = sim.step(STAY)
                        total_reward += r
                    break
                planned_path, _ = result

            if len(planned_path) < 2:
                # Already at target — take one STAY step so the simulator
                # processes the arrival (recharge / delivery credit) and
                # advances the timestep. Without this, sim.done never changes
                # and the outer training loop spins forever.
                _, r, _, _ = sim.step(STAY)
                total_reward += r
                break

            next_cell = planned_path[1]
            dr = next_cell[0] - sim.drone_pos[0]
            dc = next_cell[1] - sim.drone_pos[1]
            try:
                action_idx = DIRECTIONS.index((dr, dc))
            except ValueError:
                # Unexpected delta — discard plan and recompute
                planned_path = None
                continue

            _, reward, _, _ = sim.step(action_idx)
            total_reward += reward

            # Advance remaining plan by one cell
            planned_path = planned_path[1:]

        return total_reward

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        sim_config:    Dict[str, Any],
        num_episodes:  int = 5000,
        save_path:     str = "qtable.pkl",
        plot_path:     str = "training_curve.png",
        _ep_offset:    int = 0,
        _total_eps:    Optional[int] = None,
    ) -> List[float]:
        """Train the agent on the given simulator configuration.

        Each episode uses seed = base_seed + episode_index, exposing the
        agent to varied city layouts and wind maps so the learned policy
        generalises across episodes rather than memorising a single map.

        Returns a list of per-episode total rewards.
        """
        total_eps = _total_eps if _total_eps is not None else num_episodes
        base_seed = sim_config.get("seed", 0)
        episode_rewards: List[float] = []

        print(
            f"Training Q-learning agent  episodes={_ep_offset+1}–{_ep_offset+num_episodes}/{total_eps}  "
            f"grid={sim_config['grid_size']}×{sim_config['grid_size']}  "
            f"deliveries={sim_config['num_deliveries']}  "
            f"ε={self.epsilon:.3f}→{self.epsilon_end:.3f}"
        )

        for ep in range(num_episodes):
            ep_config = {**sim_config, "seed": base_seed + ep}
            sim = CityGridSimulator(**ep_config)
            sim.reset()

            ep_reward = 0.0

            while not sim.done:
                s = encode_state(
                    sim.drone_pos, sim.battery,
                    sim.deliveries, sim.timestep, sim.max_steps,
                )
                valid = get_valid_actions(s, sim.deliveries, sim.charging_stations)
                if not valid:
                    break

                action  = self.select_action(s, valid)
                reward  = self.execute_action(sim, action)
                ep_reward += reward

                s_next      = encode_state(
                    sim.drone_pos, sim.battery,
                    sim.deliveries, sim.timestep, sim.max_steps,
                )
                valid_next  = get_valid_actions(
                    s_next, sim.deliveries, sim.charging_stations
                )
                self.update(s, action, reward, s_next, valid_next, sim.done)

            # Linear ε decay over the full training run (offset-aware)
            effective_ep = _ep_offset + ep
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start
                - effective_ep * (self.epsilon_start - self.epsilon_end) / total_eps,
            )
            episode_rewards.append(ep_reward)

            if (effective_ep + 1) % 500 == 0:
                avg = float(np.mean(episode_rewards[-100:]))
                n_states = len(self.Q)
                print(
                    f"  ep {effective_ep+1:5d}/{total_eps}  "
                    f"ε={self.epsilon:.3f}  "
                    f"avg-100={avg:+.1f}  "
                    f"Q-states={n_states}"
                )

        if save_path:
            self.save(save_path)
        if plot_path:
            self._plot_training_curve(episode_rewards, plot_path)
        return episode_rewards

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "qtable.pkl") -> None:
        """Serialise Q-table and current ε to a pickle file."""
        payload = {
            "q_table": {s: dict(av) for s, av in self.Q.items()},
            "epsilon": self.epsilon,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"Q-table saved → {path}  ({len(self.Q)} states)")

    def load(self, path: str = "qtable.pkl") -> None:
        """Restore Q-table and ε from a pickle file."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.Q = defaultdict(lambda: defaultdict(float))
        for s, av in payload["q_table"].items():
            for a, v in av.items():
                self.Q[s][a] = v
        self.epsilon = payload.get("epsilon", self.epsilon_end)
        print(f"Q-table loaded ← {path}  ({len(self.Q)} states)")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _plot_training_curve(
        self, rewards: List[float], path: str = "training_curve.png"
    ) -> None:
        """Save rolling-average reward curve to a PNG file."""
        window = min(100, len(rewards))
        if len(rewards) >= window:
            rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        else:
            rolling = np.array(rewards, dtype=float)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(len(rolling)), rolling, linewidth=1.2, color="steelblue",
                label=f"{window}-ep rolling avg")
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Q-Learning Training Curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Training curve saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from simulator import TIER_CONFIGS

    parser = argparse.ArgumentParser(description="Train the Q-learning dispatch agent.")
    parser.add_argument("--tier",     choices=["easy", "medium", "hard"], default="easy",
                        help="Simulator difficulty tier to train on (default: easy).")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of training episodes (default: 5000).")
    parser.add_argument("--save",     default="qtable.pkl",
                        help="Output path for the Q-table (default: qtable.pkl).")
    args = parser.parse_args()

    agent = QLearningAgent()
    agent.train(TIER_CONFIGS[args.tier], num_episodes=args.episodes, save_path=args.save)
