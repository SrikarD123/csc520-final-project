"""
integrate.py — Phase 3: Two-Layer System Integration

Wires the Phase 1 A* pathfinder and the Phase 3 Q-learning agent together:

    ┌─────────────────────────────────────────────────────┐
    │  Q-learning agent  (high level)                     │
    │  Selects: which delivery to pursue, or when to      │
    │           recharge, or which delivery to skip       │
    └──────────────────┬──────────────────────────────────┘
                       │  destination cell
    ┌──────────────────▼──────────────────────────────────┐
    │  A* pathfinder  (low level)                         │
    │  Computes: grid path to destination                 │
    │  Returns:  (path, estimated_battery_cost)           │
    │  The battery cost feeds back into the MDP state     │
    │  transition so the Q-learner can reason about       │
    │  battery usage when choosing the next action.       │
    └──────────────────┬──────────────────────────────────┘
                       │  step(action) × N
    ┌──────────────────▼──────────────────────────────────┐
    │  CityGridSimulator  (environment)                   │
    └─────────────────────────────────────────────────────┘

Usage
-----
    python integrate.py --scenario easy --render
    python integrate.py --scenario hard --seed 7
    python integrate.py --train-if-missing --episodes 5000
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Dict, List, Set, Tuple

from mdp import encode_state, get_valid_actions
from qlearning import QLearningAgent
from simulator import CityGridSimulator, TIER_CONFIGS


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def run_integrated_episode(
    sim:          CityGridSimulator,
    agent:        QLearningAgent,
    render:       bool  = False,
    render_delay: float = 0.05,
) -> Dict[str, Any]:
    """Execute one full episode using the two-layer Q-learning + A* agent.

    The Q-learning agent makes high-level dispatch decisions (which delivery
    to pursue next, when to recharge, which to skip).  A* translates each
    decision into a sequence of low-level grid moves that the simulator
    executes one step at a time.

    The estimated battery cost returned by A* for each high-level action is
    reflected in the simulator's battery level after execution, which is then
    re-encoded into the MDP state — closing the feedback loop between the
    two layers.

    Parameters
    ----------
    sim:          A freshly reset (or not yet reset) CityGridSimulator.
    agent:        A trained (or zero-initialised) QLearningAgent.
    render:       If True, call sim.render() at every step.
    render_delay: Seconds to sleep between frames when rendering.

    Returns
    -------
    Metrics dict with keys:
      total_reward, delivered, total_deliveries, delivery_success_rate,
      route_efficiency, battery_remaining, steps, high_level_actions.
    """
    sim.reset()
    start_pos = sim.drone_pos

    total_reward = 0.0
    delivered_ids: Set[int] = set()
    delivery_order: List[Tuple[int, int]] = []   # dropoff positions in completion order
    high_level_actions = 0

    while not sim.done:
        if render:
            sim.render()
            time.sleep(render_delay)

        s = encode_state(
            sim.drone_pos, sim.battery,
            sim.deliveries, sim.timestep, sim.max_steps,
        )
        valid = get_valid_actions(s, sim.deliveries, sim.charging_stations)
        if not valid:
            break

        # ── High-level decision (Q-learning layer) ────────────────────
        action = agent.select_action(s, valid, exploit=True)
        high_level_actions += 1

        # ── Low-level execution (A* layer) ────────────────────────────
        reward = agent.execute_action(sim, action)
        total_reward += reward

        # Track newly completed deliveries for route-efficiency metric
        for d in sim.deliveries:
            if d.status == "delivered" and d.id not in delivered_ids:
                delivered_ids.add(d.id)
                delivery_order.append(d.dropoff)

    if render:
        sim.render()  # final frame

    delivered       = len(delivered_ids)
    total_deliveries = len(sim.deliveries)

    # Route efficiency: sum of optimal waypoint distances / actual steps
    if delivery_order and sim.timestep > 0:
        waypoints   = [start_pos] + delivery_order
        optimal_dist = sum(
            math.sqrt(
                (waypoints[i + 1][0] - waypoints[i][0]) ** 2
                + (waypoints[i + 1][1] - waypoints[i][1]) ** 2
            )
            for i in range(len(waypoints) - 1)
        )
        route_efficiency = optimal_dist / sim.timestep
    else:
        route_efficiency = 0.0

    return {
        "total_reward":          total_reward,
        "delivered":             delivered,
        "total_deliveries":      total_deliveries,
        "delivery_success_rate": delivered / max(total_deliveries, 1),
        "route_efficiency":      route_efficiency,
        "battery_remaining":     sim.battery,
        "steps":                 sim.timestep,
        "high_level_actions":    high_level_actions,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the two-layer drone delivery agent (Q-learning + A*)."
    )
    parser.add_argument(
        "--scenario", choices=["easy", "medium", "hard"], default="easy",
        help="Simulation difficulty tier (default: easy).",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable Pygame grid visualisation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the simulation (default: 42).",
    )
    parser.add_argument(
        "--qtable", default="qtable.pkl",
        help="Path to the trained Q-table pickle file (default: qtable.pkl).",
    )
    parser.add_argument(
        "--train-if-missing", action="store_true",
        help="Auto-train a Q-table on easy if none exists at --qtable.",
    )
    parser.add_argument(
        "--episodes", type=int, default=5000,
        help="Training episodes (used only with --train-if-missing).",
    )
    args = parser.parse_args()

    cfg = {**TIER_CONFIGS[args.scenario], "seed": args.seed}
    sim = CityGridSimulator(**cfg)

    agent = QLearningAgent()
    try:
        agent.load(args.qtable)
    except FileNotFoundError:
        if args.train_if_missing:
            print(
                f"No Q-table found at '{args.qtable}'.\n"
                f"Training for {args.episodes} episodes on easy …"
            )
            from simulator import TIER_CONFIGS as TC
            agent.train(
                {**TC["easy"], "seed": 0},
                num_episodes=args.episodes,
                save_path=args.qtable,
            )
        else:
            print(
                f"No Q-table at '{args.qtable}'.\n"
                "Run:  python qlearning.py  to train, or pass --train-if-missing."
            )
            return

    print(f"\nRunning integrated episode  "
          f"scenario={args.scenario}  seed={args.seed}")
    metrics = run_integrated_episode(sim, agent, render=args.render)

    print("\n── Episode Results ──────────────────────────────────────────")
    print(f"  Deliveries:          {metrics['delivered']}/{metrics['total_deliveries']}")
    print(f"  Delivery success:    {metrics['delivery_success_rate']*100:.1f}%")
    print(f"  Total reward:        {metrics['total_reward']:.1f}")
    print(f"  Route efficiency:    {metrics['route_efficiency']:.3f}")
    print(f"  Battery remaining:   {metrics['battery_remaining']:.1f}%")
    print(f"  Simulator steps:     {metrics['steps']}")
    print(f"  High-level actions:  {metrics['high_level_actions']}")


if __name__ == "__main__":
    main()
