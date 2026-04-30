"""
benchmark.py — Phase 3: Policy Evaluation and Reporting

Loads the 100-scenario benchmark suite (benchmarks.pkl) and evaluates three
policies on every scenario:

  1. Q-Learning + A*  (the integrated two-layer agent)
  2. Greedy Nearest-Neighbour baseline
  3. Random-action baseline

Metrics collected per scenario
-------------------------------
delivery_success_rate  : fraction of deliveries completed on time
route_efficiency       : straight-line optimal distance / actual steps
                         (higher → more direct routing; ≤ 1.0)
battery_remaining      : battery % at episode end (higher → more conservative)
fleet_throughput       : deliveries completed per episode (count)

Output
------
results.csv  : per-policy, per-tier mean metrics table
results.png  : 2×2 bar-chart grid comparing policies across tiers

Usage
-----
    # After training:  python qlearning.py
    python benchmark.py

    # Auto-train if no Q-table exists:
    python benchmark.py --train-episodes 5000
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Any, Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import GreedyNearestNeighborPolicy, RandomPolicy
from integrate import run_integrated_episode
from mdp import encode_state, get_valid_actions
from qlearning import QLearningAgent
from simulator import CityGridSimulator, TIER_CONFIGS


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def _euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _route_efficiency(
    start:             Tuple[int, int],
    delivery_order:    List[Tuple[int, int]],
    total_steps:       int,
) -> float:
    """Ratio of optimal waypoint path to actual steps taken.

    Optimal path = straight-line distances: start → dropoff_1 → dropoff_2 → …
    This is the Euclidean lower-bound on any ordered delivery route.
    """
    if not delivery_order or total_steps == 0:
        return 0.0
    waypoints   = [start] + delivery_order
    optimal     = sum(_euclidean(waypoints[i], waypoints[i + 1])
                      for i in range(len(waypoints) - 1))
    return optimal / total_steps


# ---------------------------------------------------------------------------
# Per-policy episode runners
# ---------------------------------------------------------------------------

def _run_baseline(sim: CityGridSimulator, policy) -> Dict[str, Any]:
    """Evaluate one episode of a step-level baseline policy."""
    state          = sim.reset()
    start          = sim.drone_pos
    delivered_ids: Set[int] = set()
    delivery_order: List[Tuple[int, int]] = []

    while not sim.done:
        action         = policy.select_action(state)
        state, _, _, _ = sim.step(action)
        for d in sim.deliveries:
            if d.status == "delivered" and d.id not in delivered_ids:
                delivered_ids.add(d.id)
                delivery_order.append(d.dropoff)

    delivered        = len(delivered_ids)
    total            = len(sim.deliveries)
    return {
        "delivery_success_rate": delivered / max(total, 1),
        "route_efficiency":      _route_efficiency(start, delivery_order, sim.timestep),
        "battery_remaining":     sim.battery,
        "fleet_throughput":      float(delivered),
    }


def _run_qlearning(sim: CityGridSimulator, agent: QLearningAgent) -> Dict[str, Any]:
    """Evaluate one episode of the two-layer Q-learning + A* agent."""
    metrics = run_integrated_episode(sim, agent, render=False)
    return {
        "delivery_success_rate": metrics["delivery_success_rate"],
        "route_efficiency":      metrics["route_efficiency"],
        "battery_remaining":     metrics["battery_remaining"],
        "fleet_throughput":      float(metrics["delivered"]),
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "delivery_success_rate",
    "route_efficiency",
    "battery_remaining",
    "fleet_throughput",
]

METRIC_LABELS = {
    "delivery_success_rate": "Delivery Success Rate",
    "route_efficiency":      "Route Efficiency\n(optimal dist / steps)",
    "battery_remaining":     "Avg Battery Remaining (%)",
    "fleet_throughput":      "Fleet Throughput\n(deliveries / episode)",
}

POLICY_COLORS = {
    "Q-Learning": "#2196F3",
    "Greedy":     "#FF9800",
    "Random":     "#9E9E9E",
}


def run_benchmarks(
    scenarios:    List[Dict[str, Any]],
    agent:        QLearningAgent,
    results_csv:  str = "results.csv",
    results_png:  str = "results.png",
) -> pd.DataFrame:
    """Evaluate all three policies on every scenario and produce reports.

    Parameters
    ----------
    scenarios:   List loaded from benchmarks.pkl.
    agent:       Trained QLearningAgent (zero-Q table is acceptable; agent
                 will fall back to first-valid-action greedy).
    results_csv: Output path for the summary CSV.
    results_png: Output path for the bar-chart figure.

    Returns
    -------
    Pandas DataFrame with one row per (policy, tier) combination.
    """
    greedy_policy = GreedyNearestNeighborPolicy()
    random_policy = RandomPolicy(seed=7)
    records: List[Dict[str, Any]] = []
    total = len(scenarios)

    for idx, scenario in enumerate(scenarios):
        tier   = scenario["tier"]
        config = scenario["config"]
        sim    = CityGridSimulator(**config)

        records.append({"policy": "Q-Learning", "tier": tier,
                         **_run_qlearning(sim, agent)})
        records.append({"policy": "Greedy",     "tier": tier,
                         **_run_baseline(sim, greedy_policy)})
        records.append({"policy": "Random",     "tier": tier,
                         **_run_baseline(sim, random_policy)})

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            print(f"  Evaluated {idx + 1:3d}/{total} scenarios …")

    df = pd.DataFrame(records)

    summary = (
        df.groupby(["policy", "tier"])[METRIC_COLS]
        .mean()
        .reset_index()
    )

    # Preserve natural tier ordering in output
    tier_order   = ["easy", "medium", "hard"]
    policy_order = ["Q-Learning", "Greedy", "Random"]
    summary["tier_ord"]   = summary["tier"].map({t: i for i, t in enumerate(tier_order)})
    summary["policy_ord"] = summary["policy"].map({p: i for i, p in enumerate(policy_order)})
    summary = (summary
               .sort_values(["policy_ord", "tier_ord"])
               .drop(columns=["tier_ord", "policy_ord"])
               .reset_index(drop=True))

    summary.to_csv(results_csv, index=False)
    print(f"\nResults table saved → {results_csv}")
    print(summary.to_string(index=False))

    _plot_results(summary, results_png)
    return summary


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_results(summary: pd.DataFrame, save_path: str) -> None:
    """Save a 2×2 grouped bar chart comparing the three policies per tier."""
    tiers    = ["easy", "medium", "hard"]
    policies = ["Q-Learning", "Greedy", "Random"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    bar_width = 0.25
    x = np.arange(len(tiers))

    for ax, metric in zip(axes, METRIC_COLS):
        for i, policy in enumerate(policies):
            vals = []
            for tier in tiers:
                row = summary[
                    (summary["policy"] == policy) & (summary["tier"] == tier)
                ]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)

            offset = (i - 1) * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                label=policy, color=POLICY_COLORS[policy], alpha=0.85,
            )
            y_max = max(vals) if vals else 1.0
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.015 * max(y_max, 1e-6),
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tiers])
        ax.set_xlabel("Difficulty Tier")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Policy Comparison: Q-Learning vs Greedy vs Random\nacross Difficulty Tiers",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Results chart saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Q-Learning, Greedy, and Random policies."
    )
    parser.add_argument("--qtable",      default="qtable.pkl",
                        help="Path to trained Q-table (default: qtable.pkl).")
    parser.add_argument("--benchmarks",  default="benchmarks.pkl",
                        help="Benchmark scenarios file (default: benchmarks.pkl).")
    parser.add_argument("--results-csv", default="results.csv")
    parser.add_argument("--results-png", default="results.png")
    parser.add_argument(
        "--train-episodes", type=int, default=0,
        help="If > 0 and no Q-table found, auto-train this many episodes on easy.",
    )
    args = parser.parse_args()

    # Load benchmark scenarios
    with open(args.benchmarks, "rb") as fh:
        scenarios = pickle.load(fh)
    print(f"Loaded {len(scenarios)} benchmark scenarios from '{args.benchmarks}'")

    # Load or train Q-learning agent
    agent = QLearningAgent()
    if os.path.exists(args.qtable):
        agent.load(args.qtable)
    elif args.train_episodes > 0:
        print(f"No Q-table found — training for {args.train_episodes} episodes on easy …")
        agent.train(
            {**TIER_CONFIGS["easy"], "seed": 0},
            num_episodes=args.train_episodes,
            save_path=args.qtable,
        )
    else:
        raise FileNotFoundError(
            f"No Q-table at '{args.qtable}'.\n"
            "Run  python qlearning.py  first, or pass --train-episodes N."
        )

    print("\nRunning benchmark evaluation …")
    run_benchmarks(
        scenarios, agent,
        results_csv=args.results_csv,
        results_png=args.results_png,
    )
