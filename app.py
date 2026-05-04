"""
app.py — Streamlit frontend for the Delivery Drone Route Planner.

Run with:
    streamlit run app.py

Features
--------
- Pick tier (easy/medium/hard), policy (Q-Learning/Greedy/Random), and seed
- Step through an episode one action at a time, or auto-play
- Live grid visualization with colour-coded entities
- Battery gauge, delivery status table, and per-step reward log
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drone Delivery Route Planner",
    page_icon="🚁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Imports from the project (after set_page_config to avoid double-init)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from simulator import CityGridSimulator, TIER_CONFIGS, DIRECTIONS, STAY
from baselines import GreedyNearestNeighborPolicy, RandomPolicy

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_BG       = np.array([0.95, 0.95, 0.95])   # open cell
C_NFZ      = np.array([0.85, 0.20, 0.20])   # no-fly zone (red)
C_CHARGE   = np.array([0.20, 0.75, 0.35])   # charging station (green)
C_PENDING  = np.array([1.00, 0.85, 0.10])   # pending delivery (yellow)
C_DONE     = np.array([0.60, 0.90, 0.60])   # delivered (light green)
C_FAILED   = np.array([1.00, 0.55, 0.55])   # failed delivery (pink)
C_OBSTACLE = np.array([1.00, 0.55, 0.15])   # moving obstacle (orange)
C_DRONE    = np.array([0.15, 0.40, 0.85])   # drone (blue)


def _render_grid(sim: CityGridSimulator) -> plt.Figure:
    n = sim.grid_size
    img = np.tile(C_BG, (n, n, 1))

    for r, c in sim.no_fly_zones:
        img[r, c] = C_NFZ
    for r, c in sim.charging_stations:
        img[r, c] = C_CHARGE
    for d in sim.deliveries:
        if d.status == "pending":
            img[d.dropoff[0], d.dropoff[1]] = C_PENDING
        elif d.status == "delivered":
            img[d.dropoff[0], d.dropoff[1]] = C_DONE
        elif d.status == "failed":
            img[d.dropoff[0], d.dropoff[1]] = C_FAILED
    for obs in sim.moving_obstacles:
        img[obs[0], obs[1]] = C_OBSTACLE

    dr, dc = sim.drone_pos
    img[dr, dc] = C_DRONE

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, origin="upper", interpolation="nearest")

    # Grid lines
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.4)
        ax.axvline(i - 0.5, color="white", linewidth=0.4)

    # Drone marker
    ax.plot(dc, dr, marker="^", markersize=max(6, 14 - n // 5),
            color="white", markeredgecolor="black", markeredgewidth=0.8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Step {sim.timestep} / {sim.max_steps}", fontsize=11)

    legend_items = [
        mpatches.Patch(color=C_DRONE,    label="Drone"),
        mpatches.Patch(color=C_NFZ,      label="No-fly zone"),
        mpatches.Patch(color=C_CHARGE,   label="Charging station"),
        mpatches.Patch(color=C_PENDING,  label="Delivery (pending)"),
        mpatches.Patch(color=C_DONE,     label="Delivery (done)"),
        mpatches.Patch(color=C_FAILED,   label="Delivery (failed)"),
        mpatches.Patch(color=C_OBSTACLE, label="Moving obstacle"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Q-Learning agent (lazy-loaded, cached so it's not re-read every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_q_agent():
    from qlearning import QLearningAgent
    agent = QLearningAgent()
    qtable = os.path.join(os.path.dirname(__file__), "qtable.pkl")
    if os.path.exists(qtable):
        agent.load(qtable)
    return agent


def _ql_step(sim: CityGridSimulator) -> float:
    from mdp import encode_state, get_valid_actions
    agent = _load_q_agent()
    s = encode_state(sim.drone_pos, sim.battery, sim.deliveries, sim.timestep, sim.max_steps)
    valid = get_valid_actions(s, sim.deliveries, sim.charging_stations)
    if not valid:
        _, r, _, _ = sim.step(STAY)
        return r
    action = agent.select_action(s, valid, exploit=True)
    return agent.execute_action(sim, action)


def _greedy_step(sim: CityGridSimulator) -> float:
    policy = GreedyNearestNeighborPolicy()
    state = sim._make_state()
    action = policy.select_action(state)
    _, r, _, _ = sim.step(action)
    return r


def _random_step(sim: CityGridSimulator) -> float:
    policy = RandomPolicy(seed=sim.timestep)
    state = sim._make_state()
    action = policy.select_action(state)
    _, r, _, _ = sim.step(action)
    return r


STEP_FN = {
    "Q-Learning": _ql_step,
    "Greedy":     _greedy_step,
    "Random":     _random_step,
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _init_sim(tier: str, policy: str, seed: int) -> None:
    cfg = {**TIER_CONFIGS[tier], "seed": seed}
    sim = CityGridSimulator(**cfg)
    sim.reset()
    st.session_state.sim        = sim
    st.session_state.tier       = tier
    st.session_state.policy     = policy
    st.session_state.seed       = seed
    st.session_state.rewards    = []
    st.session_state.total_reward = 0.0
    st.session_state.log        = []


def _needs_reset(tier, policy, seed) -> bool:
    return (
        "sim" not in st.session_state
        or st.session_state.tier   != tier
        or st.session_state.policy != policy
        or st.session_state.seed   != seed
    )


def _do_step() -> None:
    sim: CityGridSimulator = st.session_state.sim
    if sim.done:
        return
    fn = STEP_FN[st.session_state.policy]
    r  = fn(sim)
    st.session_state.total_reward += r
    st.session_state.rewards.append(r)
    delivered = sum(1 for d in sim.deliveries if d.status == "delivered")
    failed    = sum(1 for d in sim.deliveries if d.status == "failed")
    st.session_state.log.append({
        "step":    sim.timestep,
        "reward":  f"{r:+.1f}",
        "battery": f"{sim.battery:.1f}",
        "done":    f"{delivered}✓  {failed}✗",
    })


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🚁 Delivery Drone Route Planner")
st.caption("Two-layer AI: Q-Learning dispatch + A* pathfinding")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    tier   = st.selectbox("Difficulty tier", ["easy", "medium", "hard"])
    policy = st.selectbox("Policy", ["Q-Learning", "Greedy", "Random"])
    seed   = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)

    if policy == "Q-Learning" and not os.path.exists(
        os.path.join(os.path.dirname(__file__), "qtable.pkl")
    ):
        st.warning("qtable.pkl not found — train first:\n```\npython3 qlearning.py\n```")

    st.divider()
    if st.button("Reset episode", use_container_width=True):
        _init_sim(tier, policy, seed)

    auto_play  = st.toggle("Auto-play", value=False)
    play_speed = st.slider("Steps / second", 1, 10, 3)

    st.divider()
    st.subheader("Tier config")
    cfg = TIER_CONFIGS[tier]
    for k, v in cfg.items():
        st.write(f"**{k.replace('_', ' ').title()}:** {v}")

# Auto-init on first load or config change
if _needs_reset(tier, policy, seed):
    _init_sim(tier, policy, seed)

sim: CityGridSimulator = st.session_state.sim

# ── Main layout ──────────────────────────────────────────────────────────────
col_grid, col_stats = st.columns([3, 2])

with col_grid:
    grid_placeholder = st.empty()
    grid_placeholder.pyplot(_render_grid(sim))

with col_stats:
    # Battery gauge
    bat_pct = sim.battery / 100.0
    bat_color = "#2ecc71" if bat_pct > 0.4 else "#e67e22" if bat_pct > 0.2 else "#e74c3c"
    st.markdown(f"**Battery:** {sim.battery:.1f}%")
    st.markdown(
        f'<div style="background:#ddd;border-radius:4px;height:14px;">'
        f'<div style="width:{bat_pct*100:.0f}%;background:{bat_color};height:14px;border-radius:4px;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Timestep:** {sim.timestep} / {sim.max_steps}")
    st.markdown(f"**Cumulative reward:** {st.session_state.total_reward:+.1f}")

    # Delivery status table
    st.subheader("Deliveries")
    for d in sim.deliveries:
        icon = {"pending": "🟡", "delivered": "✅", "failed": "❌"}.get(d.status, "?")
        st.markdown(f"{icon} **#{d.id}** dropoff {d.dropoff}  "
                    f"window [{d.time_window[0]}–{d.time_window[1]}]")

    # Step log
    if st.session_state.log:
        st.subheader("Step log")
        import pandas as pd
        df = pd.DataFrame(st.session_state.log[-20:]).set_index("step")
        st.dataframe(df, use_container_width=True, height=200)

# ── Controls ─────────────────────────────────────────────────────────────────
ctrl1, ctrl2 = st.columns([1, 4])
with ctrl1:
    step_btn = st.button("▶ Step", disabled=sim.done, use_container_width=True)
with ctrl2:
    if sim.done:
        delivered = sum(1 for d in sim.deliveries if d.status == "delivered")
        total     = len(sim.deliveries)
        st.success(f"Episode done — {delivered}/{total} deliveries · total reward {st.session_state.total_reward:+.1f}")

if step_btn:
    _do_step()
    st.rerun()

# Auto-play loop
if auto_play and not sim.done:
    _do_step()
    grid_placeholder.pyplot(_render_grid(sim))
    time.sleep(1.0 / play_speed)
    st.rerun()
