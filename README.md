# Delivery Drone Route Planner

A two-layer AI system for autonomous drone delivery routing on a discretized city grid.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive frontend (opens in your browser)
python3 -m streamlit run app.py

# 3. Train the Q-learning agent (saves qtable.pkl)
python3 qlearning.py --tier easy --episodes 5000

# 4. Run the benchmark (requires a trained qtable.pkl)
python3 benchmark.py

# 5. Run unit tests
pytest test_astar.py -v

# 6. Watch one episode with a Pygame visualisation
python3 integrate.py --scenario easy --render
```

---

## Overview

This project implements a full autonomous delivery drone system that combines two classical AI planning paradigms:

- **Layer 1 (Low-Level) — A\* Search**: Computes the optimal spatial path between any two grid cells, respecting obstacles, no-fly zones, and battery budgets.
- **Layer 2 (High-Level) — MDP + Q-Learning**: Makes high-level dispatch decisions — which delivery to attempt next, when to recharge — by learning an optimal policy through trial-and-error interaction with the simulation environment.

The two layers work in concert: Q-learning selects the *destination*, A\* finds the *route* to get there.

---

## Project Structure

```
.
├── astar.py          # Phase 1 — A* pathfinder + dynamic replanning
├── test_astar.py     # Phase 1 — Unit tests
├── simulator.py      # Phase 2 — City grid simulation environment
├── baselines.py      # Phase 2 — Greedy and random baseline policies
├── mdp.py            # Phase 3 — MDP formulation
├── qlearning.py      # Phase 3 — Tabular Q-learning agent
├── integrate.py      # Phase 3 — Two-layer system integration
├── benchmark.py      # Phase 3 — Evaluation and results
└── README.md
```

---

## Phase 1 — A\* Pathfinder (`astar.py`)

### AI Theory: A\* Search

A\* is a best-first graph search algorithm that finds the shortest path between two nodes while guaranteeing optimality, provided the heuristic is **admissible** (never overestimates the true cost).

**The evaluation function:**

```
f(n) = g(n) + h(n)
```

- `g(n)` — actual cost accumulated from the start node to node `n`
- `h(n)` — heuristic estimate of cost from `n` to the goal
- `f(n)` — estimated total path cost through `n`

A\* expands nodes in order of increasing `f(n)`, guaranteed to find the optimal path when `h` is admissible.

**Why Euclidean distance?**  
The city grid uses 8-directional movement. The Euclidean distance between two cells is always ≤ the actual path cost (because we can only step one cell at a time in any direction), making it both **admissible** and **consistent** (monotone). Consistency ensures that once a node is expanded, we have found the optimal cost to it — so the algorithm never re-expands nodes.

**Wind-penalized edge costs:**

Each edge `(u → v)` has cost:

```
edge_cost(u, v) = euclidean_distance(u, v) × wind_factor(u, v)
```

Wind factors `∈ [0.5, 2.0]` scale battery consumption. Higher wind = more battery burned per unit of distance. This is a real-world-motivated cost model that affects pathfinding: A\* may route the drone around high-wind zones even if a shorter path exists through them.

**Battery constraint:**  
Any partial path whose accumulated cost exceeds `battery_remaining` is pruned during search. If the goal cannot be reached within the budget, the function returns `None`.

**Dynamic replanning:**  
Mid-flight, if a moving obstacle or newly declared no-fly zone intersects the drone's planned route, `replan()` detects the conflict and immediately re-invokes A\* from the drone's current position to the original goal — no human intervention needed.

### Key API

```python
from astar import find_path, replan

# Find a path
result = find_path(
    grid,             # 2D numpy array: 0=open, 1=obstacle
    start,            # (row, col)
    goal,             # (row, col)
    no_fly_zones,     # set of (row, col)
    wind_map,         # 2D numpy array or dict {(from,to): factor}
    battery_remaining # float — max allowed path cost
)
# Returns: (path, battery_cost)  or  None

# Mid-flight replanning
result = replan(
    grid, current_pos, goal,
    no_fly_zones, wind_map, battery_remaining,
    planned_path      # previously computed path
)
# Returns: (new_path_from_current, cost)  or  None
```

### Running Tests

```bash
pip install numpy pytest
pytest test_astar.py -v
```

The test suite covers:
- Shortest path on open grid (cardinal and diagonal)
- Obstacle rerouting through a vertical wall
- No-fly zone avoidance (including goal/start in no-fly zone)
- Battery-constrained failure and success cases
- Wind factor effects on battery cost
- Wind-induced failure that would otherwise succeed
- `wind_map=None` default behaviour
- Dict-format wind maps
- Path contiguity and validity invariants
- `replan()`: clean path (no replan needed), new grid obstacle, new no-fly zone, off-plan drone position, battery constraint on replanned path

---

## Phase 2 — Simulation Environment + Baselines (`simulator.py`, `baselines.py`)

### AI Theory: Stochastic Environment Modelling

Phase 2 wraps the pathfinder in a full **Markov Decision Process environment** — the world the learning agent trains and evaluates in.  Several AI/simulation principles are applied here:

**Procedural map generation with a fixed seed**  
Every episode is generated by a seeded NumPy RNG, so results are fully reproducible.  The seed controls placement of every entity and all obstacle movement, enabling apples-to-apples comparison across policies.

**Stochastic wind model**  
Wind factors are sampled from a Gaussian distribution N(1.0, 0.2) and clipped to [0.5, 2.0], then stored as a 2-D array compatible with `astar.find_path()`.  Wind introduces non-determinism in battery consumption: the same path costs different amounts of battery across episodes, forcing policies to reason under uncertainty.

**Moving obstacles as adversarial dynamics**  
Each obstacle takes a random valid step every timestep, creating a partially observable hazard.  This tests the dynamic replanning capability of A* (Phase 1) and the robustness of the Q-learning policy (Phase 3).

**Reward shaping for RL**  
The reward function is designed to incentivise the right behaviours:
- `−1` per timestep → encourages efficiency
- `+100` per on-time delivery → incentivises completing jobs
- `−50` for battery crash or obstacle collision → discourages risky decisions

### Grid Environment

| Tier | Grid | Deliveries | No-fly zones | Chargers | Obstacles | Max steps |
|------|------|-----------|--------------|----------|-----------|-----------|
| Easy | 10×10 | 5 | 5 | 2 | 3 | 200 |
| Medium | 20×20 | 10 | 15 | 4 | 6 | 800 |
| Hard | 50×50 | 20 | 50 | 10 | 15 | 4 000 |

### Key API

```python
from simulator import CityGridSimulator

sim = CityGridSimulator(
    grid_size=10, num_deliveries=5, num_no_fly_zones=5,
    num_charging_stations=2, num_moving_obstacles=3, seed=42
)

state = sim.reset()          # dict with drone_pos, battery_level, deliveries, …
state, reward, done, info = sim.step(action)   # action: 0–7 move, 8 stay
sim.render()                 # Pygame visualisation (optional)
valid = sim.get_valid_actions()
```

**State dict keys:** `drone_pos`, `battery_level`, `deliveries`, `timestep`, `no_fly_zones`, `charging_stations`, `moving_obstacles`, `wind_map`, `grid`, `grid_size`, `max_steps`.

### Baselines

```python
from baselines import GreedyNearestNeighborPolicy, RandomPolicy

greedy = GreedyNearestNeighborPolicy()  # nearest dropoff, recharge < 20 %
random = RandomPolicy(seed=0)           # uniform random valid action

action = greedy.select_action(state)
action = random.select_action(state)
```

### Benchmark Suite

```bash
python3 simulator.py        # generates benchmarks.pkl (100 scenarios)
```

`benchmarks.pkl` is a list of 100 dicts, each with keys `tier`, `seed`, and `config`.  Pass `config` as `**kwargs` to `CityGridSimulator` to reproduce any scenario exactly:

```python
import pickle
from simulator import CityGridSimulator

with open("benchmarks.pkl", "rb") as f:
    scenarios = pickle.load(f)

sim = CityGridSimulator(**scenarios[0]["config"])
```

### Render legend (Pygame)

| Colour | Entity |
|--------|--------|
| Blue | Drone |
| Red | No-fly zone |
| Green | Charging station |
| Yellow | Pending delivery dropoff |
| Orange | Moving obstacle |

---

## Phase 3 — MDP + Q-Learning + Integration + Benchmarking

### AI Theory: Reinforcement Learning via Q-Learning

Phase 3 implements the high-level dispatch intelligence using **tabular Q-learning**, a model-free reinforcement learning algorithm that learns an optimal policy through direct interaction with the environment.

**The Q-learning update rule:**

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
```

- `Q(s, a)` — estimated long-term value of taking action `a` in state `s`
- `α = 0.1` — learning rate; controls how strongly new experience overwrites old estimates
- `r` — immediate reward from the simulator
- `γ = 0.95` — discount factor; future rewards count almost as much as immediate rewards
- `max_a' Q(s', a')` — best possible value from the next state (the Bellman optimality target)

**Why Q-learning here?**  
The dispatch problem is a classic Markov Decision Process: the drone's best next action depends only on the current state (position, battery, remaining deliveries, time), not on the full history. Q-learning provably converges to the optimal Q-function for finite MDPs, given sufficient exploration. The key challenge is the state space explosion with many deliveries — hence tabular Q-learning is used with a sparse `defaultdict` representation.

**MDP State space:**

| Component | Encoding |
|-----------|----------|
| `drone_pos` | (row, col) — discrete by nature |
| `battery_bin` | int 0–9 (each bin = 10% of 100) |
| `remaining_deliveries` | `frozenset` of pending delivery IDs |
| `time_bin` | int 0–9 (fraction of `max_steps` elapsed) |

**High-level actions:**

| Action | Effect |
|--------|--------|
| `fly_to_delivery_i` | A* routes drone to delivery i's dropoff |
| `fly_to_charging_station` | A* routes drone to nearest charger |
| `skip_delivery_i` | Mark delivery i as failed; conserve battery |

**ε-greedy exploration:**  
ε decays linearly from 1.0 (fully random) to 0.05 (mostly greedy) over 5 000 training episodes. Early random exploration fills the Q-table with diverse experiences; later, the greedy policy exploits the learned values.

**Two-layer feedback loop:**  
After each high-level action, A* returns the actual battery consumed. This updates `sim.battery`, which is re-encoded into the next MDP state — closing the feedback loop between the planning layer (Q-learning) and the execution layer (A*).

### Key API

```python
from qlearning import QLearningAgent
from simulator import TIER_CONFIGS

agent = QLearningAgent(alpha=0.1, gamma=0.95)
agent.train(TIER_CONFIGS["easy"], num_episodes=5000)   # saves qtable.pkl
agent.load("qtable.pkl")

# High-level action selection (exploit=True disables exploration)
action = agent.select_action(state, valid_actions, exploit=True)
reward = agent.execute_action(sim, action)   # A* path execution
```

### Running the Integrated Agent

```bash
# Train (saves qtable.pkl + training_curve.png)
python3 qlearning.py --tier easy --episodes 5000

# Run one episode with Pygame render
python3 integrate.py --scenario easy --render

# Auto-train if no Q-table exists
python3 integrate.py --scenario medium --train-if-missing
```

### Benchmarking

```bash
python3 benchmark.py              # requires pre-trained qtable.pkl
python3 benchmark.py --train-episodes 5000   # auto-trains if needed
```

Produces `results.csv` (per-policy per-tier metrics) and `results.png` (2×2 bar chart).

**Results across 100 scenarios per tier:**

| Policy | Tier | Delivery Success | Route Efficiency | Battery Remaining | Fleet Throughput |
|--------|------|-----------------|-----------------|-------------------|-----------------|
| Q-Learning | Easy | **83.5%** | 1.019 | 79.9 | 4.18 |
| Q-Learning | Medium | 54.8% | 0.809 | 64.4 | 5.48 |
| Q-Learning | Hard | 15.6% | 0.431 | 42.3 | 3.12 |
| Greedy | Easy | 82.9% | 1.000 | 85.8 | 4.15 |
| Greedy | Medium | 67.9% | 0.961 | 67.5 | 6.79 |
| Greedy | Hard | **66.1%** | **0.996** | **48.7** | **13.2** |
| Random | Easy | 20.0% | 0.112 | 75.2 | 1.00 |
| Random | Medium | 6.4% | 0.031 | 47.1 | 0.64 |
| Random | Hard | 1.4% | 0.010 | 17.7 | 0.27 |

Key observations:
- Q-Learning edges out Greedy on easy tier (83.5% vs 82.9%), demonstrating that the learned policy can match and slightly surpass a hand-crafted heuristic.
- Greedy outperforms Q-Learning on medium/hard tiers — a known limitation of tabular Q-learning, where the state space grows exponentially with grid size and delivery count, requiring more training episodes for full generalisation.
- Both learned and heuristic policies vastly outperform the Random baseline across all tiers.

### Training Curve

`training_curve.png` shows the 100-episode rolling-average reward during Q-learning training, illustrating the progression from random exploration to learned dispatch.

---

## How It Was Built

### Phase 3 (MDP + Q-Learning + Integration + Benchmarking) — `mdp.py`, `qlearning.py`, `integrate.py`, `benchmark.py`

Built by **Neel**. Design decisions:

- **Sparse Q-table via `defaultdict`** — unvisited `(state, action)` pairs default to Q=0. For tabular Q-learning, this gives O(1) lookup and only occupies memory for states actually visited during training.
- **Episode seed variation** — each training episode uses `seed = base_seed + episode`, so the agent trains on varied city layouts. This is critical for the Q-table to generalise beyond a single fixed map.
- **A* called once per high-level action, not per step** — `_execute_to_target` plans the full path upfront and only calls `replan()` when the path is blocked by a moving obstacle. This keeps training fast without sacrificing dynamic re-routing.
- **SKIP action as a "cut losses" mechanism** — the agent can voluntarily abandon a delivery (mark it failed) to conserve battery for higher-value deliveries or charging. This is an important capability for the RL agent to learn when deliveries are far apart or time windows are expiring.
- **Benchmark design** — 100 scenarios across 3 tiers provide statistical significance while remaining computationally tractable. The Q-table trained on easy scenarios extrapolates to medium/hard, demonstrating both the capability and limits of tabular Q-learning.

### Phase 2 (Simulation + Baselines) — `simulator.py`, `baselines.py`

Built by **Lawrence**. Design decisions:

- **Single seeded RNG per episode** — all randomness (wind, placement, obstacle movement) flows from one `np.random.default_rng(seed)`.  The obstacle-movement RNG is forked separately so placement and dynamics don't interfere with each other.
- **Wind as a numpy 2-D array** — `wind_map[r][c]` is the factor at destination cell `(r,c)`, directly compatible with `astar._wind_factor()` without any conversion.
- **Battery instant-fill on arrival** — when the drone lands on a charging station, battery resets to 100.0 immediately, keeping the decision logic clean for both baselines and the Q-learner.
- **No fixed grid obstacles** — the grid stays all-zeros; no-fly zones carry the blocking role.  This means A* can always rely on `grid == 0` as the navigability signal, while no-fly zones add the dynamic hard-block layer.
- **`GreedyNearestNeighborPolicy._best_valid_direction`** — picks the 8-directional move with the highest dot-product with the vector to the target, filtering out NFZ cells and current obstacle positions.  This avoids the drone immediately walking into a wall.

### Phase 1 (A\* + Dynamic Replanning) — `astar.py`

Built by **Srikar**. Design decisions:

- **8-directional movement** gives the drone realistic freedom of motion; diagonal moves cost √2 vs. 1.0 for cardinal steps.
- **Tie-breaking** in the priority queue (monotone counter) prevents heap instability when multiple nodes share the same `f` value.
- **Stale-entry detection** avoids re-processing nodes when a shorter path is discovered after they were enqueued.
- `wind_map` accepts both a 2D numpy array (one factor per destination cell) and a dict (true per-edge granularity), making it forward-compatible with the simulator's edge-level wind model.
- `replan()` is a thin wrapper: it checks validity of the remaining planned segment and only calls `find_path` again when needed, keeping re-routing latency proportional to actual change.

---

## Interactive Frontend (`app.py`)

A Streamlit web app that lets you watch any policy control the drone in real time.

### Setup and Launch

```bash
pip install streamlit
python3 -m streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### Sidebar Controls

| Control | What it does |
|---------|-------------|
| **Difficulty tier** | Switches between easy (10×10), medium (20×20), and hard (50×50) grids. Changing tier resets the episode. |
| **Policy** | Selects which AI drives the drone: Q-Learning (trained agent), Greedy (nearest-neighbour heuristic), or Random. |
| **Seed** | Determines the procedurally generated map — same seed always produces the same city layout, wind map, delivery locations, and obstacle starting positions. Change it to explore different scenarios. |
| **Reset episode** | Starts a fresh episode with the current tier/policy/seed combination. |
| **Auto-play toggle** | When on, the app automatically steps through the episode without you pressing anything. |
| **Steps / second** | Controls auto-play speed (1–10 steps per second). |

### Grid Visualisation

The coloured grid is the live city map, redrawn after every action:

| Colour | Entity | What to watch |
|--------|--------|--------------|
| **Blue ▲** | Drone | Tracks the drone's current position; the triangle points in the direction of the last move. |
| **Red** | No-fly zone | Hard-blocked cells — A* will always route around these, even if it means a longer path. |
| **Green** | Charging station | Drone must land here to refill battery to 100%. Watch whether Q-Learning proactively routes to chargers before the battery runs out. |
| **Yellow** | Pending delivery | A dropoff cell the drone hasn't reached yet. |
| **Light green** | Completed delivery | Dropoff cell the drone reached within the time window — reward +100 was earned. |
| **Pink/red** | Failed delivery | Drone skipped this delivery or the time window expired before arrival. |
| **Orange** | Moving obstacle | Relocates one cell per timestep. If the drone's planned A* path is blocked by an obstacle, the path is automatically replanned from the drone's current position. |

### Stats Panel (right side)

- **Battery gauge** — colour-coded bar: green > 40%, orange 20–40%, red < 20%. Shows how aggressively the policy consumes power. Q-Learning may accept more battery spend per delivery than Greedy if it learned that recharging mid-route is too costly.
- **Timestep counter** — how far through the episode budget the drone is. Hard tier has 4,000 steps; watch whether any policy runs out of time before finishing all deliveries.
- **Cumulative reward** — running total of all rewards since episode start. Each timestep costs −1; each on-time delivery earns +100; a battery crash or obstacle collision costs −50.
- **Deliveries table** — shows every delivery's ID, dropoff cell, time window (earliest–latest step), and current status icon (🟡 pending, ✅ delivered, ❌ failed).

### Step Log

The last 20 actions are logged in a table showing step number, reward earned, battery level, and delivery counts. This makes it easy to spot exactly when a delivery was completed (+100 reward spike) or when the drone crashed (−50 reward).

### What to Look for by Policy

**Q-Learning**
- On easy tier, watch the drone plan multi-step routes that sequence deliveries efficiently rather than taking the geometrically nearest one each time.
- At low battery, the Q-learner sometimes skips a distant delivery entirely (SKIP action) to preserve battery for closer ones — a behaviour it learned because running out of battery costs more in the long run.
- On medium/hard tiers you may see suboptimal decisions; the Q-table was trained on easy maps, so harder grids expose generalisation limits of tabular RL.

**Greedy**
- The drone always targets the nearest pending delivery dropoff, and recharges when battery drops below 20%. This is a strong, interpretable baseline — compare its route efficiency against Q-Learning.
- On hard tier (large grid, many deliveries), Greedy's simple nearest-neighbour heuristic achieves 66% delivery success vs Q-Learning's 16%, because it doesn't need a pre-trained Q-table to handle unseen grid sizes.

**Random**
- Moves are sampled uniformly from the 8 directions + stay. The drone wanders aimlessly, rarely reaching a delivery within its time window. Useful as a lower-bound sanity check — any trained policy should beat it by a large margin.

### Connecting the Frontend to the AI Code

Each ▶ Step click triggers the full two-layer decision cycle:

1. **High-level action** — the selected policy (`qlearning.py`, `baselines.py`) chooses *where* to go (which delivery, which charger, or skip).
2. **Low-level path** — `astar.find_path()` computes the optimal route from the drone's current cell to that target, respecting no-fly zones, wind costs, and battery budget.
3. **Execution** — the drone follows the path one cell per `sim.step()` call. After each cell, `astar.replan()` checks if a moving obstacle has entered the remaining route; if so, A* reruns from the current position.
4. **State update** — battery, drone position, delivery statuses, and timestep are updated by the simulator and reflected immediately in the grid and stats panel.

---

## Dependencies

```
numpy
pandas
matplotlib
pygame
pytest
streamlit
```

Install all with:

```bash
pip install numpy pandas matplotlib pygame pytest streamlit
```

---

## Team

| Phase | Owner | Module |
|-------|-------|--------|
| 1 — A\* Pathfinder | Srikar | `astar.py` |
| 2 — Simulation + Baselines | Lawrence | `simulator.py`, `baselines.py` |
| 3 — MDP + Q-Learning + Benchmarking | Neel | `mdp.py`, `qlearning.py`, `integrate.py`, `benchmark.py` |
