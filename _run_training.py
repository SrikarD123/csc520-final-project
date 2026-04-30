"""Run 20 chunks of 50 episodes each, loading and saving qtable.pkl each time."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from qlearning import QLearningAgent
from simulator import TIER_CONFIGS

for i in range(1, 21):
    plot_path = 'training_curve.png' if i == 20 else '/dev/null'
    a = QLearningAgent()
    a.load('qtable.pkl')
    a.train(TIER_CONFIGS['easy'], 50, 'qtable.pkl', plot_path)
    print(f"=== Run {i}/20 complete. Q-table states: {len(a.Q)} ===", flush=True)

print("ALL 20 RUNS COMPLETE", flush=True)
