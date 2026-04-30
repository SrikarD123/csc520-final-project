"""Standalone training script — run directly: python3 run_training.py"""
import time
from qlearning import QLearningAgent
from simulator import TIER_CONFIGS

t0 = time.time()
agent = QLearningAgent()
agent.train(
    TIER_CONFIGS["easy"],
    num_episodes=5000,
    save_path="qtable.pkl",
    plot_path="training_curve.png",
)
elapsed = time.time() - t0
print(f"Training finished in {elapsed:.1f}s ({elapsed/5000*1000:.1f}ms/episode)")
