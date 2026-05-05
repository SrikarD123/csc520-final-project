"""Single continuous 3000-episode training run — generates training_curve.png."""
import sys, time, os
sys.stdout.reconfigure(line_buffering=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import simulator as sim_mod
from qlearning import QLearningAgent

N   = 3000
cfg = sim_mod.TIER_CONFIGS['easy']

agent = QLearningAgent()
t0    = time.time()
rewards = agent.train(cfg, N, 'qtable.pkl', 'training_curve.png')
elapsed = time.time() - t0
print(f'Done: {N} eps in {elapsed:.1f}s  Q-states={len(agent.Q)}  eps={agent.epsilon:.4f}')

