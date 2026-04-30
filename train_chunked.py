"""
Chunked training driver.

Runs TOTAL_EPS episodes of Q-learning in CHUNK_SIZE-episode batches so that
each Python call stays within the synchronous-execution window of this shell.
Epsilon decays correctly across chunks via the _ep_offset / _total_eps params.
"""
import os, sys, time
from qlearning import QLearningAgent
from simulator import TIER_CONFIGS

TOTAL_EPS  = int(os.environ.get("TOTAL_EPS",  "5000"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "50"))
SAVE_PATH  = os.environ.get("SAVE_PATH", "qtable.pkl")
PLOT_PATH  = os.environ.get("PLOT_PATH", "training_curve.png")
CHUNK_IDX  = int(os.environ.get("CHUNK_IDX", "0"))

cfg = TIER_CONFIGS["easy"]
offset = CHUNK_IDX * CHUNK_SIZE
remaining = TOTAL_EPS - offset
if remaining <= 0:
    print(f"All {TOTAL_EPS} episodes already done.")
    sys.exit(0)

chunk = min(CHUNK_SIZE, remaining)

agent = QLearningAgent()
if CHUNK_IDX > 0 and os.path.exists(SAVE_PATH):
    agent.load(SAVE_PATH)

t0 = time.time()
agent.train(
    cfg, chunk, SAVE_PATH, PLOT_PATH,
    _ep_offset=offset, _total_eps=TOTAL_EPS,
)
elapsed = time.time() - t0
print(f"chunk {CHUNK_IDX}: eps {offset+1}–{offset+chunk}/{TOTAL_EPS}  {elapsed:.2f}s")
