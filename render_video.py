"""Renders the greedy trained policy (seed-0 agent, best config) to results/trained_agent.mp4."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, imageio
from qlearning import make_env

Q = np.load("results/best_Q.npy")
env = make_env(render_mode="rgb_array")
frames = []
for ep in range(3):
    s, _ = env.reset(seed=100 + ep)
    frames.append(env.render())
    done = False
    while not done:
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
        frames.append(env.render())
        done = term or trunc
    frames += [frames[-1]] * 6  # pause at end of each episode
env.close()
imageio.mimsave("results/trained_agent.mp4", frames, fps=4, macro_block_size=1)
print(len(frames), "frames")
