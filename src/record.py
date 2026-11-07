"""Record one attempt as a GIF. Usage: python src/record.py hover_hack_seed1 out.gif [--seed N] [--gravity G]"""
import argparse
import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("out")
p.add_argument("--seed", type=int, default=10_000); p.add_argument("--gravity", type=float, default=-10)
a = p.parse_args()
env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=a.gravity)
model = None if a.model == "random" else PPO.load(f"models/{a.model}", device="cpu")
obs, _ = env.reset(seed=a.seed)
frames, total, done = [env.render()], 0, False
while not done:
    act = env.action_space.sample() if model is None else int(model.predict(obs, deterministic=True)[0])
    obs, r, term, trunc, _ = env.step(act)
    frames.append(env.render()); total += r; done = term or trunc
imageio.mimsave(a.out, frames[::2], fps=25, loop=0)
print(f"{a.out}: {len(frames)} frames, score {total:.0f}, last reward {r:.0f}, truncated {trunc}")
