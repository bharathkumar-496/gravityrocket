"""Train PPO on Lunar Lander under one reward rule and seed.
Usage: python src/train.py --rule hover_hack --seed 1 --steps 1000000"""
import argparse, os, time
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from rewards import RULES

p = argparse.ArgumentParser()
p.add_argument("--rule", choices=RULES, default="baseline")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--tag", default="")
a = p.parse_args()
torch.set_num_threads(1)
name = f"{a.rule}{a.tag}_seed{a.seed}"
os.makedirs(f"logs/{name}", exist_ok=True)


def make(i):
    def thunk():
        env = RULES[a.rule](gym.make("LunarLander-v3"))
        return Monitor(env, f"logs/{name}/{i}")  # logs the score PPO sees
    return thunk


env = DummyVecEnv([make(i) for i in range(16)])
# Settings from the Hugging Face Deep RL course, Unit 1.
model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999,
            gae_lambda=0.98, ent_coef=0.01, seed=a.seed, device="cpu", verbose=0)
t = time.time()
model.learn(a.steps)
model.save(f"models/{name}")
print(f"{name} done in {(time.time()-t)/60:.1f} min")
