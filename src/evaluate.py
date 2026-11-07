"""Score every saved model on the plain game, with reward rules OFF, over 100 attempts.
Also records how each attempt ended, since score alone can hide hovering.
Usage: python src/evaluate.py  (writes results/eval.csv)"""
import glob, os, re
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


def run(model, episodes=100, **env_kw):
    env = gym.make("LunarLander-v3", **env_kw)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        total, frames, fuel, done = 0.0, 0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += r; frames += 1; fuel += int(action) == 2
            done = term or trunc
        rows.append(dict(score=total, frames=frames, main_burns=fuel,
                         outcome="timeout" if trunc else "landed" if r == 100 else "crashed"))
    env.close()
    return pd.DataFrame(rows)


def summary(df):
    return dict(score=df.score.mean(), score_sd=df.score.std(), frames=df.frames.mean(),
                main_burns=df.main_burns.mean(), solved=(df.score >= 200).mean(),
                landed=(df.outcome == "landed").mean(), crashed=(df.outcome == "crashed").mean(),
                timeout=(df.outcome == "timeout").mean())


if __name__ == "__main__":
    out = []
    for path in sorted(glob.glob("models/*_seed*.zip")):
        rule, seed = re.match(r"(.+)_seed(\d+)", os.path.basename(path)).groups()
        out.append(dict(rule=rule, seed=int(seed), **summary(run(PPO.load(path, device="cpu")))))
    df = pd.DataFrame(out)
    df.to_csv("results/eval.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print(df.groupby("rule").mean(numeric_only=True).drop(columns="seed").round(2).to_string())
