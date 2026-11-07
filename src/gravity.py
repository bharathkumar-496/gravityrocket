"""My twist: test rockets trained at the default gravity (-10) on gravity they never saw.
Usage: python src/gravity.py  (writes results/gravity.csv)"""
import glob
import pandas as pd
from stable_baselines3 import PPO
from evaluate import run, summary

out = []
for path in sorted(glob.glob("models/baseline_seed*.zip")):
    model = PPO.load(path, device="cpu")
    for g in [-2, -4, -6, -8, -10, -11.99]:
        out.append(dict(model=path, gravity=g, **summary(run(model, gravity=g))))
df = pd.DataFrame(out)
df.to_csv("results/gravity.csv", index=False)
print(df.groupby("gravity")[["score", "solved", "landed", "crashed", "timeout", "frames"]].mean().round(2).to_string())
