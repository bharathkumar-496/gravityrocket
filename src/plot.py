"""Learning curves: score PPO was chasing and attempt length, per rule, averaged over seeds."""
import glob, re, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"baseline": "#2a78d6", "time_fee": "#1baa7d", "hover_hack": "#e0482f", "hover_fix": "#8a55d6"}
LABELS = {"baseline": "Built-in reward", "time_fee": "Time fee", "hover_hack": "+1 per steady frame (broken)",
          "hover_fix": "Pay for change + fee (fixed)"}
grid = np.arange(25_000, 1_000_001, 25_000)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
for rule, c in COLORS.items():
    rs, ls = [], []
    for d in sorted(glob.glob(f"logs/{rule}_seed*")):
        df = pd.concat([pd.read_csv(f, skiprows=1) for f in glob.glob(f"{d}/*.monitor.csv")]).sort_values("t")
        steps = df.l.cumsum().values
        rs.append(np.interp(grid, steps, df.r.rolling(300, min_periods=1).mean()))
        ls.append(np.interp(grid, steps, df.l.rolling(300, min_periods=1).mean()))
    for ax, ys in zip(axes, (rs, ls)):
        ys = np.array(ys)
        ax.plot(grid / 1e6, ys.mean(0), color=c, lw=2, label=LABELS[rule])
        ax.fill_between(grid / 1e6, ys.min(0), ys.max(0), color=c, alpha=.12, lw=0)
axes[0].set(title="Score PPO was chasing (each rule's own points)", xlabel="Training steps (millions)", ylabel="Score per attempt")
axes[1].set(title="How long each attempt lasts", xlabel="Training steps (millions)", ylabel="Frames (cut off at 1,000)")
axes[0].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.2)
plt.tight_layout(); plt.savefig("results/learning_curves.png")
