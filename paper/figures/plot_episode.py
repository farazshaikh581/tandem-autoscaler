"""One evaluation episode, HPA+VPA vs TANDEM (paper Fig. episode):
python3 paper/figures/plot_episode.py [--workload mixed] [--episode 0]

Replays evaluation episode `--episode` of the Azure class `--workload` exactly as
train.evaluate / eval_baselines.evaluate_policy do (env seed 777, reset seed 777+ep),
with the seed-0 TANDEM model (cvar-bellman-qrdqn_all_azure) and the HPA+VPA baseline.
Each panel shows the CPU the load demands, the CPU the controller allocates, and the
intervals that miss the SLO. The mixed trace is shorter than an episode, so every
mixed episode is this one. Writes episode_<workload>.pdf/.png next to this script.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = os.path.join(HERE, "..", "..", "phase1")
sys.path.insert(0, PHASE1)
os.chdir(PHASE1)
import train  # noqa: E402
import eval_baselines as eb  # noqa: E402
from envs.joint_hv_env import (ACTIONS, CPU_DEMAND_M, INTENSITY, JointHVEnv,  # noqa: E402
                               SLA_MEAN_LATENCY_MS)


def rollout(wl, ep, model=None, ctrl=None):
    env = JointHVEnv(workload=wl, episode_len=200, seed=777)
    obs, info = env.reset(seed=777 + ep)
    m = info["metrics"]
    if ctrl is not None:
        ctrl.reset()
    rows, done = [], False
    while not done:
        a = int(model.predict(obs, deterministic=True)[0]) if model is not None \
            else ACTIONS.index(ctrl.act(m, env._backend.s))
        obs, _, term, trunc, info = env.step(a)
        m, s = info["metrics"], env._backend.s
        ok = m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99
        rows.append((m["load_frac"] * CPU_DEMAND_M * INTENSITY[wl]["cpu"] / 1000.0,
                     s.replicas * s.cpu_req / 1000.0, ok))
        done = term or trunc
    return np.array(rows, dtype=float)


ap = argparse.ArgumentParser()
ap.add_argument("--workload", default="mixed")
ap.add_argument("--episode", type=int, default=0)
args = ap.parse_args()

tandem = train.load_model("cvar-bellman-qrdqn", "models/cvar-bellman-qrdqn_all_azure",
                          train.make_env("all", 0))
runs = [("HPA+VPA", "#eb6834", rollout(args.workload, args.episode, ctrl=eb.ClassicalController(vpa=True))),
        ("TANDEM", "#2a78d6", rollout(args.workload, args.episode, model=tandem))]

INK, MUTED, GRID, MISS, DEMAND = "#222222", "#6b6b6b", "#e6e6e6", "#c62828", "#c9c9c9"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True, sharey=True)
ymax = max(max(X[:, 0].max(), X[:, 1].max()) for _, _, X in runs) * 1.3
for ax, (name, color, X) in zip(axes, runs):
    t = np.arange(len(X)) * 15 / 60.0
    ax.fill_between(t, X[:, 0], step="post", color=DEMAND, lw=0, label="CPU demanded by the load")
    ax.plot(t, X[:, 1], drawstyle="steps-post", color=color, lw=1.6, label="CPU allocated")
    miss = X[:, 2] < 0.5
    ax.plot(t[miss], np.full(miss.sum(), -0.06 * ymax), "|", color=MISS, ms=6, mew=1.1,
            label="SLO missed", clip_on=False)
    ax.set_ylim(0, ymax)
    ax.text(0.99, 0.99, f"{name}: SLO compliance {100 * (1 - miss.mean()):.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_ylabel("CPU (vCPU)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)
axes[-1].set_xlabel("time (min)")
axes[-1].set_xlim(0, 50)
h = [plt.Rectangle((0, 0), 1, 1, color=DEMAND),
     plt.Line2D([], [], color="#eb6834", lw=1.6), plt.Line2D([], [], color="#2a78d6", lw=1.6),
     plt.Line2D([], [], color=MISS, marker="|", ls="none", ms=6, mew=1.1)]
fig.legend(h, ["CPU demanded by the load", "allocated, HPA+VPA", "allocated, TANDEM", "SLO missed"],
           loc="upper center", ncol=2, frameon=False, fontsize=6.6, handlelength=1.8,
           columnspacing=1.2, bbox_to_anchor=(0.55, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=0.9)
out = os.path.join(HERE, f"episode_{args.workload}.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print(out)
