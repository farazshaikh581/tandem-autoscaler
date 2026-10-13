"""Shared style and data loading for the evaluation figures."""
import collections
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = os.path.normpath(os.path.join(HERE, "..", "..", "phase1"))
EVAL = os.path.join(PHASE1, "eval")
CLASSES = ["cpu-bound", "memory-bound", "io-bound", "mixed"]
CLASS_TITLES = ["CPU-bound", "memory-bound", "IO-bound", "mixed"]
FAMILIES = [("azure", "Azure"), ("bitbrains", "Bitbrains")]

INK, MUTED, GRID, BAND = "#222222", "#6b6b6b", "#e6e6e6", "#e8f0fb"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc2"
# trace family is encoded by marker fill, never by hue, so hue stays free for controllers
FAM_STYLE = {"azure": dict(marker="o", mfc=INK, mec=INK, label="Azure"),
             "bitbrains": dict(marker="o", mfc="white", mec=INK, label="Bitbrains")}

NAMES = {"cvar-bellman-qrdqn": "TANDEM", "qrdqn": "QR-DQN", "cvar-qrdqn": "CVaR-QR-DQN",
         "dqn": "DQN", "ppo": "PPO", "trpo": "TRPO", "sac-discrete": "SAC-Discrete",
         "hpa-only": "HPA", "hpa-vpa": "HPA+VPA", "predictive": "Predictive",
         "predictive-vpa": "Predictive+VPA", "mpc": "MPC", "coscal": "CoScal",
         "slo-cost-aware": "SLO-driven", "slo-cost-aware-vpa": "SLO-driven+VPA",
         "pandey-ppo": "Pandey (PPO)", "pandey-ddpg": "Pandey (DDPG)"}


def setup():
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def clean(ax, grid="x"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def save(fig, name):
    out = os.path.join(HERE, f"{name}.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=300)
    print(out)


def per_seed(family):
    """{algo: {seed: {workload: row}}} for every controller in the main comparison.
    Multi-seed rows come from the nominal re-evaluation (identical to the committed
    eval CSVs); single-seed learners and baselines from comparison_<family>.csv."""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(os.path.join(PHASE1, "calibration", "knee_sensitivity", f"{family}_nominal.csv")) as f:
        for r in csv.DictReader(f):
            d[r["algo"]][int(r["seed"])][r["workload"]] = r
    seeded = set(d)
    with open(os.path.join(EVAL, f"comparison_{family}.csv")) as f:
        for r in csv.DictReader(f):
            if r["algo"] not in seeded:
                d[r["algo"]][0][r["workload"]] = r
    return d


def stat(rows_by_seed, wl, key):
    v = np.array([float(s[wl][key]) for s in rows_by_seed.values()])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def reversals_per_episode(rows_by_seed):
    tot = [sum(float(s[w][k]) for w in CLASSES
               for k in ("replica_reversals", "cpu_reversals", "mem_reversals")) / 80.0
           for s in rows_by_seed.values()]
    return float(np.mean(tot))


def churn_per_episode(rows_by_seed):
    tot = [sum(float(s[w]["churn"]) for w in CLASSES) / 80.0 for s in rows_by_seed.values()]
    return float(np.mean(tot))


def mean_cpu(rows_by_seed):
    return float(np.mean([np.mean([float(s[w]["mean_cost_vcpu"]) for w in CLASSES])
                          for s in rows_by_seed.values()]))


XMIN = 85.0


def dotrow(ax, y, x, err, fam, xmin=None, value_label=None):
    st = FAM_STYLE[fam]
    off = -0.17 if fam == "azure" else 0.17
    clipped = xmin is not None and x < xmin
    xs = xmin if clipped else x
    if err and not clipped:
        ax.errorbar(xs, y + off, xerr=err, fmt="none", ecolor=MUTED, elinewidth=0.8,
                    capsize=1.5, zorder=2)
    ax.plot(xs, y + off, marker="<" if clipped else st["marker"], ms=4.2, mfc=st["mfc"],
            mec=st["mec"], mew=0.9, ls="none", zorder=3, clip_on=False)
    if clipped:
        ax.text(xs + 0.4, y + off, f"{x:.0f}", va="center", ha="left", fontsize=6, color=INK)
