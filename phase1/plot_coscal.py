"""CoScal (#21) head-to-head figure: the reproduced literature baseline against
this project's RL headliners and the MPC challenger, read off
`eval/comparison_azure.csv`.

  eval/coscal_vs_rl_azure.png   SLA%, mean latency and mean cost by workload
                                class for coscal vs qrdqn / Option B / mpc,
                                with the 20 ms SLA line and the 99% SLA line
                                marked

  python plot_coscal.py

The story the panels carry: CoScal tracks the RL headliners on the two
CPU-shaped classes but trails by 3-11 points, collapses on memory-bound
(11.3%, the workload-level state cannot see memory pressure), and its low cost
is under-provisioning, not efficiency.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "eval")
CLASSES = ["cpu-bound", "memory-bound", "io-bound", "mixed"]
ALGOS = ["coscal", "qrdqn", "cvar-bellman-qrdqn", "mpc"]
LABEL = {"coscal": "CoScal (#21)", "qrdqn": "QR-DQN", "mpc": "MPC (#16)",
         "cvar-bellman-qrdqn": "Option B (CVaR-Bellman)"}
COLOR = {"coscal": "#8C8C8C", "qrdqn": "#4C78A8",
         "cvar-bellman-qrdqn": "#F58518", "mpc": "#E45756"}
SLA_MS = 20.0


def _bars(ax, g, metric, title, ylabel):
    x = np.arange(len(CLASSES))
    w = 0.2
    for i, a in enumerate(ALGOS):
        vals = [float(g[(g.algo == a) & (g.workload == c)][metric].iloc[0])
                for c in CLASSES]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=COLOR[a],
                      label=LABEL[a], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if metric == "churn"
                    else f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right", fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#e6e8eb", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> None:
    df = pd.read_csv(os.path.join(EVAL, "comparison_azure.csv"))
    g = df[df.algo.isin(ALGOS)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    _bars(axes[0], g, "sla_pct", "SLA attainment", "SLA (%)")
    axes[0].axhline(99, color="#c0392b", lw=1, ls=":", zorder=1)
    axes[0].text(0.01, 99, " 99%", color="#c0392b", fontsize=8, va="bottom",
                 transform=axes[0].get_yaxis_transform())
    axes[0].set_ylim(0, 108)

    _bars(axes[1], g, "mean_lat_ms", "Mean latency", "latency (ms)")
    axes[1].axhline(SLA_MS, color="#c0392b", lw=1, ls=":", zorder=1)
    axes[1].text(0.01, SLA_MS, " 20 ms SLA", color="#c0392b", fontsize=8,
                 va="bottom", transform=axes[1].get_yaxis_transform())

    _bars(axes[2], g, "mean_cost_vcpu", "Mean cost (lower is cheaper, "
          "but only a win at equal SLA)", "cost (vCPU)")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("CoScal vs this project's RL headliners and MPC  "
                 "(Azure organic traces, seed 0)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(EVAL, "coscal_vs_rl_azure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    print(f"\n{'':22s} " + " ".join(f"{c:>13s}" for c in CLASSES))
    for a in ALGOS:
        row = " ".join(f"{float(g[(g.algo == a) & (g.workload == c)].sla_pct.iloc[0]):13.1f}"
                       for c in CLASSES)
        print(f"{LABEL[a]:22s} {row}")


if __name__ == "__main__":
    main()
