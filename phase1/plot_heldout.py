"""Figure: does in-distribution performance survive on held-out (unseen) VMs?

Two panels sharing the class axis. For QR-DQN and Option B (CVaR-Bellman), each
class shows the in-distribution number (solid) next to the held-out mean over
unseen same-class VMs (hatched):
  top    SLA%           -- generalizes: bars stay near the target line
  bottom total reversals -- does NOT reliably generalize: held-out bars blow up

Reads eval/heldout_<algo>_all[_<tag>]_summary.csv (written by eval_heldout.py).

  python plot_heldout.py
  python plot_heldout.py --tag azure   # -> eval/heldout_generalization_azure.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.joint_hv_env import SLA_MEAN_LATENCY_MS
from train import CLASSES

HERE = os.path.dirname(__file__)
QR = "#4C78A8"    # QR-DQN  (blue)
OB = "#F58518"    # Option B (orange)
REV = ["replica_reversals", "cpu_reversals", "mem_reversals"]


def load(algo: str, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(HERE, "eval", f"heldout_{algo}_all{suffix}_summary.csv"))
    return df.set_index("workload")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="e.g. 'azure' -> heldout_<algo>_all_azure_summary.csv")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    qr, ob = load("qrdqn", suffix), load("cvar-bellman-qrdqn", suffix)
    x = np.arange(len(CLASSES))
    w = 0.2

    def series(df, suffix, cols):
        return np.array([sum(df.loc[c][f"{k}_{suffix}"] for k in cols)
                         if c in df.index else np.nan for c in CLASSES])

    fig, (ax_sla, ax_rev) = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 3]})

    # --- SLA% (generalizes) ---
    sla = {("QR-DQN", "in-dist"): series(qr, "indist", ["sla_pct"]),
           ("QR-DQN", "held-out"): series(qr, "heldout_mean", ["sla_pct"]),
           ("Option B", "in-dist"): series(ob, "indist", ["sla_pct"]),
           ("Option B", "held-out"): series(ob, "heldout_mean", ["sla_pct"])}
    for i, ((algo, kind), vals) in enumerate(sla.items()):
        color = QR if algo == "QR-DQN" else OB
        hatch = "//" if kind == "held-out" else None
        ax_sla.bar(x + (i - 1.5) * w, vals, w, color=color, hatch=hatch,
                   edgecolor="white", label=f"{algo}, {kind}")
    ax_sla.axhline(99, color="#9aa0a6", ls=":", lw=1.2)
    ax_sla.text(len(CLASSES) - 0.5, 99.05, "99%", color="#6b7075", fontsize=8, va="bottom", ha="right")
    ax_sla.set_ylim(80, 100.5)
    ax_sla.set_ylabel("SLA satisfied (%)")
    ax_sla.set_title("Generalization to held-out VMs: SLA% vs. stability (in-dist vs. unseen VMs)",
                     fontsize=11, loc="left")
    ax_sla.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")

    # --- total reversals (does not generalize), log scale ---
    rev = {("QR-DQN", "in-dist"): series(qr, "indist", REV),
           ("QR-DQN", "held-out"): series(qr, "heldout_mean", REV),
           ("Option B", "in-dist"): series(ob, "indist", REV),
           ("Option B", "held-out"): series(ob, "heldout_mean", REV)}
    for i, ((algo, kind), vals) in enumerate(rev.items()):
        color = QR if algo == "QR-DQN" else OB
        hatch = "//" if kind == "held-out" else None
        ax_rev.bar(x + (i - 1.5) * w, np.maximum(vals, 0.5), w, color=color, hatch=hatch,
                   edgecolor="white", label=f"{algo}, {kind}")
    ax_rev.set_yscale("log")
    ax_rev.set_ylabel("total reversals (rep+cpu+mem)")
    ax_rev.set_xticks(x)
    ax_rev.set_xticklabels(CLASSES)
    ax_rev.set_xlabel("workload class")

    for ax in (ax_sla, ax_rev):
        ax.grid(True, axis="y", color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    out_name = f"heldout_generalization{suffix}.png"
    out = os.path.join(HERE, "eval", out_name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
