"""Figure: does domain randomization fix the held-out stability blowup?

Companion to plot_heldout.py. The held-out finding was that stability trained on a
single representative VM/class does NOT transfer to unseen same-class VMs. Here we
retrain over a 34-VM pool (traces_train/, `train.py --domain-rand`, seeds 0/1/2) and
re-run the SAME held-out protocol, then compare:

  single-rep held-out   (solid)   -- from heldout_<algo>_all_summary.csv (*_heldout_mean)
  domain-rand held-out  (hatched)  -- mean over 3 seeds of heldout_<algo>_all_dr_seed*.csv

Two panels sharing the class axis:
  top    SLA%            -- holds (and Option B's memory-bound collapse recovers)
  bottom total reversals -- Option B + DR drops it on every class; QR-DQN + DR is mixed

  python plot_domain_rand.py
  python plot_domain_rand.py --tag azure   # -> eval/domain_rand_generalization_azure.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import CLASSES

HERE = os.path.dirname(__file__)
QR = "#4C78A8"    # QR-DQN  (blue)
OB = "#F58518"    # Option B (orange)
REV = ["replica_reversals", "cpu_reversals", "mem_reversals"]


def baseline(algo: str, suffix: str) -> pd.DataFrame:
    """Single-rep held-out means, per class (from eval_heldout.py summary)."""
    df = pd.read_csv(os.path.join(HERE, "eval", f"heldout_{algo}_all{suffix}_summary.csv"))
    df = df.set_index("workload")
    out = pd.DataFrame(index=CLASSES)
    out["sla_pct"] = [df.loc[c, "sla_pct_heldout_mean"] for c in CLASSES]
    out["total_rev"] = [sum(df.loc[c, f"{k}_heldout_mean"] for k in REV) for c in CLASSES]
    return out


def domain_rand(algo: str, dr_prefix: str) -> pd.DataFrame:
    """Domain-rand held-out means, per class (mean over 3 seeds x unseen VMs)."""
    frames = [pd.read_csv(f) for f in
              sorted(glob.glob(os.path.join(HERE, "eval", f"heldout_{algo}_all_{dr_prefix}*.csv")))
              if not f.endswith("_summary.csv")]
    d = pd.concat(frames, ignore_index=True)
    d["total_rev"] = d[REV].sum(axis=1)
    g = d.groupby("workload")[["sla_pct", "total_rev"]].mean()
    return g.reindex(CLASSES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="e.g. 'azure' -> heldout_<algo>_all_azure[_dr_seed*] files")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    dr_prefix = f"{args.tag}_dr_seed" if args.tag else "dr_seed"

    data = {"QR-DQN": (QR, baseline("qrdqn", suffix), domain_rand("qrdqn", dr_prefix)),
            "Option B": (OB, baseline("cvar-bellman-qrdqn", suffix), domain_rand("cvar-bellman-qrdqn", dr_prefix))}
    x = np.arange(len(CLASSES))
    w = 0.2

    fig, (ax_sla, ax_rev) = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 3]})

    order = [("QR-DQN", "single-rep"), ("QR-DQN", "domain-rand"),
             ("Option B", "single-rep"), ("Option B", "domain-rand")]
    for i, (algo, kind) in enumerate(order):
        color, base, dr = data[algo]
        src = base if kind == "single-rep" else dr
        hatch = "//" if kind == "domain-rand" else None
        ax_sla.bar(x + (i - 1.5) * w, src["sla_pct"].values, w, color=color, hatch=hatch,
                   edgecolor="white", label=f"{algo}, {kind}")
        ax_rev.bar(x + (i - 1.5) * w, np.maximum(src["total_rev"].values, 0.5), w,
                   color=color, hatch=hatch, edgecolor="white")

    ax_sla.axhline(99, color="#9aa0a6", ls=":", lw=1.2)
    ax_sla.text(len(CLASSES) - 0.5, 99.05, "99%", color="#6b7075", fontsize=8,
                va="bottom", ha="right")
    ax_sla.set_ylim(80, 100.5)
    ax_sla.set_ylabel("SLA satisfied (%)")
    ax_sla.set_title("Domain randomization on the held-out VMs: single-rep vs. domain-rand training",
                     fontsize=11, loc="left")
    ax_sla.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")

    ax_rev.set_yscale("log")
    ax_rev.set_ylabel("held-out total reversals (rep+cpu+mem)")
    ax_rev.set_xticks(x)
    ax_rev.set_xticklabels(CLASSES)
    ax_rev.set_xlabel("workload class")

    for ax in (ax_sla, ax_rev):
        ax.grid(True, axis="y", color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    out_name = f"domain_rand_generalization{suffix}.png"
    out = os.path.join(HERE, "eval", out_name)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
