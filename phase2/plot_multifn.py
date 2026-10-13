"""Contention-onset figure for the multi-function ablation (#22).

Two panels, never a second y-axis: the same measure under two capacity anchorings.
Read together they separate a controller's resource *appetite* from its ability to
*handle* contention, which a single anchoring cannot do.

  shared  one cluster sized from conventional HPA+VPA demand -- what happens when a
          new controller is dropped into a cluster that already exists.
  self    capacity scaled to each controller's own p95 demand -- every controller
          feels the same relative pressure.

    python plot_multifn.py                       # N=4, legacy profiles
    python plot_multifn.py --profiles azure --n 4
    python plot_multifn.py --family bitbrains     # from multifn_summary_bitbrains.csv
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito: a published colourblind-safe qualitative palette. Assigned in fixed
# order and never cycled, so a controller keeps its colour across every figure.
# hpa-vpa and predictive-vpa land on nearly identical values, so colour alone would
# hide one under the other; the dash patterns keep every series readable where they
# coincide, and double as the print / colour-blind fallback.
SERIES = [
    ("hpa-vpa", "#0072B2", "o", (0, (5, 2))),
    ("predictive-vpa", "#E69F00", "s", "solid"),
    ("qrdqn", "#009E73", "^", "solid"),
    ("cvar-bellman-qrdqn", "#CC79A7", "D", (0, (1, 1.4))),
    ("ppo", "#D55E00", "v", (0, (6, 1.5, 1, 1.5))),
]
PHI_ORDER = ["inf", "1.5", "1.0", "0.75", "0.5"]
ANCHORS = [("shared", "Shared cluster\n(sized for conventional HPA+VPA)"),
           ("self", "Per-controller cluster\n(scaled to each one's own demand)")]


def _x(df: pd.DataFrame) -> pd.Series:
    return df["phi"].astype(str).map({p: i for i, p in enumerate(PHI_ORDER)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="legacy", choices=["legacy", "azure"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--family", choices=["azure", "bitbrains"], default="azure")
    args = ap.parse_args()
    sfx = "_bitbrains" if args.family == "bitbrains" else ""

    src = os.path.join(HERE, "eval", f"multifn_summary{sfx}.csv")
    d = pd.read_csv(src)
    d = d[(d.profiles == args.profiles) & (d.n_fns == args.n)]
    if d.empty:
        raise SystemExit(f"no rows for profiles={args.profiles} n={args.n} in {src}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.18,
                                              wspace=0.16))
    for col, (anchor, title) in enumerate(ANCHORS):
        sub = d[d.anchor == anchor]
        for metric, row, label, lo in (("sla_pct", 0, "Mean SLA attainment (%)", None),
                                       ("jain", 1, "Fairness (Jain index)", None)):
            ax = axes[row][col]
            for name, colour, marker, dash in SERIES:
                s = sub[sub.algo == name].copy()
                if s.empty:
                    continue
                s = s.assign(_x=_x(s)).sort_values("_x")
                y = s[metric] if metric != "jain" else s[metric]
                ax.plot(s["_x"], y, color=colour, marker=marker, markersize=5.5,
                        linewidth=2.0, linestyle=dash, label=name, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.8)
            ax.grid(True, axis="y", color="0.88", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("0.6")
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(title, fontsize=10.5, pad=9)
                ax.set_ylim(0, 103)
            else:
                ax.set_ylim(0.4, 1.04)
            ax.set_xticks(range(len(PHI_ORDER)))
            ax.set_xticklabels(PHI_ORDER)
        axes[1][col].set_xlabel(r"capacity ratio $\varphi$  (tighter $\rightarrow$)",
                                fontsize=10)

    # share the y-range across a row so the two anchorings are directly comparable
    for row in range(2):
        lo = min(a.get_ylim()[0] for a in axes[row])
        hi = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(lo, hi)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Phase 1 controllers at N={args.n} co-located functions "
                 f"({args.profiles} service profiles"
                 f"{', Bitbrains traces' if sfx else ''})", fontsize=12, y=0.975)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = os.path.join(HERE, "eval", f"contention_onset_{args.profiles}_n{args.n}{sfx}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
