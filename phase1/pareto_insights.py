"""Descriptive analytic figures over the cost/SLA Pareto sweep (#20), read off
`eval/pareto_frontier.csv` (written by pareto_sweep.py).

Complements pareto_sweep.py's raw per-controller frontier plots with the three
figures the README writeup leans on:

  eval/pareto_frontiers_nd.png    non-dominated cost/SLA envelope per class,
                                  RL vs MPC vs the best classical controller,
                                  with the 99% SLA line marked
  eval/pareto_memory_vertical.png memory-bound class only: best achievable SLA
                                  per controller, showing horizontal-only
                                  control (hpa-only, predictive) collapsing to 0
  eval/pareto_stability_wcost.png cost and churn/reversals vs w_cost for Option
                                  B: past the knee, cost keeps falling but the
                                  policy thrashes

  python pareto_insights.py            # write all three + print the numeric summary
  python pareto_insights.py --family bitbrains   # same, from pareto_frontier_bitbrains.csv,
                                                 # figures suffixed "_bitbrains"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "eval")
CLASSES = ["cpu-bound", "memory-bound", "io-bound", "mixed"]
RL = ["qrdqn", "cvar-bellman-qrdqn", "cvar-qrdqn", "ppo", "dqn"]
CLASSICAL = ["hpa-only", "hpa-vpa", "predictive", "predictive-vpa"]
COLOR = {
    "qrdqn": "#4C78A8", "cvar-bellman-qrdqn": "#F58518", "cvar-qrdqn": "#54A24B",
    "ppo": "#B279A2", "dqn": "#9D755D", "mpc": "#E45756",
}
LABEL = {"cvar-bellman-qrdqn": "Option B (CVaR-Bellman)", "cvar-qrdqn": "CVaR-QRDQN",
         "qrdqn": "QR-DQN", "ppo": "PPO", "dqn": "DQN", "mpc": "MPC"}


SFX = ""  # output suffix, set from --family in main()


def load() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(EVAL, f"pareto_frontier{SFX}.csv"))
    return df.groupby(
        ["controller", "family", "knob", "knob_value", "w_viol", "workload"],
        dropna=False).agg(
        sla=("sla_pct", "mean"), cost=("mean_cost_vcpu", "mean"),
        lat=("mean_lat_ms", "mean"), churn=("churn", "mean"),
        rev=("total_reversals", "mean"), n=("seed", "nunique")).reset_index()


def nd_frontier(sub: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated envelope: ascending cost, keep strictly rising SLA."""
    s = sub.sort_values("cost")
    best, keep = -np.inf, []
    for _, r in s.iterrows():
        if r.sla > best + 1e-9:
            keep.append(r)
            best = r.sla
    return pd.DataFrame(keep)


def min_cost_for(sub: pd.DataFrame, target: float) -> float:
    ok = sub[sub.sla >= target].sort_values("cost")
    return ok.cost.iloc[0] if len(ok) else np.nan


def fig_frontiers(g: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, wl in zip(axes.ravel(), CLASSES):
        for ctrl in RL + ["mpc"]:
            sub = g[(g.controller == ctrl) & (g.workload == wl) &
                    ((g.w_viol == 1.0) | (g.controller == "mpc"))]
            if sub.empty:
                continue
            f = nd_frontier(sub)
            ax.plot(f.cost, f.sla, marker="o", ms=5, lw=2, color=COLOR[ctrl],
                    label=LABEL.get(ctrl, ctrl), zorder=3)
        cl = g[(g.controller.isin(CLASSICAL)) & (g.workload == wl)]
        if not cl.empty:
            f = nd_frontier(cl)
            ax.plot(f.cost, f.sla, marker="s", ms=5, lw=2, ls="--", color="#555",
                    label="classical best (HPA/VPA/predict)", zorder=2)
        ax.axhline(99, color="#c0392b", lw=1, ls=":", zorder=1)
        ax.text(0.02, 0.06, "99% SLA", transform=ax.transAxes, color="#c0392b", fontsize=8)
        ax.set_title(wl, fontweight="bold")
        ax.set_xlabel("mean cost (vCPU)")
        ax.set_ylabel("SLA attainment (%)")
        ax.grid(True, color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Non-dominated cost / SLA frontier by workload class  (RL at w_viol = 1.0)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(EVAL, f"pareto_frontiers_nd{SFX}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_memory_vertical(g: pd.DataFrame) -> None:
    mb = g[g.workload == "memory-bound"].copy()
    order = ["hpa-only", "predictive", "hpa-vpa", "predictive-vpa", "mpc",
             "dqn", "ppo", "qrdqn", "cvar-qrdqn", "cvar-bellman-qrdqn"]
    best = []
    for ctrl in order:
        sub = mb[(mb.controller == ctrl) &
                 ((mb.w_viol == 1.0) | (mb.knob == "hpa_target") | (ctrl == "mpc"))]
        best.append(sub.sla.max() if len(sub) else np.nan)
    cols = ["#c0392b" if c in ("hpa-only", "predictive") else
            ("#e08e45" if c in ("hpa-vpa", "predictive-vpa") else COLOR.get(c, "#4C78A8"))
            for c in order]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(range(len(order)), best, color=cols)
    for i, v in enumerate(best):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ceiling = np.nanmax([b for c, b in zip(order, best) if c not in CLASSICAL])
    ax.axhline(ceiling, color="#333", ls=":", lw=1)
    ax.text(0.2, ceiling + 1.2, f"RL / MPC ceiling ~ {ceiling:.1f}%", fontsize=8, color="#333")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL.get(c, c).replace(" (CVaR-Bellman)", "") for c in order],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("best achievable SLA (%)")
    ax.set_title("Memory-bound class: horizontal-only control collapses (OOM), "
                 "vertical actuation is mandatory", fontweight="bold")
    ax.grid(True, axis="y", color="#e6e8eb", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(EVAL, f"pareto_memory_vertical{SFX}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_stability(g: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True)
    ctrl = "cvar-bellman-qrdqn"
    for ax, wl in zip(axes, CLASSES):
        sub = g[(g.controller == ctrl) & (g.workload == wl) &
                (g.w_viol == 1.0)].sort_values("knob_value")
        ax.plot(sub.knob_value, sub.cost, "o-", color="#4C78A8", label="cost (vCPU)")
        ax.set_xscale("log")
        ax.set_title(wl, fontweight="bold")
        ax.set_xlabel("w_cost (log)")
        ax.grid(True, color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.plot(sub.knob_value, sub.rev, "s--", color="#E45756", label="reversals")
        ax2.plot(sub.knob_value, sub.churn, "^:", color="#9D755D", label="churn")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("mean cost (vCPU)", color="#4C78A8")
        if ax is axes[-1]:
            ax2.set_ylabel("churn / reversals", color="#E45756")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center")
    fig.suptitle("Cost and churn / reversals against w_cost (Option B, w_viol = 1.0)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(EVAL, f"pareto_stability_wcost{SFX}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def summary(g: pd.DataFrame) -> None:
    print("\n== min cost (vCPU) to reach 99% SLA  (RL @ w_viol = 1.0) ==")
    print(f"{'controller':24s} " + " ".join(f"{c:>13s}" for c in CLASSES))
    for ctrl in ["qrdqn", "cvar-bellman-qrdqn", "cvar-qrdqn", "ppo", "dqn", "mpc"]:
        row = []
        for wl in CLASSES:
            sub = g[(g.controller == ctrl) & (g.workload == wl) &
                    ((g.w_viol == 1.0) | (g.controller == "mpc"))]
            v = min_cost_for(sub, 99.0)
            row.append("  unreachable" if np.isnan(v) else f"{v:13.2f}")
        print(f"{LABEL.get(ctrl, ctrl):24s} " + " ".join(row))
    print("\n== classical best SLA at any utilisation target ==")
    for ctrl in CLASSICAL:
        row = [f"{g[(g.controller == ctrl) & (g.workload == wl)].sla.max():13.1f}"
               for wl in CLASSES]
        print(f"{ctrl:24s} " + " ".join(row))


def main() -> None:
    global SFX
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["azure", "bitbrains"], default="azure")
    SFX = "" if ap.parse_args().family == "azure" else "_bitbrains"
    g = load()
    fig_frontiers(g)
    fig_memory_vertical(g)
    fig_stability(g)
    summary(g)


if __name__ == "__main__":
    main()
