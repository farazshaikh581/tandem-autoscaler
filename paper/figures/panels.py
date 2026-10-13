"""Panel drawing functions shared by the standalone figures and the full-page overviews.
Each function draws into the axes it is given and returns (handles, labels) for a legend."""
import csv
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import figstyle as fs

MAIN_CTRL = [("cvar-bellman-qrdqn", fs.BLUE), ("qrdqn", fs.AQUA), ("ppo", fs.PURPLE),
             ("hpa-vpa", fs.ORANGE), ("hpa-only", "#f6b48f")]
_DATA = {}


def data(fam):
    if fam not in _DATA:
        _DATA[fam] = fs.per_seed(fam)
    return _DATA[fam]


def title(ax, text):
    ax.set_title(text, fontsize=7.5, color=fs.INK, pad=3)


# --- simulation -------------------------------------------------------------------------
def slo_bars(axes):
    x = np.arange(4); w = 0.16
    for ax, (fam, name) in zip(axes, fs.FAMILIES):
        for k, (algo, color) in enumerate(MAIN_CTRL):
            st = [fs.stat(data(fam)[algo], wl, "sla_pct") for wl in fs.CLASSES]
            m = np.array([s[0] for s in st]); e = np.array([s[1] for s in st])
            xs = x + (k - 2) * w
            ax.bar(xs, m, w * 0.9, color=color, label=fs.NAMES[algo], zorder=2)
            if e.any():
                ax.errorbar(xs, m, yerr=e, fmt="none", ecolor=fs.INK, elinewidth=0.7, capsize=1.4, zorder=3)
            for xi, mi in zip(xs, m):
                if mi < 1:
                    ax.text(xi, 1.5, "0", ha="center", va="bottom", fontsize=6, color=fs.INK)
        ax.set_xticks(x); ax.set_xticklabels(fs.CLASS_TITLES); ax.set_ylim(0, 100)
        title(ax, name); fs.clean(ax, grid="y"); ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("SLO compliance (%)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    return axes[0].get_legend_handles_labels()


def cost_rev(ax_cpu, ax_rev):
    y = np.arange(len(MAIN_CTRL)); h = 0.36
    for ax, metric, label in [(ax_cpu, "cpu", "allocated CPU (vCPU)"), (ax_rev, "rev", "reversals per episode")]:
        for j, (fam, _) in enumerate(fs.FAMILIES):
            vals = [fs.mean_cpu(data(fam)[a]) if metric == "cpu" else fs.reversals_per_episode(data(fam)[a])
                    for a, _ in MAIN_CTRL]
            colors = [c for _, c in MAIN_CTRL]
            ax.barh(y + (j - 0.5) * h, vals, h * 0.9, color=colors, alpha=1.0 if j == 0 else 0.45,
                    edgecolor=colors, lw=0.6, zorder=2)
        ax.set_yticks(y); ax.set_yticklabels([fs.NAMES[a] for a, _ in MAIN_CTRL] if ax is ax_cpu else [])
        ax.set_ylim(len(MAIN_CTRL) - 0.4, -0.6); ax.set_xlabel(label)
        fs.clean(ax, grid="x"); ax.tick_params(axis="y", length=0)
    hl = [plt.Rectangle((0, 0), 1, 1, color=fs.MUTED), plt.Rectangle((0, 0), 1, 1, color=fs.MUTED, alpha=0.45)]
    return hl, ["Azure", "Bitbrains"]


def ablation(axes):
    variants = [("", "joint", fs.BLUE), ("-h-only", "H-only (replicas)", fs.ORANGE),
                ("-v-only", "V-only (resize)", fs.AQUA)]
    x = np.arange(4); w = 0.27
    for ax, (fam, name) in zip(axes, fs.FAMILIES):
        for k, (v, label, color) in enumerate(variants):
            sfx = ("_azure" if fam == "azure" else "_bb") if v else ("_azure" if fam == "azure" else "")
            with open(os.path.join(fs.EVAL, f"eval_cvar-bellman-qrdqn{v}_all{sfx}.csv")) as f:
                vals = {r["workload"]: float(r["sla_pct"]) for r in csv.DictReader(f)}
            xs = x + (k - 1) * w; yv = [vals[wl] for wl in fs.CLASSES]
            ax.bar(xs, yv, w * 0.9, color=color, label=label, zorder=2)
            for xi, yi in zip(xs, yv):
                if yi < 1:
                    ax.text(xi, 1.5, "0", ha="center", va="bottom", fontsize=6, color=fs.INK)
        ax.set_xticks(x); ax.set_xticklabels(["CPU", "mem.", "IO", "mixed"]); ax.set_ylim(0, 100)
        title(ax, name); fs.clean(ax, grid="y"); ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("SLO compliance (%)")
    axes[1].tick_params(labelleft=False)
    return axes[0].get_legend_handles_labels()


def dotplot(axes, rows, divider, last="churn"):
    ypos = {a: i for i, a in enumerate(rows) if a}
    for ax in axes:
        ax.axhspan(-0.5, 0.5, color=fs.BAND, lw=0, zorder=0)
        ax.axhline(divider, color=fs.MUTED, lw=0.5)
    for c, wl in enumerate(fs.CLASSES):
        ax = axes[c]
        for a, y in ypos.items():
            for fam, _ in fs.FAMILIES:
                m, s = fs.stat(data(fam)[a], wl, "sla_pct")
                fs.dotrow(ax, y, m, s, fam, xmin=fs.XMIN)
        ax.set_xlim(fs.XMIN - 0.6, 100.4); ax.xaxis.set_major_locator(FixedLocator([90, 95, 100]))
        title(ax, fs.CLASS_TITLES[c]); fs.clean(ax)
    axes[1].set_xlabel("SLO compliance (%)", x=1.0)
    ax = axes[4]
    for a, y in ypos.items():
        for fam, _ in fs.FAMILIES:
            fs.dotrow(ax, y, max(fs.churn_per_episode(data(fam)[a]), 0.5), 0, fam)
    ax.set_xscale("log"); ax.set_xlim(0.4, 400); ax.minorticks_off()
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    title(ax, "churn"); ax.set_xlabel("changes per episode"); fs.clean(ax)
    for ax in axes:
        ax.set_yticks(list(ypos.values())); ax.set_ylim(len(rows) - 0.5, -0.5)
    axes[0].set_yticklabels([fs.NAMES[a] for a in ypos])
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=4.2, mfc=fs.INK, mec=fs.INK),
         plt.Line2D([], [], marker="o", ls="none", ms=4.2, mfc="white", mec=fs.INK),
         plt.Line2D([], [], marker="<", ls="none", ms=4.2, mfc=fs.INK, mec=fs.INK)]
    return h, ["Azure", "Bitbrains", "below 85% (value shown)"]


def pareto(axes2x4):
    groups = [("TANDEM", fs.BLUE, "-", "o", 3.4), ("MPC", fs.PURPLE, "-", "D", 2.8),
              ("threshold autoscalers", fs.ORANGE, "-", "s", 3.0),
              ("fixed configuration", fs.INK, (0, (3, 1.6)), None, 0)]
    ymin = 85.0

    def envelope(points):
        out, best = [], -np.inf
        for c, s in sorted(set(points), key=lambda p: (p[0], -p[1])):
            if s > best + 1e-9:
                out.append((c, s)); best = s
        return out

    for r, (fam, name) in enumerate(fs.FAMILIES):
        sfx = "" if fam == "azure" else "_bitbrains"
        agg = pd.read_csv(os.path.join(fs.EVAL, f"pareto_frontier_agg{sfx}.csv"))
        static = pd.read_csv(os.path.join(fs.EVAL, f"static_frontier_{fam}.csv"))
        for c, wl in enumerate(fs.CLASSES):
            ax = axes2x4[r][c]
            a = agg[agg.workload == wl]
            pick = lambda m: list(zip(a[m].cost_mean, a[m].sla_mean))
            st = static[static.workload == wl]
            g = {"TANDEM": pick(a.controller == "cvar-bellman-qrdqn"), "MPC": pick(a.controller == "mpc"),
                 "threshold autoscalers": pick(a.knob == "hpa_target"),
                 "fixed configuration": list(zip(st.mean_cost_vcpu, st.sla_pct))}
            tan = envelope(g["TANDEM"])
            reach = [x for x, s in envelope(g["fixed configuration"]) if s >= tan[-1][1] - 1e-9]
            xmax = min(max(tan[-1][0], reach[0] if reach else 0) * 1.12, 7.0)
            for label, color, ls, mk, ms in groups:
                env = [(x, s) for x, s in envelope(g[label]) if x <= xmax]
                if not env:
                    continue
                xs, ss = map(np.array, zip(*env)); shown = np.maximum(ss, ymin)
                ax.plot(xs, shown, color=color, ls=ls, lw=1.3, zorder=3, label=label, clip_on=False)
                if mk:
                    below = ss < ymin
                    ax.plot(xs[~below], shown[~below], mk, ms=ms, color=color, mec="white", mew=0.5,
                            zorder=4, clip_on=False)
                    ax.plot(xs[below], shown[below], "v", ms=ms + 0.6, color=color, mec="white", mew=0.5,
                            zorder=4, clip_on=False)
            ax.set_ylim(ymin, 100.3); ax.set_xlim(0, xmax)
            ax.axhline(99, color=fs.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
            fs.clean(ax, grid="y")
            if r == 0:
                title(ax, fs.CLASS_TITLES[c])
            if c == 0:
                ax.set_ylabel(f"{name}\nSLO compliance (%)")
            else:
                ax.tick_params(labelleft=False)
            if r == 1:
                ax.set_xlabel("allocated CPU (vCPU)")
    h, l = axes2x4[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=fs.MUTED, marker="v", ls="none", ms=4)); l.append("below 85%")
    return h, l


# --- beyond the training setting -----------------------------------------------------------
def heldout(axes):
    learners = [("cvar-bellman-qrdqn", "TANDEM"), ("qrdqn", "QR-DQN")]
    tag = {"azure": "_azure", "bitbrains": ""}
    bars = [("training trace", fs.BLUE), ("held-out machines", "#9ec5f4"), ("held-out, trained on pool", fs.AQUA)]
    ymin = 55
    rng = np.random.default_rng(0)

    def rows(p):
        with open(p) as f:
            return list(csv.DictReader(f))

    def per_class(paths):
        d = {wl: [] for wl in fs.CLASSES}
        for p in paths:
            for r in rows(p):
                d[r["workload"]].append(float(r["sla_pct"]))
        return d

    x = np.arange(4); w = 0.26
    for i, (fam, fname) in enumerate(fs.FAMILIES):
        for j, (algo, name) in enumerate(learners):
            ax = axes[2 * i + j]
            base = os.path.join(fs.EVAL, f"heldout_{algo}_all{tag[fam]}")
            indist = {r["workload"]: float(r["sla_pct_indist"]) for r in rows(base + "_summary.csv")}
            series = [{wl: [indist[wl]] for wl in fs.CLASSES}, per_class([base + ".csv"]),
                      per_class(sorted(p for p in glob.glob(base + "_dr_seed*.csv") if not p.endswith("_summary.csv")))]
            for k, ((label, color), vals) in enumerate(zip(bars, series)):
                xs = x + (k - 1) * w
                means = [np.mean(vals[wl]) for wl in fs.CLASSES]
                ax.bar(xs, [m - ymin for m in means], w * 0.92, bottom=ymin, color=color, lw=0, label=label, zorder=2)
                if k > 0:
                    for xi, wl in zip(xs, fs.CLASSES):
                        v = np.array(vals[wl])
                        ax.plot(xi + rng.uniform(-0.06, 0.06, len(v)), np.maximum(v, ymin), "o", ms=2.0,
                                mfc=fs.INK, mec="none", alpha=0.7, zorder=3, clip_on=False)
            ax.set_xticks(x); ax.set_xticklabels(["CPU", "mem.", "IO", "mixed"]); ax.set_ylim(ymin, 100.5)
            title(ax, f"{name}, {fname}"); fs.clean(ax, grid="y"); ax.tick_params(axis="x", length=0)
            if ax is not axes[0]:
                ax.tick_params(labelleft=False)
    axes[0].set_ylabel("SLO compliance (%)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", ls="none", ms=2.6, mfc=fs.INK, mec="none")); l.append("one machine (and seed)")
    return h, l


def simreal(axes):
    exps = [("cpu-bound_2node", "CPU app, CPU-bound,\n2 workers"), ("cpu-bound_4node", "CPU app, CPU-bound,\n4 workers"),
            ("io-bound_io-burn", "IO app, IO-bound"), ("mixed_io-burn", "IO app, mixed")]
    learners = [("cvar-bellman-qrdqn", fs.BLUE), ("qrdqn", fs.AQUA)]
    tag = {"azure": "azure", "bitbrains": "bb"}
    centers = [e * 2.4 for e in range(len(exps))]
    for ax, (fam, name) in zip(axes, fs.FAMILIES):
        for e, (exp, _) in enumerate(exps):
            if e % 2 == 0:
                ax.axhspan(centers[e] - 0.95, centers[e] + 0.95, color="#f4f4f4", lw=0, zorder=0)
            for k, (algo, color) in enumerate(learners):
                with open(os.path.join(fs.EVAL, f"sim_vs_real_{algo}_{exp}_{tag[fam]}.csv")) as f:
                    r = list(csv.DictReader(f))
                s = 100 * np.mean([float(q["sim_sla"]) for q in r]); rr = 100 * np.mean([float(q["real_sla"]) for q in r])
                y = centers[e] + (k - 0.5) * 0.8
                if abs(rr - s) > 1:
                    ax.annotate("", xy=(rr, y), xytext=(s, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                shrinkA=3.5, shrinkB=3.5, mutation_scale=7))
                ax.plot(s, y, "o", ms=4.2, mfc="white", mec=color, mew=1.2, zorder=3)
                ax.plot(rr, y, "o", ms=3.0 if abs(rr - s) <= 1 else 4.2, mfc=color, mec=color, zorder=4)
        ax.set_xlim(10, 105); ax.set_xticks([20, 60, 100]); title(ax, name)
        ax.set_xlabel("SLO compliance (%)"); fs.clean(ax); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(centers); axes[0].set_yticklabels([l for _, l in exps], fontsize=6.6)
    axes[0].set_ylim(centers[-1] + 1.0, -1.0); axes[1].set_ylim(centers[-1] + 1.0, -1.0)
    axes[1].set_yticks(centers); axes[1].tick_params(labelleft=False)
    h = [plt.Line2D([], [], marker="o", ls="none", ms=4.2, mfc="white", mec=fs.INK, mew=1.1),
         plt.Line2D([], [], marker="o", ls="none", ms=4.2, mfc=fs.INK, mec=fs.INK),
         plt.Line2D([], [], color=fs.BLUE, lw=1.6), plt.Line2D([], [], color=fs.AQUA, lw=1.6)]
    return h, ["simulator", "real cluster", "TANDEM", "QR-DQN"]


def keda(axes):
    ctrl = [("keda", "KEDA", "#c4c4c4"), ("keda-vpa", "KEDA+VPA", fs.MUTED),
            ("QR-DQN", "QR-DQN", fs.AQUA), ("Option B", "TANDEM", fs.BLUE)]
    w = 0.2
    for c, (wl, name) in enumerate([("cpu-bound", "CPU-bound"), ("memory-bound", "memory-bound")]):
        ax = axes[c]
        for fi, (fam, fname) in enumerate(fs.FAMILIES):
            sfx = "" if fam == "azure" else "_bitbrains"
            with open(os.path.join(fs.EVAL, f"keda_vs_rl_summary{sfx}.csv")) as f:
                rows = list(csv.DictReader(f))
            for k, (key, lab, color) in enumerate(ctrl):
                v = float(next(r["sla_pct"] for r in rows if r["workload"] == wl and r["controller"] == key))
                xi = fi + (k - 1.5) * w
                ax.bar(xi, max(v, 0.6), w * 0.9, color=color, lw=0, zorder=2, label=lab if (c == 0 and fi == 0) else None)
                if v < 1:
                    ax.text(xi, 2, "0", ha="center", va="bottom", fontsize=6, color=fs.INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels([n for _, n in fs.FAMILIES]); ax.set_ylim(0, 100)
        title(ax, name); fs.clean(ax, grid="y"); ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("SLO compliance (%)"); axes[1].tick_params(labelleft=False)
    return axes[0].get_legend_handles_labels()


def multifn(axes):
    phase2 = os.path.join(fs.PHASE1, "..", "phase2", "eval")
    series = [("cvar-bellman-qrdqn", "TANDEM", fs.BLUE, "-", "o"), ("qrdqn", "QR-DQN", fs.AQUA, (0, (4, 1.5)), "^"),
              ("ppo", "PPO", fs.PURPLE, (0, (1.2, 1.2)), "D"), ("hpa-vpa", "HPA+VPA", fs.ORANGE, (0, (3.5, 1.8)), "s")]
    phis = ["inf", "1.5", "1.0", "0.75", "0.5"]
    panels = [("self", "azure"), ("self", "bitbrains"), ("shared", "azure"), ("shared", "bitbrains")]
    names = {"self": "own-demand capacity", "shared": "one capacity for all"}
    for ax, (anchor, fam) in zip(axes, panels):
        sfx = "" if fam == "azure" else "_bitbrains"
        with open(os.path.join(phase2, f"multifn_summary{sfx}.csv")) as f:
            rows = [r for r in csv.DictReader(f) if r["profiles"] == "legacy" and r["n_fns"] == "4"]
        for algo, name, color, ls, mk in series:
            yv = [float(next(r["sla_pct"] for r in rows if r["anchor"] == anchor and r["algo"] == algo and r["phi"] == p))
                  for p in phis]
            ax.plot(range(5), yv, color=color, ls=ls, lw=1.3, marker=mk, ms=3.4, mec="white", mew=0.5, label=name)
        ax.set_ylim(30, 100); ax.set_xticks(range(5)); ax.set_xticklabels(["$\\infty$", "1.5", "1", "0.75", "0.5"])
        title(ax, f"{names[anchor]}, {dict(fs.FAMILIES)[fam]}"); fs.clean(ax, grid="y")
        ax.set_xlabel(r"capacity ratio $\phi$")
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("SLO compliance (%)")
    return axes[0].get_legend_handles_labels()
