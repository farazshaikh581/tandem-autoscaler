"""Calibration of the CPU axis (paper Fig. knee): python3 paper/figures/plot_knee.py

(a) full range: the fit sweep (phase1/calibration/sweep_v2.csv) and the fitted knee model.
(b) transition: the dense sweep (sweep_dense_A_r{1,2,3}.csv, not used for the fit), mean of
    three runs with the min-max range, against the fitted model and its refit on all levels
    (constants from phase1/calibration/knee_sensitivity.py, "refit").
Writes knee_fit.pdf (vector) and knee_fit.png (300 dpi) next to this script.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "..", "phase1", "calibration")
M_PER_RPS, LIMIT = 3.27, 400.0
FIT = dict(l0=5.26, lsat=403.3, rk=0.910, w=0.015, sd=0.052)
REFIT = dict(l0=5.26, lsat=397.2, rk=0.880, w=0.0125, sd=0.042)
ZOOM = (0.76, 1.015)


def load(name):
    with open(os.path.join(CAL, name)) as f:
        return [{k: float(v) for k, v in r.items() if k != "per_pod_m"} for r in csv.DictReader(f)]


def model(x, p):
    k = 1 / (1 + np.exp(-(x - p["rk"]) / p["w"]))
    return p["l0"] + (p["lsat"] - p["l0"]) * k, 1 - p["sd"] * k


v2 = load("sweep_v2.csv")
rho = np.array([r["rps"] for r in v2]) * M_PER_RPS / LIMIT
lat = np.array([r["mean_ms"] for r in v2])
suc = np.array([r["success"] for r in v2])
dense = [load(f"sweep_dense_A_r{i}.csv") for i in (1, 2, 3)]
d_rho = np.array([r["rps"] for r in dense[0]]) * M_PER_RPS / LIMIT
d_lat = np.array([[r["mean_ms"] for r in d] for d in dense])
d_suc = np.array([[r["success"] for r in d] for d in dense])

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.3), sharey="row",
                       gridspec_kw={"height_ratios": [1.45, 1], "width_ratios": [1, 1]})
(la, lb), (sa, sb) = ax
xa = np.linspace(0, 1.4, 800)
xb = np.linspace(*ZOOM, 800)
fa_l, fa_s = model(xa, FIT)
fb_l, fb_s = model(xb, FIT)
rb_l, rb_s = model(xb, REFIT)

for a in (la, sa):
    a.axvspan(*ZOOM, color="#f2f2f2", lw=0, zorder=0)
la.plot(xa, fa_l, color=BLUE, lw=1.5, zorder=2, label="fitted model")
la.plot(rho, lat, "o", ms=4.0, mfc="white", mec=INK, mew=1.0, zorder=3, label="fit sweep")
sa.plot(xa, fa_s, color=BLUE, lw=1.5, zorder=2)
sa.plot(rho, suc, "o", ms=4.0, mfc="white", mec=INK, mew=1.0, zorder=3)

lb.plot(xb, fb_l, color=BLUE, lw=1.5, zorder=2)
lb.plot(xb, rb_l, color=ORANGE, lw=1.5, ls=(0, (3.5, 1.8)), zorder=2, label="refit on all levels")
sb.plot(xb, fb_s, color=BLUE, lw=1.5, zorder=2)
sb.plot(xb, rb_s, color=ORANGE, lw=1.5, ls=(0, (3.5, 1.8)), zorder=2)
for a, y in ((lb, d_lat), (sb, d_suc)):
    m = y.mean(0)
    a.errorbar(d_rho, m, yerr=[m - y.min(0), y.max(0) - m], fmt="s", ms=3.6, color=INK, mfc=INK,
               mec="white", mew=0.6, elinewidth=0.8, capsize=1.6, zorder=3,
               label="dense sweep (3 runs)" if a is lb else None)

for a in (la, lb):
    a.axhline(20, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    a.set_yscale("log"); a.set_ylim(3, 900)
la.text(0.03, 23, r"$\ell_{\max} = 20$ ms", color=MUTED, fontsize=6.5)
la.set_ylabel("mean latency (ms)")
sa.set_ylabel("success ratio")
sa.set_ylim(0.935, 1.006)
for a in (la, sa):
    a.set_xlim(-0.03, 1.4); a.set_xticks([0, 0.5, 1.0])
for a in (lb, sb):
    a.set_xlim(*ZOOM); a.set_xticks([0.8, 0.9, 1.0])
for a in (la, lb):
    a.tick_params(labelbottom=False)
sa.set_xlabel(r"utilization $\rho$")
sb.set_xlabel(r"utilization $\rho$ (transition)")
la.set_title("(a) full range", fontsize=7.5, color=INK, loc="left", pad=3)
lb.set_title("(b) transition", fontsize=7.5, color=INK, loc="left", pad=3)
for a in ax.ravel():
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.5); a.set_axisbelow(True)
    a.tick_params(length=2.5, width=0.6)

h1, l1 = la.get_legend_handles_labels()
h2, l2 = lb.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=False, fontsize=6.8,
           handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.54, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.5, w_pad=0.6)
out = os.path.join(HERE, "knee_fit.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
print(out)
