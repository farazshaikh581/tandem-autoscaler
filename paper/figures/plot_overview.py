"""Two full-page result figures (Elsevier full text width x most of the text height).
overview_sim.pdf   (a) SLO compliance per class, (b) CPU and reversals, (c) action-space
                   ablation, (d) published and model-based baselines, (e) CPU vs SLO compliance.
overview_beyond.pdf (a) held-out machines, (b) simulator vs real cluster, (c) KEDA on the
                   real cluster, (d) co-located functions.
python3 paper/figures/plot_overview.py"""
import matplotlib.pyplot as plt

import figstyle as fs
import panels as P

W, H = 7.16, 9.3


def header(fig, ax_left, letter, text, handles=None, labels=None, ncol=5, dy=0.03, lx=0.0, x0=0.008,
           second_line=False):
    """Panel letter and title on one line above the panel titles, legend on the same line."""
    pos = ax_left.get_position()
    y = pos.y1 + dy
    fig.text(x0, y, f"({letter})", fontsize=8.5, fontweight="bold", color=fs.INK, va="baseline")
    fig.text(x0 + 0.037, y, text, fontsize=8, color=fs.INK, va="baseline")
    if handles:
        fig.legend(handles, labels, loc="lower left", ncol=ncol, frameon=False, fontsize=6.6,
                   handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(x0 + 0.037 + lx, y - (0.03 if second_line else 0.006)),
                   borderaxespad=0, handletextpad=0.5)


fs.setup()
# ---------------------------------------------------------------- page 1: simulation
fig = plt.figure(figsize=(W, H))
gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.35, 1.75, 2.6], hspace=0.5,
                      left=0.075, right=0.99, top=0.945, bottom=0.05)
g_a = gs[0].subgridspec(1, 2, wspace=0.05)
a = [fig.add_subplot(g_a[0, i]) for i in range(2)]
g_bc = gs[1].subgridspec(1, 5, width_ratios=[1, 1, 0.25, 1, 1], wspace=0.12)
b = [fig.add_subplot(g_bc[0, 0]), fig.add_subplot(g_bc[0, 1])]
c = [fig.add_subplot(g_bc[0, 3]), fig.add_subplot(g_bc[0, 4])]
g_d = gs[2].subgridspec(1, 6, width_ratios=[0.42, 1, 1, 1, 1, 1], wspace=0.1)
d = [fig.add_subplot(g_d[0, i]) for i in range(1, 6)]
g_e = gs[3].subgridspec(2, 5, width_ratios=[0.07, 1, 1, 1, 1], wspace=0.08, hspace=0.32)
e = [[fig.add_subplot(g_e[r, i]) for i in range(1, 5)] for r in range(2)]
ha = P.slo_bars(a)
hb = P.cost_rev(*b)
hc = P.ablation(c)
hd = P.dotplot(d, ["cvar-bellman-qrdqn", None, "mpc", "pandey-ppo", "pandey-ddpg", "coscal",
                   "slo-cost-aware", "slo-cost-aware-vpa"], divider=1)
he = P.pareto(e)
fig.canvas.draw()
header(fig, a[0], "a", "SLO compliance per class", *ha, lx=0.24)
header(fig, b[0], "b", "Cost and stability", *hb, ncol=2, lx=0.0, dy=0.05, second_line=True)
header(fig, c[0], "c", "Action-space ablation (TANDEM)", *hc, ncol=3, dy=0.05, lx=0.0,
       x0=c[0].get_position().x0 - 0.05, second_line=True)
header(fig, d[0], "d", "Published and model-based baselines", *hd, ncol=3, lx=0.33)
header(fig, e[0][0], "e", "Allocated CPU against SLO compliance", *he, ncol=5, lx=0.35)
fs.save(fig, "overview_sim")

# ---------------------------------------------------------------- page 2: beyond
fig = plt.figure(figsize=(W, H * 0.78))
gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.9, 1.55], hspace=0.7,
                      left=0.075, right=0.99, top=0.92, bottom=0.07)
g_a = gs[0].subgridspec(1, 4, wspace=0.06)
a = [fig.add_subplot(g_a[0, i]) for i in range(4)]
g_bc = gs[1].subgridspec(1, 6, width_ratios=[0.55, 1, 1, 0.3, 1, 1], wspace=0.1)
b = [fig.add_subplot(g_bc[0, 1]), fig.add_subplot(g_bc[0, 2])]
c = [fig.add_subplot(g_bc[0, 4]), fig.add_subplot(g_bc[0, 5])]
g_d = gs[2].subgridspec(1, 4, wspace=0.06)
d = [fig.add_subplot(g_d[0, i]) for i in range(4)]
ha = P.heldout(a)
hb = P.simreal(b)
hc = P.keda(c)
hd = P.multifn(d)
fig.canvas.draw()
header(fig, a[0], "a", "Held-out machines", *ha, ncol=4, lx=0.17, dy=0.04)
header(fig, b[0], "b", "Simulator and real cluster, same window", dy=0.08, x0=0.008)
fig.legend(*hb, loc="lower left", ncol=4, frameon=False, fontsize=6.6, handlelength=1.4, columnspacing=0.8,
           bbox_to_anchor=(0.045, b[0].get_position().y1 + 0.03), borderaxespad=0)
header(fig, c[0], "c", "KEDA on the real cluster", dy=0.08, x0=c[0].get_position().x0 - 0.05)
fig.legend(*hc, loc="lower left", ncol=4, frameon=False, fontsize=6.6, handlelength=1.4, columnspacing=0.8,
           bbox_to_anchor=(c[0].get_position().x0 - 0.013, c[0].get_position().y1 + 0.03), borderaxespad=0)
header(fig, d[0], "d", "Four co-located functions under limited capacity", *hd, ncol=4, lx=0.43, dy=0.04)
fs.save(fig, "overview_beyond")
