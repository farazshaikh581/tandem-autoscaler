"""Figure: sim-vs-real trajectory for a replayed policy (issue #4).

Two stacked panels sharing the step axis (never a dual y-axis): measured latency
(log scale, with the SLA line) and the replica count the policy chose, sim vs
real. Identity is carried by colour AND line style + marker, so it survives
greyscale print and colour-vision deficiency.

  python plot_sim_real.py --algo qrdqn --workload cpu-bound
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.joint_hv_env import SLA_MEAN_LATENCY_MS

HERE = os.path.dirname(__file__)
SIM = "#4C78A8"    # blue  (CVD-safe pair; also distinguished by dashed + circle)
REAL = "#F58518"   # orange (solid + square)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="qrdqn")
    ap.add_argument("--workload", default="cpu-bound")
    ap.add_argument("--app", default="cpu-burn", choices=["cpu-burn", "io-burn"])
    ap.add_argument("--cluster-tag", default="",
                    help="tag for the cluster topology, e.g. '4node' (issue #13)")
    ap.add_argument("--model-tag", default="all")
    args = ap.parse_args()

    app_sfx = "" if args.app == "cpu-burn" else f"_{args.app}"
    cluster_sfx = f"_{args.cluster_tag}" if args.cluster_tag else ""
    sfx = "" if args.model_tag == "all" else "_" + args.model_tag.removeprefix("all_")
    path = os.path.join(HERE, "eval",
                        f"sim_vs_real_{args.algo}_{args.workload}{app_sfx}{cluster_sfx}{sfx}.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    step = [int(r["step"]) for r in rows]
    def col(k): return [float(r[k]) for r in rows]

    fig, (ax_lat, ax_rep) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2]})

    # --- latency (log; SLA reference) ---
    ax_lat.axhline(SLA_MEAN_LATENCY_MS, color="#9aa0a6", ls=":", lw=1.2, zorder=1)
    ax_lat.text(step[-1], SLA_MEAN_LATENCY_MS * 1.15, f"SLA {SLA_MEAN_LATENCY_MS:.0f} ms",
                color="#6b7075", fontsize=8, ha="right", va="bottom")
    ax_lat.plot(step, col("sim_latency_ms"), color=SIM, ls="--", marker="o", ms=4,
                lw=2, label="sim", zorder=3)
    ax_lat.plot(step, col("real_latency_ms"), color=REAL, ls="-", marker="s", ms=4,
                lw=2, label="real", zorder=3)
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("mean latency (ms)")
    ax_lat.legend(frameon=False, loc="upper right", ncol=2)
    regime = "memory-stress (OOM cliff)" if args.workload == "memory-bound" else "~1200 rps"
    app_label = "" if args.app == "cpu-burn" else f", {args.app} app"
    cluster_label = "" if not args.cluster_tag else f", {args.cluster_tag}"
    ax_lat.set_title(f"Sim vs real replay: {args.algo.upper()}{sfx.replace('_', ' ')}, "
                     f"{args.workload}, {regime}{app_label}{cluster_label}", fontsize=11, loc="left")

    # --- actuated knob (step-hold): the memory ratchet for the memory-bound class,
    # otherwise the replica count. memory-bound is a vertical-scaling story, so the
    # replica panel would hide the axis the policy actually drives to escape OOM.
    knob, knob_label = ("mem_m", "memory request (Mi)") if args.workload == "memory-bound" \
        else ("replicas", "replicas")
    ax_rep.step(step, col(f"sim_{knob}"), where="post", color=SIM, ls="--", lw=2, label="sim")
    ax_rep.step(step, col(f"real_{knob}"), where="post", color=REAL, ls="-", lw=2, label="real")
    ax_rep.set_ylabel(knob_label)
    ax_rep.set_xlabel("control step")

    for ax in (ax_lat, ax_rep):
        ax.grid(True, axis="y", color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = os.path.join(HERE, "eval",
                        f"sim_vs_real_{args.algo}_{args.workload}{app_sfx}{cluster_sfx}{sfx}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
