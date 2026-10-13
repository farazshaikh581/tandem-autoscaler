"""Cost/SLA Pareto-frontier sweep (#20): collect the tagged eval CSVs from the
reward-weight sweep (RL + MPC) and the utilisation-target sweep (classical), and
render the frontier per workload class.

Inputs (all in eval/, produced by train.py / eval_baselines.py with the sweep tags;
<fam> is the trace-family token, "azure" or "bb"):
  RL         eval_<algo>_all_<fam>_wc<c>_wv<v>[_seed<n>].csv
             algo in {qrdqn, cvar-bellman-qrdqn, cvar-qrdqn, ppo, dqn}
  MPC        eval_mpc_all_<fam>_wc<c>_wv<v>.csv
  classical  eval_<name>_all_<fam>_hpa<t>.csv
             name in {hpa-only, hpa-vpa, predictive, predictive-vpa}
  fixed pt   comparison_azure.csv / comparison_bitbrains.csv  (default-weight
             reference, drawn as a star)

Outputs (Bitbrains outputs carry a "_bitbrains" suffix):
  eval/pareto_frontier.csv       long table, one row per (controller, knob, seed, class)
  eval/pareto_frontier_agg.csv    seed-aggregated (mean/std) companion
  eval/cost_sla_pareto.png        w_viol = 1.0 slice, 2x2 by class, one line per controller
  eval/cost_sla_pareto_wviol.png  headline pair only, frontier shift across w_viol

  python pareto_sweep.py                # render everything available
  python pareto_sweep.py --dry-run       # list what is found / still missing
  python pareto_sweep.py --family bitbrains
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import CLASSES

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "eval")

RL_ALGOS = ["qrdqn", "cvar-bellman-qrdqn", "cvar-qrdqn", "ppo", "dqn"]
HEADLINE = ["qrdqn", "cvar-bellman-qrdqn"]
CLASSICAL = ["hpa-only", "hpa-vpa", "predictive", "predictive-vpa"]
DIST_RL = {"qrdqn", "cvar-bellman-qrdqn", "cvar-qrdqn"}
REV_COLS = ["replica_reversals", "cpu_reversals", "mem_reversals"]

COLOR = {
    "qrdqn": "#4C78A8", "cvar-bellman-qrdqn": "#F58518", "cvar-qrdqn": "#54A24B",
    "ppo": "#B279A2", "dqn": "#9D755D", "mpc": "#E45756",
    "hpa-only": "#72B7B2", "hpa-vpa": "#0F8299",
    "predictive": "#BAB0AC", "predictive-vpa": "#7F7F7F",
}
FAMILY = {**{a: "distributional RL" for a in DIST_RL},
          "ppo": "value / on-policy RL", "dqn": "value / on-policy RL",
          "mpc": "MPC", **{c: "classical" for c in CLASSICAL}}

_WCWV = re.compile(r"_wc([0-9]*\.?[0-9]+)_wv([0-9]*\.?[0-9]+)")
_HPA = re.compile(r"_hpa([0-9]*\.?[0-9]+)")
_SEED = re.compile(r"_seed([0-9]+)\.csv$")

# trace family -> (file-name token, default-weight reference table, output suffix)
FAMILIES = {
    "azure": ("azure", "comparison_azure.csv", ""),
    "bitbrains": ("bb", "comparison_bitbrains.csv", "_bitbrains"),
}


def _rows_from(path: str, controller: str, knob: str, knob_value: float,
               w_viol: float, seed: int) -> list[dict]:
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        out.append(dict(
            controller=controller, family=FAMILY[controller], knob=knob,
            knob_value=knob_value, w_viol=w_viol, seed=seed,
            workload=r["workload"], sla_pct=r["sla_pct"],
            mean_lat_ms=r["mean_lat_ms"], mean_cost_vcpu=r["mean_cost_vcpu"],
            churn=r["churn"],
            total_reversals=int(sum(r[c] for c in REV_COLS)),
            **{c: int(r[c]) for c in REV_COLS},
        ))
    return out


def collect(fam: str) -> pd.DataFrame:
    rows: list[dict] = []
    missing_note = []

    # RL: eval_<algo>_all_<fam>_wc<c>_wv<v>[_seed<n>].csv
    for algo in RL_ALGOS:
        hits = glob.glob(os.path.join(EVAL, f"eval_{algo}_all_{fam}_wc*_wv*.csv"))
        for p in sorted(hits):
            m = _WCWV.search(os.path.basename(p))
            if not m:
                continue
            wc, wv = float(m.group(1)), float(m.group(2))
            sm = _SEED.search(os.path.basename(p))
            seed = int(sm.group(1)) if sm else 0
            rows += _rows_from(p, algo, "w_cost", wc, wv, seed)

    # MPC: eval_mpc_all_<fam>_wc<c>_wv<v>.csv
    for p in sorted(glob.glob(os.path.join(EVAL, f"eval_mpc_all_{fam}_wc*_wv*.csv"))):
        m = _WCWV.search(os.path.basename(p))
        if m:
            rows += _rows_from(p, "mpc", "w_cost", float(m.group(1)), float(m.group(2)), 0)

    # classical: eval_<name>_all_<fam>_hpa<t>.csv
    for name in CLASSICAL:
        for p in sorted(glob.glob(os.path.join(EVAL, f"eval_{name}_all_{fam}_hpa*.csv"))):
            m = _HPA.search(os.path.basename(p))
            if m:
                rows += _rows_from(p, name, "hpa_target", float(m.group(1)), np.nan, 0)

    if not rows:
        raise SystemExit(f"no sweep eval CSVs found in eval/ (expected *_{fam}_wc*_wv*.csv / *_{fam}_hpa*.csv)")
    df = pd.DataFrame(rows)

    # coverage report
    for algo in RL_ALGOS:
        sub = df[df.controller == algo]
        cells = sub.groupby(["knob_value", "w_viol"]).ngroups
        seeds = sorted(sub.seed.unique().tolist())
        print(f"  {algo:22s} {len(sub):4d} rows  {cells:2d} (w_cost,w_viol) cells  seeds {seeds}")
    for name in ["mpc", *CLASSICAL]:
        sub = df[df.controller == name]
        if len(sub):
            knobs = sorted(sub.knob_value.unique().tolist())
            print(f"  {name:22s} {len(sub):4d} rows  {len(knobs):2d} {sub.knob.iloc[0]} values {knobs}")
        else:
            missing_note.append(name)
    if missing_note:
        print(f"  (no data yet for: {', '.join(missing_note)})")
    return df


def fixed_points(table: str) -> pd.DataFrame:
    """Default-weight reference from the family's comparison table, one (algo, class) -> (cost, sla)."""
    p = os.path.join(EVAL, table)
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p)
    keep = d[d.algo.isin(RL_ALGOS + ["mpc"] + CLASSICAL)]
    return keep[["algo", "workload", "mean_cost_vcpu", "sla_pct"]].rename(columns={"algo": "controller"})


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["controller", "family", "knob", "knob_value", "w_viol", "workload"], dropna=False)
    return g.agg(
        sla_mean=("sla_pct", "mean"), sla_std=("sla_pct", "std"),
        cost_mean=("mean_cost_vcpu", "mean"), cost_std=("mean_cost_vcpu", "std"),
        lat_mean=("mean_lat_ms", "mean"),
        rev_mean=("total_reversals", "mean"), rev_std=("total_reversals", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()


def plot_main(agg: pd.DataFrame, fp: pd.DataFrame, wviol: float, tag: str) -> None:
    sl = agg[(agg.w_viol == wviol) | (agg.knob == "hpa_target")].copy()
    if sl.empty:
        print(f"  [skip] no data at w_viol={wviol}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, wl in zip(axes.ravel(), CLASSES):
        d = sl[sl.workload == wl]
        for ctrl, sub in d.groupby("controller"):
            sub = sub.sort_values("cost_mean")
            ax.plot(sub.cost_mean, sub.sla_mean, marker="o", ms=5, lw=1.6,
                    color=COLOR.get(ctrl, "#333"),
                    ls="--" if FAMILY[ctrl] == "classical" else "-",
                    label=ctrl)
            has_std = sub.sla_std.notna() & (sub.n_seeds > 1)
            if has_std.any():
                s = sub[has_std]
                ax.errorbar(s.cost_mean, s.sla_mean, yerr=s.sla_std, xerr=s.cost_std,
                            fmt="none", ecolor=COLOR.get(ctrl, "#333"), alpha=0.5, lw=1)
        if not fp.empty:
            for ctrl, s in fp[fp.workload == wl].groupby("controller"):
                ax.scatter(s.mean_cost_vcpu, s.sla_pct, marker="*", s=140, zorder=5,
                           color=COLOR.get(ctrl, "#333"), edgecolor="black", lw=0.6)
        ax.set_title(wl)
        ax.set_xlabel("mean cost (vCPU)")
        ax.set_ylabel("SLA (%)")
        ax.grid(True, color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Cost / SLA frontier  (w_viol = {wviol}; star = default-weight point)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(EVAL, f"cost_sla_pareto{tag}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  wrote {out}")


def plot_wviol(agg: pd.DataFrame, tag: str) -> None:
    d = agg[agg.controller.isin(HEADLINE) & (agg.knob == "w_cost")]
    wvs = sorted(d.w_viol.dropna().unique().tolist())
    if len(wvs) < 2:
        print("  [skip] cost_sla_pareto_wviol: need >= 2 w_viol levels")
        return
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, wl in zip(axes.ravel(), CLASSES):
        for ctrl in HEADLINE:
            for wv in wvs:
                sub = d[(d.controller == ctrl) & (d.w_viol == wv) & (d.workload == wl)].sort_values("cost_mean")
                if sub.empty:
                    continue
                ax.plot(sub.cost_mean, sub.sla_mean, marker="o", ms=4, lw=1.4,
                        color=cmap(wvs.index(wv) / max(1, len(wvs) - 1)),
                        ls="-" if ctrl == "qrdqn" else "--",
                        label=f"{ctrl}, w_viol={wv}")
        ax.set_title(wl)
        ax.set_xlabel("mean cost (vCPU)")
        ax.set_ylabel("SLA (%)")
        ax.grid(True, color="#e6e8eb", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(wvs), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Frontier shift with the SLA-violation weight  (solid QR-DQN, dashed Option B)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(EVAL, f"cost_sla_pareto_wviol{tag}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-wviol", type=float, default=1.0,
                    help="w_viol level for the main frontier figure (default 1.0)")
    ap.add_argument("--family", choices=list(FAMILIES), default="azure",
                    help="trace family whose sweep to collect (default azure)")
    ap.add_argument("--tag", default="", help="extra output filename suffix")
    ap.add_argument("--dry-run", action="store_true", help="only report coverage, do not write")
    args = ap.parse_args()
    fam, table, fam_sfx = FAMILIES[args.family]
    tag = fam_sfx + (f"_{args.tag}" if args.tag else "")

    print(f"== sweep eval coverage ({args.family}) ==")
    df = collect(fam)
    if args.dry_run:
        return
    df.to_csv(os.path.join(EVAL, f"pareto_frontier{tag}.csv"), index=False)
    agg = _agg(df)
    agg.to_csv(os.path.join(EVAL, f"pareto_frontier_agg{tag}.csv"), index=False)
    print(f"  wrote pareto_frontier{tag}.csv ({len(df)} rows), pareto_frontier_agg{tag}.csv ({len(agg)} rows)")
    plot_main(agg, fixed_points(table), args.slice_wviol, tag)
    plot_wviol(agg, tag)


if __name__ == "__main__":
    main()
