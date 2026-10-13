"""Held-out generalization eval: run a trained model on unseen VMs (same class).

Companion to select_heldout.py. For each held-out VM trace it replays the model
(same deterministic eval protocol as train.evaluate) and reports the same metrics,
then aggregates per class (mean +/- std over the VMs) and prints it side by side
with the in-distribution numbers from eval/comparison.csv.

  python eval_heldout.py --algo qrdqn
  python eval_heldout.py --algo cvar-bellman-qrdqn

Writes eval/heldout_<algo>.csv (per-VM rows) + eval/heldout_<algo>_summary.csv.
"""
import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd

from envs.joint_hv_env import (TRACE_FAMILIES, JointHVEnv, SLA_MEAN_LATENCY_MS,
                               set_trace_family, trace_dir)
from train import CLASSES, HERE, load_model, make_env, _reversals

EPISODES = 20


def eval_one(model, cls: str, trace_path: str, seed: int) -> dict:
    """Deterministic replay of `model` on a single held-out VM trace."""
    env = JointHVEnv(workload=cls, episode_len=200, seed=seed + 777,
                     trace_paths={cls: trace_path})
    sla, lat, cost, churn = [], [], [], 0
    reversals = dict(replica=0, cpu=0, mem=0)
    for ep in range(EPISODES):
        obs, info = env.reset(seed=seed + 777 + ep)
        reps, cpus, mems, prev_a = [], [], [], None
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            m = info["metrics"]; s = env._backend.s
            lat.append(m["latency"])
            sla.append(m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99)
            cost.append(s.replicas * s.cpu_req / 1000.0)
            if prev_a is not None and int(a) != prev_a:
                churn += 1
            prev_a = int(a)
            reps.append(s.replicas); cpus.append(s.cpu_req); mems.append(s.mem_req)
            done = term or trunc
        reversals["replica"] += _reversals(reps)
        reversals["cpu"] += _reversals(cpus)
        reversals["mem"] += _reversals(mems)
    return dict(sla_pct=round(100 * np.mean(sla), 1),
                mean_lat_ms=round(float(np.mean(lat)), 1),
                mean_cost_vcpu=round(float(np.mean(cost)), 2),
                churn=churn, replica_reversals=reversals["replica"],
                cpu_reversals=reversals["cpu"], mem_reversals=reversals["mem"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cvar-alpha", type=float, default=0.25)
    ap.add_argument("--compare-suffix", default="",
                     help="in-dist comparison source: '' -> comparison.csv, 'azure' -> comparison_azure.csv")
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure",
                    help="held-out trace family; bitbrains runs need a 'bb' tag token")
    args = ap.parse_args()
    set_trace_family(args.trace_family, args.tag)

    name = f"{args.algo}_all{('_' + args.tag) if args.tag else ''}"
    mpath = os.path.join(HERE, "models", name)
    env = make_env("all", args.seed)  # placeholder env just to load the policy
    model = load_model(args.algo, mpath, env, cvar_alpha=args.cvar_alpha, seed=args.seed)
    print(f"[heldout] loaded {mpath}")

    traces = sorted(glob.glob(os.path.join(trace_dir("traces_heldout"), "*.npz")))
    rows = []
    for tp in traces:
        base = os.path.basename(tp)[:-4]           # "<class>__<vm>"
        cls, vm = base.split("__")
        r = eval_one(model, cls, tp, args.seed)
        rows.append(dict(workload=cls, vm=int(vm), **r))
        print(f"  {cls:13s} vm {vm:>5s}: SLA={r['sla_pct']:>5}%  lat={r['mean_lat_ms']:>5} ms  "
              f"cost={r['mean_cost_vcpu']:>4}  churn={r['churn']:>4}  "
              f"osc(r/c/m)={r['replica_reversals']}/{r['cpu_reversals']}/{r['mem_reversals']}")

    os.makedirs(os.path.join(HERE, "eval"), exist_ok=True)
    out = os.path.join(HERE, "eval", f"heldout_{name}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[heldout] wrote {out}")

    # aggregate per class + compare with in-distribution comparison.csv
    df = pd.DataFrame(rows)
    metrics = ["sla_pct", "mean_lat_ms", "mean_cost_vcpu", "churn",
               "replica_reversals", "cpu_reversals", "mem_reversals"]
    agg = df.groupby("workload")[metrics].agg(["mean", "std"]).round(2)
    comp_name = f"comparison_{args.compare_suffix}.csv" if args.compare_suffix else "comparison.csv"
    comp = pd.read_csv(os.path.join(HERE, "eval", comp_name))
    indist = comp[comp.algo == args.algo].set_index("workload") if "algo" in comp else None

    summ = []
    print(f"\n[heldout vs in-distribution]  algo={args.algo}  "
          f"(held-out = mean over unseen VMs, n per class in parens)")
    print(f"{'workload':13s}{'SLA% id/ho':>16s}{'lat id/ho':>16s}{'cost id/ho':>14s}"
          f"{'churn id/ho':>16s}")
    for cls in CLASSES:
        if cls not in df.workload.values:
            continue
        n = int((df.workload == cls).sum())
        ho = agg.loc[cls]
        id_row = indist.loc[cls] if indist is not None and cls in indist.index else None
        def cell(k, width):
            idv = f"{id_row[k]:g}" if id_row is not None else "-"
            hov = f"{ho[(k, 'mean')]:g}"
            return f"{idv + '/' + hov:>{width}s}"
        label = f"{cls} (n={n})"
        print(f"{label:13s}{cell('sla_pct', 16)}{cell('mean_lat_ms', 16)}"
              f"{cell('mean_cost_vcpu', 14)}{cell('churn', 16)}")
        srow = dict(workload=cls, n_vms=n)
        for k in metrics:
            srow[f"{k}_indist"] = float(id_row[k]) if id_row is not None else None
            srow[f"{k}_heldout_mean"] = float(ho[(k, "mean")])
            srow[f"{k}_heldout_std"] = float(ho[(k, "std")])
        summ.append(srow)

    sout = os.path.join(HERE, "eval", f"heldout_{name}_summary.csv")
    pd.DataFrame(summ).to_csv(sout, index=False)
    print(f"[heldout] wrote {sout}")


if __name__ == "__main__":
    main()
