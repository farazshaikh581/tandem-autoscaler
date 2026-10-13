"""Sim-vs-real comparison for a replayed policy (issue #4).

Runs the SAME trained policy on the simulator at the same trace window that
eval_real.py drove on the cluster, then merges the two per-step trajectories so
the sim and the real deployment can be compared step for step (latency, chosen
replicas/CPU, SLA). Produces sim_vs_real_<algo>_<workload>.csv (committed) and a
printed table.

  # 1) drive the real cluster first
  python eval_real.py --algo qrdqn --workload cpu-bound --start 3800 --steps 20
  # 2) sim counterpart + comparison
  python compare_sim_real.py --algo qrdqn --workload cpu-bound --start 3800 --steps 20
"""
import argparse
import csv
import os

import numpy as np

from envs.joint_hv_env import (ACTIONS, SLA_MEAN_LATENCY_MS, TRACE_FAMILIES, JointHVEnv,
                               set_trace_family)
from train import load_model

HERE = os.path.dirname(__file__)


def sim_rollout(algo, workload, start, steps, model_tag, cvar_alpha, seed):
    """Run the policy on the sim at a fixed trace start; return per-step rows."""
    env = JointHVEnv(workload=workload, episode_len=steps, backend="sim", seed=seed)
    mpath = os.path.join(HERE, "models", f"{algo}_{model_tag}")
    model = load_model(algo, mpath, env, cvar_alpha=cvar_alpha, seed=seed)
    obs, info = env.reset(seed=seed, options={"start": start})
    rows = []
    for i in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(a))
        m = info["metrics"]; s = env._backend.s
        rows.append(dict(step=i, replicas=s.replicas, cpu_req_m=s.cpu_req, mem_req_mi=s.mem_req,
                         load_frac=round(m["load_frac"], 3), latency_ms=round(m["latency"], 1),
                         sla_ok=int(m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99)))
        if term or trunc:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="qrdqn")
    ap.add_argument("--workload", default="cpu-bound")
    ap.add_argument("--app", default="cpu-burn", choices=["cpu-burn", "io-burn"],
                    help="which real deployment the real-csv was recorded against")
    ap.add_argument("--cluster-tag", default="",
                    help="tag for the cluster topology the real-csv was recorded against, "
                         "e.g. '4node' (issue #13); empty means the 2-worker testbed")
    ap.add_argument("--start", type=int, default=3800)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--model-tag", default="all")
    ap.add_argument("--cvar-alpha", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-csv", default=None)
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure",
                    help="must match the family the real-csv was recorded on")
    args = ap.parse_args()
    set_trace_family(args.trace_family, args.cluster_tag)

    # app_sfx/cluster_sfx must match eval_real.py's real_*.csv naming (both empty
    # for the original cpu-burn app on the 2-worker testbed, so the canonical
    # artifacts are untouched).
    app_sfx = "" if args.app == "cpu-burn" else f"_{args.app}"
    cluster_sfx = f"_{args.cluster_tag}" if args.cluster_tag else ""
    real_path = args.real_csv or os.path.join(
        HERE, "eval", f"real_{args.algo}_{args.workload}{app_sfx}{cluster_sfx}.csv")
    if not os.path.exists(real_path):
        raise SystemExit(f"real trajectory not found: {real_path}\nRun eval_real.py first.")
    with open(real_path) as f:
        real = {int(r["step"]): r for r in csv.DictReader(f)}

    sim = {r["step"]: r for r in sim_rollout(args.algo, args.workload, args.start, args.steps,
                                             args.model_tag, args.cvar_alpha, args.seed)}

    # non-default model tags (e.g. domain-rand "all_dr_seed0") get a filename suffix
    # so they don't clobber the canonical single-rep sim_vs_real_<algo>_<workload>.*
    sfx = "" if args.model_tag == "all" else "_" + args.model_tag.removeprefix("all_")
    out = os.path.join(HERE, "eval",
                        f"sim_vs_real_{args.algo}_{args.workload}{app_sfx}{cluster_sfx}{sfx}.csv")
    fields = ["step", "load_frac",
              "sim_replicas", "real_replicas", "sim_cpu_m", "real_cpu_m",
              "sim_mem_m", "real_mem_m",
              "sim_latency_ms", "real_latency_ms", "sim_sla", "real_sla"]
    print(f"{'step':>4}{'load':>6}{'sim_rep':>8}{'real_rep':>9}{'sim_cpu':>8}{'real_cpu':>9}"
          f"{'sim_lat':>8}{'real_lat':>9}{'sim_sla':>8}{'real_sla':>9}")
    rows, sim_sla, real_sla = [], 0, 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i in sorted(sim):
            s, r = sim[i], real.get(i)
            if r is None:
                continue
            row = dict(step=i, load_frac=s["load_frac"],
                       sim_replicas=s["replicas"], real_replicas=r["replicas"],
                       sim_cpu_m=s["cpu_req_m"], real_cpu_m=r["cpu_req_m"],
                       sim_mem_m=s["mem_req_mi"], real_mem_m=r["mem_req_mi"],
                       sim_latency_ms=s["latency_ms"], real_latency_ms=r["latency_ms"],
                       sim_sla=s["sla_ok"], real_sla=r["sla_ok"])
            w.writerow(row); rows.append(row)
            sim_sla += int(s["sla_ok"]); real_sla += int(r["sla_ok"])
            print(f"{i:>4}{s['load_frac']:>6}{s['replicas']:>8}{r['replicas']:>9}"
                  f"{s['cpu_req_m']:>8}{r['cpu_req_m']:>9}{s['latency_ms']:>8}"
                  f"{r['latency_ms']:>9}{s['sla_ok']:>8}{r['sla_ok']:>9}")
    n = len(rows)
    print(f"\nSLA: sim {100*sim_sla/n:.0f}%  real {100*real_sla/n:.0f}%  ({n} steps)  wrote {out}")


if __name__ == "__main__":
    main()
