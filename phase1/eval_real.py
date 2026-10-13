"""Replay a sim-trained policy on the real hv-phase0 cluster (issue #4).

Loads a model trained by train.py and steps it through JointHVEnv with the
RealClusterBackend: each control step actuates the deployment (kubectl scale +
in-place resize) and measures real latency/CPU. Writes a per-step CSV and a
summary so sim vs real behaviour can be compared.

  python eval_real.py --algo qrdqn --workload cpu-bound --steps 30
  python eval_real.py --trace-family bitbrains --cluster-tag 4node_bb ...   # Bitbrains
                       (needs a 'bb' token in --cluster-tag so the files never collide)

Slow by design (each step drives ~step-seconds of HTTP load plus kubectl), so
keep --steps small. Only ever touches hv-phase0 (guarded in the backend).
"""
import argparse
import csv
import os

import numpy as np

from envs.joint_hv_env import (ACTIONS, SLA_MEAN_LATENCY_MS, TRACE_FAMILIES, JointHVEnv,
                               set_trace_family)
from train import load_model

HERE = os.path.dirname(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="qrdqn",
                    choices=["qrdqn", "dqn", "ppo", "trpo", "cvar-qrdqn",
                             "cvar-bellman-qrdqn", "sac-discrete"])
    ap.add_argument("--workload", default="cpu-bound",
                    help="class trace to replay (cpu-bound exercises the real CPU-only app)")
    ap.add_argument("--app", default="cpu-burn", choices=["cpu-burn", "io-burn"],
                    help="which real deployment to target (see envs/real_backend.py APP_PRESETS)")
    ap.add_argument("--cluster-tag", default="",
                    help="tag for the current cluster topology, e.g. '4node' (issue #13); "
                         "empty reproduces the original filename for the 2-worker testbed")
    ap.add_argument("--model-tag", default="all", help="model file is <algo>_<tag>")
    ap.add_argument("--steps", type=int, default=30, help="control steps (keep small; real time)")
    ap.add_argument("--start", type=int, default=None,
                    help="trace index to start at (default random; use a busy window for load)")
    ap.add_argument("--max-rps", type=int, default=180)
    ap.add_argument("--step-seconds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cvar-alpha", type=float, default=0.25)
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_trace_family(args.trace_family, args.cluster_tag)

    mpath = os.path.join(HERE, "models", f"{args.algo}_{args.model_tag}")
    env = JointHVEnv(workload=args.workload, episode_len=args.steps, backend="real",
                     seed=args.seed,
                     backend_kwargs=dict(max_rps=args.max_rps, step_seconds=args.step_seconds,
                                          app=args.app))
    model = load_model(args.algo, mpath, env, cvar_alpha=args.cvar_alpha, seed=args.seed)
    print(f"[eval-real] {args.algo} on '{args.workload}' vs real cluster ({args.app}), "
          f"{args.steps} steps @ {args.max_rps} max rps")

    # app_sfx/cluster_sfx keep a second app or a different cluster topology from
    # clobbering the 2-worker cpu-burn app's existing real_*.csv.
    app_sfx = "" if args.app == "cpu-burn" else f"_{args.app}"
    cluster_sfx = f"_{args.cluster_tag}" if args.cluster_tag else ""
    out = args.out or os.path.join(HERE, "eval",
                                    f"real_{args.algo}_{args.workload}{app_sfx}{cluster_sfx}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ["step", "action", "d_rep", "d_cpu", "d_mem", "replicas", "cpu_req_m",
              "mem_req_mi", "load_frac", "rho", "cpu_util", "latency_ms", "success",
              "mem_pressure", "sla_ok", "reward"]

    reset_opts = {"start": args.start} if args.start is not None else None
    obs, info = env.reset(seed=args.seed, options=reset_opts)
    rows, sla_hits = [], 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i in range(args.steps):
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            d_rep, d_cpu, d_mem = ACTIONS[a]
            obs, r, term, trunc, info = env.step(a)
            m = info["metrics"]; s = env._backend.s
            sla_ok = m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99
            sla_hits += int(sla_ok)
            row = dict(step=i, action=a, d_rep=d_rep, d_cpu=d_cpu, d_mem=d_mem,
                       replicas=s.replicas, cpu_req_m=s.cpu_req, mem_req_mi=s.mem_req,
                       load_frac=round(m["load_frac"], 3), rho=round(m["rho"], 3),
                       cpu_util=round(m["cpu_util"], 3), latency_ms=round(m["latency"], 1),
                       success=round(m["success"], 4), mem_pressure=round(m["mem_pressure"], 3),
                       sla_ok=int(sla_ok), reward=round(float(r), 3))
            w.writerow(row); f.flush(); rows.append(row)
            print(f"  step {i:>2} act={a:>2}({d_rep:+d},{d_cpu:+d},{d_mem:+d}) "
                  f"rep={s.replicas} cpu={s.cpu_req}m mem={s.mem_req}Mi "
                  f"lat={m['latency']:.1f}ms rho={m['rho']:.2f} sla={int(sla_ok)}")
            if term or trunc:
                break

    lat = [r["latency_ms"] for r in rows]
    print(f"\n[eval-real] {args.algo}/{args.workload}: SLA {100 * sla_hits / len(rows):.1f}% "
          f"mean_lat {np.mean(lat):.1f}ms  wrote {out}")
    # restore the deployment to baseline so the namespace is left clean
    b = env._backend
    b.reset(b.t)
    print(f"[eval-real] restored {b.deploy} to baseline "
          f"({b.base_replicas} replicas, {b.base_cpu_m}m/{b.base_mem_mi}Mi)")


if __name__ == "__main__":
    main()
