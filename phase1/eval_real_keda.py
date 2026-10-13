"""KEDA production-autoscaler baseline on the real hv-phase0 cluster.

The in-sim baselines (`hpa-only`, `slo-cost-aware`, `predictive`) are all
rule-based controllers running against the simulator. This replays a class trace
window on the *live* cluster while the actual KEDA operator drives horizontal
scaling, so the RL policy is measured against a production autoscaler, not only a
reimplementation of one. Analogue of the sim baselines' H-only / +VPA pair; both
variants use the same cpu-only ScaledObject (`scaledobject-cpu.yaml`) -- KEDA
never scales on memory here, mirroring the sim HPA, which is CPU-only:

  * default  -> `keda`.     KEDA owns replicas off CPU utilization; this script is
                            passive -- it only replays load and records metrics.
                            The memory request stays at baseline. (`hpa-only`.)
  * --vpa    -> `keda-vpa`. KEDA owns replicas off CPU utilization; this script
                            right-sizes the MEMORY request each step with the same
                            working-set x1.15 rule as ClassicalController(vpa=True)
                            in eval_baselines.py, actuated the way a non-in-place
                            k8s VPA does -- update the deployment request, roll the
                            pods (vpa_set_memory). (`hpa-vpa`.)

The ScaledObject must already be applied (see phase1/keda/). Only ever touches
hv-phase0 (guarded in RealClusterBackend). Restores the deployment to baseline on
exit; deleting the ScaledObject is the caller's job (run_keda_real.sh).

  python eval_real_keda.py --workload cpu-bound   --start 1095 --steps 20
  python eval_real_keda.py --workload memory-bound --start 7670 --steps 20 --vpa

Slow by design (each step drives ~step-seconds of HTTP load plus kubectl), so
keep --steps small.
"""
import argparse
import csv
import os

import numpy as np

from envs.joint_hv_env import (INTENSITY, MEM_MAX, MEM_MIN, SLA_MEAN_LATENCY_MS,
                               TRACE_FAMILIES, set_trace_family, trace_dir)
from envs.real_backend import RealClusterBackend
from train import _reversals

HERE = os.path.dirname(__file__)
VPA_MEM_HEADROOM = 1.15   # keep the memory request ~15% above the working set
                          # (identical to eval_baselines.ClassicalController)


def vpa_d_mem(mem_need_now: float, mem_req: int) -> int:
    """One +-64 MiB step of the ClassicalController(vpa=True) memory rule
    (eval_baselines.py): grow if under the working set + headroom, shrink only
    when comfortably (>64) over, clamp at the actuator bounds."""
    target = mem_need_now * VPA_MEM_HEADROOM
    if target > mem_req and mem_req < MEM_MAX:
        return 64
    if target < mem_req - 64 and mem_req > MEM_MIN:
        return -64
    return 0


def vpa_set_memory(backend, mem_mi: int) -> None:
    """Right-size the deployment's MEMORY request/limit to `mem_mi` and wait for
    the roll-out -- how a production k8s VPA that is NOT configured for in-place
    resize actuates: the updater evicts the pod and the admission controller
    recreates it at the new request. KEDA still owns replicas.

    Why not RealClusterBackend.apply(0, 0, d_mem): that drives memory through the
    in-place `resize` subresource, which silently no-ops for memory on this k3s
    when the container resizePolicy needs a restart (documented in
    real_backend.py). Using it here would score keda-vpa on an actuation-mechanism
    gap rather than on the VPA policy. The sim's ClassicalController(vpa=True)
    assumes the memory change lands; this makes the real run match that.
    """
    from envs.real_backend import NS
    from common import kubectl
    _, mem_r = backend._qos_ratios()
    if mem_r <= 1.0:
        mem_r = 2.0                      # web-service*.yaml ship limits = 2x requests
    lim = int(round(mem_mi * mem_r))
    kubectl(["set", "resources", f"deploy/{backend.deploy}", "-n", NS,
             f"--requests=memory={mem_mi}Mi", f"--limits=memory={lim}Mi"])
    kubectl(["rollout", "status", f"deploy/{backend.deploy}", "-n", NS,
             "--timeout=120s"], timeout=130)
    backend.s.mem_req = mem_mi
    backend.s.replicas = backend._ready_replicas() or backend.s.replicas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="cpu-bound",
                    choices=["cpu-bound", "memory-bound", "io-bound", "mixed"],
                    help="class trace to replay")
    ap.add_argument("--app", default="cpu-burn", choices=["cpu-burn", "io-burn"],
                    help="which real deployment to target (envs/real_backend.py APP_PRESETS)")
    ap.add_argument("--vpa", action="store_true",
                    help="right-size the memory request in-loop (keda-vpa); "
                         "default is passive (keda)")
    ap.add_argument("--steps", type=int, default=20, help="control steps (keep small; real time)")
    ap.add_argument("--start", type=int, default=None,
                    help="trace index to start at (default random; use a busy window)")
    ap.add_argument("--max-rps", type=int, default=180)
    ap.add_argument("--step-seconds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster-tag", default="",
                    help="tag for a non-default cluster topology, e.g. 'spiritquest'")
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_trace_family(args.trace_family, args.cluster_tag)

    name = "keda-vpa" if args.vpa else "keda"
    trace = dict(np.load(os.path.join(trace_dir(), f"{args.workload}.npz")))
    rng = np.random.default_rng(args.seed)
    backend = RealClusterBackend(trace, rng, INTENSITY[args.workload],
                                 max_rps=args.max_rps, step_seconds=args.step_seconds,
                                 app=args.app)
    start = args.start if args.start is not None else int(rng.integers(0, backend.n))
    print(f"[keda-real] {name} on '{args.workload}' vs real cluster ({args.app}), "
          f"{args.steps} steps @ {args.max_rps} max rps, start={start}")
    backend.reset(start)

    app_sfx = "" if args.app == "cpu-burn" else f"_{args.app}"
    cluster_sfx = f"_{args.cluster_tag}" if args.cluster_tag else ""
    out = args.out or os.path.join(
        HERE, "eval", f"real_{name}_{args.workload}{app_sfx}{cluster_sfx}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = ["step", "d_mem", "replicas", "cpu_req_m", "mem_req_mi", "load_frac",
              "rho", "cpu_util", "latency_ms", "success", "mem_pressure", "sla_ok"]

    rows, sla_hits = [], 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(args.steps):
            m = backend.step_physics()          # drives load, re-syncs s.replicas from readyReplicas
            s = backend.s
            d_mem = 0
            if args.vpa:
                d_mem = vpa_d_mem(s.mem_need_now, s.mem_req)
                if d_mem:
                    target_mem = int(np.clip(s.mem_req + d_mem, MEM_MIN, MEM_MAX))
                    vpa_set_memory(backend, target_mem)   # evict+recreate, like a real VPA
                    s = backend.s
            sla_ok = m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99
            sla_hits += int(sla_ok)
            row = dict(step=i, d_mem=d_mem, replicas=s.replicas, cpu_req_m=s.cpu_req,
                       mem_req_mi=s.mem_req, load_frac=round(m["load_frac"], 3),
                       rho=round(m["rho"], 3), cpu_util=round(m["cpu_util"], 3),
                       latency_ms=round(m["latency"], 1), success=round(m["success"], 4),
                       mem_pressure=round(m["mem_pressure"], 3), sla_ok=int(sla_ok))
            w.writerow(row)
            f.flush()
            rows.append(row)
            print(f"  step {i:>2} rep={s.replicas} cpu={s.cpu_req}m mem={s.mem_req}Mi "
                  f"d_mem={d_mem:+d} lat={m['latency']:.1f}ms rho={m['rho']:.2f} "
                  f"sla={int(sla_ok)}")

    reps = [r["replicas"] for r in rows]
    mems = [r["mem_req_mi"] for r in rows]
    lat = [r["latency_ms"] for r in rows]
    cost = [r["replicas"] * r["cpu_req_m"] / 1000.0 for r in rows]
    replica_changes = sum(1 for a, b in zip(reps, reps[1:]) if a != b)
    n = len(rows)
    print(f"\n[keda-real] {name}/{args.workload}: "
          f"SLA {100 * sla_hits / n:.1f}%  mean_lat {np.mean(lat):.1f}ms  "
          f"mean_cost {np.mean(cost):.2f}vCPU  replica_changes {replica_changes}  "
          f"replica_reversals {_reversals(reps)}  mem_reversals {_reversals(mems)}")
    print(f"[keda-real] wrote {out}")

    # restore the deployment to baseline so the namespace is left clean (the
    # ScaledObject teardown is the caller's job -- see run_keda_real.sh)
    if args.vpa:
        vpa_set_memory(backend, backend.base_mem_mi)   # undo the template memory bump
    backend.reset(backend.t)
    print(f"[keda-real] restored {backend.deploy} to baseline "
          f"({backend.base_replicas} replicas, {backend.base_cpu_m}m/{backend.base_mem_mi}Mi)")


if __name__ == "__main__":
    main()
