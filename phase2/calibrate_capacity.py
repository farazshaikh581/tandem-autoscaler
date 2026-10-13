"""Size the node pool from measured demand, not from a guessed millicore number (#22).

Capacity is expressed as a dimensionless oversubscription ratio

    phi = total_capacity / p95(summed allocation the controllers actually ask for)

so that phi is comparable across workload mixes and across N. phi=inf is the
non-binding control, phi=1.0 is "provisioned for the aggregate p95, not the
aggregate peak", which is roughly how clusters are really run, and below that the
pool is deliberately undersized.

The reference is measured by running the controllers at infinite capacity and
recording what they ask for, so the sweep is anchored to this env's own dynamics
rather than to a number carried over from the testbed.

    python calibrate_capacity.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_fn_env import DEFAULT_CLASSES, NODES, MultiFnEnv  # noqa: E402
from envs.joint_hv_env import ACTIONS  # noqa: E402
from eval_baselines import ClassicalController  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHI_SWEEP = [None, 1.5, 1.0, 0.75, 0.5]   # None = infinite capacity


def make_hpa_vpa(n: int):
    """The reference controller for calibration: one independent HPA+VPA per
    function, which is what a cluster running N functions actually looks like."""
    return [ClassicalController(vpa=True, target_util=0.70) for _ in range(n)]


def run(controllers, classes, node_cpu_m, seed, episodes, episode_len=200, strict=False,
        profiles="legacy"):
    """Roll out N per-function controllers and report demand + outcome."""
    env = MultiFnEnv(classes=classes, node_cpu_m=node_cpu_m, seed=seed,
                     episode_len=episode_len, strict=strict, profiles=profiles)
    alloc, sla_per_fn = [], [[] for _ in classes]
    lat, cost, denied, deferred = [], [], 0, 0
    for ep in range(episodes):
        _, infos = env.reset(seed=seed + ep)
        for c in controllers:
            c.reset()
        for _ in range(episode_len):
            acts = [ACTIONS.index(c.act(inf["metrics"], s))
                    for c, inf, s in zip(controllers, infos, env.states)]
            _, _, _, _, infos = env.step(acts)
            # demand = what the controllers asked for, independent of what they got
            alloc.append(sum(s.target_replicas * s.cpu_req for s in env.states))
            for i, inf in enumerate(infos):
                m = inf["metrics"]
                sla_per_fn[i].append(bool(m["sla_ok"]))
                lat.append(m["latency"])
                cost.append(m["cost_vcpu"])
                denied += int(m["denied_replicas"] > 0)
                deferred += int(m["resize_deferred"])
    per_fn = [100.0 * float(np.mean(x)) for x in sla_per_fn]
    return dict(p50=float(np.percentile(alloc, 50)), p95=float(np.percentile(alloc, 95)),
                p99=float(np.percentile(alloc, 99)), mean_alloc=float(np.mean(alloc)),
                sla_pct=float(np.mean(per_fn)), per_fn_sla=per_fn,
                sla_min=min(per_fn), sla_max=max(per_fn),
                mean_lat_ms=float(np.mean(lat)), mean_cost_vcpu=float(np.mean(cost)),
                denied_steps=denied, deferred_steps=deferred)


def node_cpu_for(phi: float | None, ref_p95: float, n_nodes: int = NODES) -> float | None:
    """phi -> per-node millicores, on a 100m grid (k8s millicore granularity).

    A coarser 500m grid collapses the low end of the sweep: at N=4 the measured
    p95 demand is only ~1600m, so phi=1.0 and phi=0.5 would round to the same
    per-node size and the sweep would silently stop varying.
    """
    if phi is None:
        return None
    return max(100.0, round(phi * ref_p95 / n_nodes / 100.0) * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    ref = run(make_hpa_vpa(len(classes)), classes, None, args.seed, args.episodes)
    print(f"reference demand at infinite capacity (hpa-vpa x{len(classes)}):")
    print(f"  p50 {ref['p50']:.0f}m   p95 {ref['p95']:.0f}m   p99 {ref['p99']:.0f}m"
          f"   mean {ref['mean_alloc']:.0f}m")
    print(f"  SLA {ref['sla_pct']:.1f}%  lat {ref['mean_lat_ms']:.1f}ms\n")
    print(f"{'phi':>6}  {'node_cpu_m':>10}  {'total':>7}  {'~6-core nodes':>13}")
    out = {"classes": classes, "reference": ref, "grid": []}
    for phi in PHI_SWEEP:
        ncpu = node_cpu_for(phi, ref["p95"])
        total = "inf" if ncpu is None else f"{ncpu * NODES:.0f}m"
        eq = "-" if ncpu is None else f"{ncpu * NODES / 6000.0:.2f}"
        print(f"{str(phi):>6}  {str(ncpu):>10}  {total:>7}  {eq:>13}")
        out["grid"].append({"phi": phi, "node_cpu_m": ncpu})
    path = os.path.join(HERE, "capacity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
