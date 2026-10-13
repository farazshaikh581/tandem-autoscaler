"""Static-configuration frontier: every fixed (replicas, CPU request, memory request)
evaluated with the standard protocol (20 episodes x 200 intervals per class, seed 0).
Two modes. "preset": the configuration is in place at the first interval and held with
the no-op action. "ramp": every episode starts from the shared initial configuration
(2 replicas, 200m, 128 MiB) like every controller, and each knob moves one step per
interval towards its target, then holds. Either way the size is chosen per class with
hindsight of the evaluation trace, so this is an optimistic bound for a fixed size.

  python3 static_frontier.py --trace-family azure|bitbrains [--mode ramp|preset]
writes eval/static_frontier_<family>[_preset].csv
"""
import argparse
import csv
import itertools
import os

import numpy as np

from envs.joint_hv_env import (ACTIONS, JointHVEnv, SLA_MEAN_LATENCY_MS, TRACE_FAMILIES,
                               set_trace_family)
from train import CLASSES

NOOP = ACTIONS.index((0, 0, 0))
REPS = range(1, 13)
CPUS = range(100, 801, 100)
MEMS = (128, 256, 384, 512)


def evaluate_static(wl, n, p, m, seed=0, episodes=20, mode="ramp"):
    env = JointHVEnv(workload=wl, episode_len=200, seed=seed + 777)
    sla, cost = [], []
    sgn = lambda d: (d > 0) - (d < 0)
    for ep in range(episodes):
        env.reset(seed=seed + 777 + ep)
        s = env._backend.s
        if mode == "preset":
            s.replicas = s.target_replicas = n
            s.cpu_req, s.mem_req = p, m
        done = False
        while not done:
            a = ACTIONS.index((sgn(n - s.target_replicas), 100 * sgn(p - s.cpu_req),
                               64 * sgn(m - s.mem_req)))
            _, _, term, trunc, info = env.step(a)
            met = info["metrics"]
            sla.append(met["latency"] < SLA_MEAN_LATENCY_MS and met["success"] >= 0.99)
            cost.append(s.replicas * s.cpu_req / 1000.0)
            done = term or trunc
    return 100 * float(np.mean(sla)), float(np.mean(cost))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure")
    ap.add_argument("--mode", choices=["ramp", "preset"], default="ramp")
    a = ap.parse_args()
    set_trace_family(a.trace_family, None)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval",
                       f"static_frontier_{a.trace_family}{'_preset' if a.mode == 'preset' else ''}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "replicas", "cpu_req", "mem_req", "sla_pct", "mean_cost_vcpu"])
        for wl in CLASSES:
            for n, p, m in itertools.product(REPS, CPUS, MEMS):
                sla, cost = evaluate_static(wl, n, p, m, mode=a.mode)
                w.writerow([wl, n, p, m, round(sla, 2), round(cost, 3)])
            f.flush()
            print("done", wl, flush=True)
    print(out)
