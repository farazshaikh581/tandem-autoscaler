"""Summary of the real-cluster repetitions (run_real_reps.sh).

Per cell, trace family and controller: SLO compliance of each repetition, its mean,
minimum and maximum, and, for the learned controllers, the simulator's compliance
on the same window for the same model seeds (deterministic policy, nominal plant).
Application performance over all intervals of the five runs pooled: mean and 95th
percentile of the per-interval mean latency (an interval whose requests all time out
counts at the 5 s client timeout) and the served fraction, sum(load * success) /
sum(load). Writes eval/real_reps_summary.csv.

  python real_reps_summary.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from envs.joint_hv_env import set_trace_family  # noqa: E402

# cell -> (workload, window key, steps); must match run_real_reps.sh,
# run_real_mem_recreate.sh (-recreate) and run_real_io_perf.sh (-perf)
CELLS = {"cpu-bound_2node": ("cpu-bound", "cpu", 20), "cpu-bound_4node": ("cpu-bound", "cpu", 20),
         "io-bound_io-burn": ("io-bound", "io", 20), "mixed_io-burn": ("mixed", "mixed", 15),
         "kedacmp_cpu-bound": ("cpu-bound", "kedacpu", 20),
         "kedacmp_memory-bound": ("memory-bound", "mem", 20),
         "kedacmp_memory-bound-recreate": ("memory-bound", "mem", 20),
         "io-bound_io-burn-perf": ("io-bound", "io", 20), "mixed_io-burn-perf": ("mixed", "mixed", 15)}
WIN = {("azure", "cpu"): 1097, ("azure", "kedacpu"): 1095, ("azure", "io"): 957,
       ("azure", "mixed"): 9, ("azure", "mem"): 7670, ("bitbrains", "cpu"): 3800,
       ("bitbrains", "kedacpu"): 3800, ("bitbrains", "io"): 4051, ("bitbrains", "mixed"): 76,
       ("bitbrains", "mem"): 7670}
RL = ("cvar-bellman-qrdqn", "qrdqn")


def model_tag(fam, s):
    if fam == "azure":
        return "all_azure" if s == 0 else f"all_azure_seed{s}"
    return {0: "all", 1: "all_seed1", 2: "all_seed2"}.get(s, f"all_bb_seed{s}")


def sla(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    return 100 * np.mean([int(x["sla_ok"]) for x in r])


def intervals(path):
    with open(path) as f:
        return [(float(x["latency_ms"]), float(x["success"]), float(x["load_frac"])) for x in csv.DictReader(f)]


def perf(rows):
    a = np.array(rows)
    lat, suc, load = a[:, 0], a[:, 1], a[:, 2]
    return (round(float(lat.mean()), 1), round(float(np.percentile(lat, 95)), 1),
            round(100 * float((load * suc).sum() / max(load.sum(), 1e-12)), 1))


def main():
    runs, ivals = {}, {}
    for p in glob.glob(os.path.join(HERE, "eval", "real_reps", "*.csv")):
        cell, ctrl, fam, rep = os.path.basename(p)[:-4].split("__")
        runs.setdefault((cell, fam, ctrl), {})[int(rep[3:])] = sla(p)
        ivals.setdefault((cell, fam, ctrl), []).extend(intervals(p))
    sim = {}
    for fam in ("azure", "bitbrains"):
        set_trace_family(fam)
        from compare_sim_real import sim_rollout
        for (cell, f, ctrl), reps in runs.items():
            if f != fam or ctrl not in RL:
                continue
            wl, key, steps = CELLS[cell]
            vals = []
            for s in sorted(reps):
                rows = sim_rollout(ctrl, wl, WIN[(fam, key)], steps, model_tag(fam, s), 0.25, s)
                vals.append(100 * np.mean([r["sla_ok"] for r in rows]))
            sim[(cell, fam, ctrl)] = float(np.mean(vals))
    out = os.path.join(HERE, "eval", "real_reps_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "family", "controller", "reps", "real_mean", "real_min", "real_max",
                    "real_per_rep", "sim_mean", "lat_mean_ms", "lat_p95_ms", "served_pct"])
        for k in sorted(runs):
            v = [runs[k][i] for i in sorted(runs[k])]
            w.writerow([*k, len(v), round(np.mean(v), 1), round(min(v), 1), round(max(v), 1),
                        " ".join(f"{x:.1f}" for x in v), round(sim[k], 1) if k in sim else "",
                        *perf(ivals[k])])
            lm, lp, sv = perf(ivals[k])
            print(f"{k[0]:29s} {k[1]:9s} {k[2]:19s} real {np.mean(v):5.1f} [{min(v):5.1f},{max(v):5.1f}]"
                  f"  lat {lm:7.1f} p95 {lp:7.1f} served {sv:5.1f}" + (f"  sim {sim[k]:5.1f}" if k in sim else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
