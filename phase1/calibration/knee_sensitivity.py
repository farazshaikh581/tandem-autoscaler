"""Re-evaluate the trained controllers under the knee refitted to the dense sweep.

The service model was fitted on sweep_v2.csv, which has one level inside the
transition (rho 0.94). The dense sweep (sweep_dense_A_r{1,2,3}.csv, rho 0.78-1.0,
three repetitions) puts the measured 20 ms crossing at rho ~0.83 against the
model's 0.86. Refitting on all 37 levels (sweep_v2 + dense) gives the "refit"
constants below. Nothing is retrained: every controller is evaluated with the
standard 20-episode protocol under both the nominal and the refitted plant, so
the difference is the effect of the calibration error on the reported results.

MPC imports the constants by name when mpc_baseline is imported, so under
"refit" it keeps planning on the nominal model while the plant is the refitted
one, as a deployed model-based controller would.

Usage:
  python3 calibration/knee_sensitivity.py --family azure --condition refit
  python3 calibration/knee_sensitivity.py --summarize
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE1 = os.path.dirname(HERE)
sys.path.insert(0, PHASE1)

import envs.joint_hv_env as hv  # noqa: E402

CONDITIONS = {
    "nominal": dict(RHO_KNEE=0.91, KNEE_WIDTH=0.015, LAT_SAT_MS=403.0, SAT_DROP=0.052),
    "refit": dict(RHO_KNEE=0.88, KNEE_WIDTH=0.0125, LAT_SAT_MS=397.2, SAT_DROP=0.042),
}
SEEDS7 = range(7)
SEEDS3 = range(3)
BASELINES = ["hpa-only", "hpa-vpa", "slo-cost-aware", "slo-cost-aware-vpa",
             "predictive", "predictive-vpa", "mpc", "coscal"]
OUT_DIR = os.path.join(HERE, "knee_sensitivity")


def rl_jobs(family: str) -> list[tuple[str, int, str]]:
    """(algo, seed, model tag) for the multi-seed RL rows of the main comparison."""
    jobs = []
    for algo in ("qrdqn", "cvar-bellman-qrdqn", "cvar-qrdqn", "ppo", "dqn"):
        seeds = SEEDS7 if algo in ("qrdqn", "cvar-bellman-qrdqn") else SEEDS3
        for s in seeds:
            if family == "azure":
                tag = "azure" if s == 0 else f"azure_seed{s}"
            else:  # Bitbrains: seeds 0-2 under the original names, 3-6 under bb_seed<n>
                tag = "" if s == 0 else (f"seed{s}" if s < 3 else f"bb_seed{s}")
            jobs.append((algo, s, tag))
    return jobs


def run(family: str, condition: str) -> None:
    for k, v in CONDITIONS[condition].items():
        setattr(hv, k, v)
    hv.set_trace_family(family, "bb" if family == "bitbrains" else "azure")
    import train
    import eval_baselines as eb

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{family}_{condition}.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["algo"], int(r["seed"])) for r in csv.DictReader(f)}
    fields = ["family", "condition", "algo", "seed", "workload", "sla_pct", "mean_lat_ms",
              "mean_cost_vcpu", "churn", "replica_reversals", "cpu_reversals", "mem_reversals"]
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()

    def emit(algo, seed, rows):
        for r in rows:
            w.writerow(dict(family=family, condition=condition, algo=algo, seed=seed, **r))
        f.flush()
        print(f"[{family}/{condition}] {algo} seed{seed}: "
              + "  ".join(f"{r['workload']} {r['sla_pct']}" for r in rows), flush=True)

    for algo, seed, tag in rl_jobs(family):
        if (algo, seed) in done:
            continue
        name = f"{algo}_all{('_' + tag) if tag else ''}"
        mpath = os.path.join(PHASE1, "models", name)
        env = train.make_env("all", seed)
        model = train.load_model(algo, mpath, env, seed=seed)
        emit(algo, seed, train.evaluate(model, seed))

    tu = eb.HPA_TARGET_UTIL
    ctrl = {
        "hpa-only": lambda: eb.ClassicalController(vpa=False, target_util=tu),
        "hpa-vpa": lambda: eb.ClassicalController(vpa=True, target_util=tu),
        "slo-cost-aware": lambda: eb.SloCostAwareController(vpa=False),
        "slo-cost-aware-vpa": lambda: eb.SloCostAwareController(vpa=True),
        "predictive": lambda: eb.PredictiveController(vpa=False, target_util=tu),
        "predictive-vpa": lambda: eb.PredictiveController(vpa=True, target_util=tu),
        "mpc": lambda: eb.MPCController(),
        "coscal": eb._make_coscal,
    }
    for name in BASELINES:
        if (name, 0) in done:
            continue
        c = ctrl[name]()
        if c is None:
            continue
        emit(name, 0, eb.evaluate_policy(c, 0))
    f.close()


def summarize() -> None:
    import numpy as np
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "*_*.csv"))):
        if p.endswith("summary.csv"):
            continue
        with open(p) as f:
            rows += list(csv.DictReader(f))
    keys = sorted({(r["family"], r["algo"], r["workload"]) for r in rows})
    out = []
    for fam, algo, wl in keys:
        rec = dict(family=fam, algo=algo, workload=wl)
        for cond in CONDITIONS:
            sel = [r for r in rows if (r["family"], r["algo"], r["workload"], r["condition"])
                   == (fam, algo, wl, cond)]
            if not sel:
                continue
            for m in ("sla_pct", "mean_lat_ms", "mean_cost_vcpu"):
                v = np.array([float(r[m]) for r in sel])
                rec[f"{m}_{cond}"] = round(float(v.mean()), 2)
            rec[f"n_{cond}"] = len(sel)
        if "sla_pct_nominal" in rec and "sla_pct_refit" in rec:
            rec["d_sla"] = round(rec["sla_pct_refit"] - rec["sla_pct_nominal"], 2)
        out.append(rec)
    fields = sorted({k for r in out for k in r}, key=lambda k: (k not in ("family", "algo", "workload"), k))
    path = os.path.join(OUT_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    print(f"wrote {path}")
    for r in out:
        print(f"{r['family']:9s} {r['algo']:20s} {r['workload']:13s} "
              f"SLA {r.get('sla_pct_nominal', float('nan')):6.2f} -> {r.get('sla_pct_refit', float('nan')):6.2f} "
              f"(d {r.get('d_sla', float('nan')):+.2f})  cost {r.get('mean_cost_vcpu_nominal', float('nan')):.2f} "
              f"-> {r.get('mean_cost_vcpu_refit', float('nan')):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["azure", "bitbrains"])
    ap.add_argument("--condition", choices=list(CONDITIONS))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.family, a.condition)
