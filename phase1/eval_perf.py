"""Application performance of every controller in the main comparison.

SLO compliance counts intervals that meet the latency and success targets; it does
not show how far latency goes when an interval misses, or how many requests are
lost. This re-runs the standard 20-episode evaluation (same models, seeds and
episodes as the main table, nominal plant) and records, per interval, the mean
request latency, the success ratio and the offered load. Per class it reports:

  lat_mean_ms     mean of the per-interval mean latency (equals mean_lat_ms)
  lat_p95_ms      95th percentile of the per-interval mean latency
  lat_p99_ms      99th percentile of the per-interval mean latency
  served_pct      served requests / offered requests (load-weighted success)
  lost_per_1000   requests lost per 1000 offered

train.evaluate and eval_baselines.evaluate_policy run unchanged; JointHVEnv.step
is wrapped only to log what it already returns, and sla_pct is re-derived as a
check against the committed main-table numbers.

  python eval_perf.py --family azure
  python eval_perf.py --family bitbrains
  python eval_perf.py --summarize
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "calibration"))

import envs.joint_hv_env as hv  # noqa: E402

OUT_DIR = os.path.join(HERE, "eval", "perf")
FIELDS = ["family", "algo", "seed", "workload", "sla_pct", "lat_mean_ms", "lat_p95_ms",
          "lat_p99_ms", "served_pct", "lost_per_1000", "intervals"]

LOG = []
_step = hv.JointHVEnv.step


def _logged_step(self, action):
    out = _step(self, action)
    m = out[-1]["metrics"]
    LOG.append((self.workload, m["latency"], m["success"], m["load_frac"]))
    return out


hv.JointHVEnv.step = _logged_step


def summarize_log(sla_ms: float) -> list[dict]:
    rows = []
    for wl in ["cpu-bound", "memory-bound", "io-bound", "mixed"]:
        sel = np.array([r[1:] for r in LOG if r[0] == wl])
        if not len(sel):
            continue
        lat, suc, load = sel[:, 0], sel[:, 1], sel[:, 2]
        served = float((load * suc).sum() / max(load.sum(), 1e-12))
        rows.append(dict(workload=wl,
                         sla_pct=round(100 * float(np.mean((lat < sla_ms) & (suc >= 0.99))), 1),
                         lat_mean_ms=round(float(lat.mean()), 1),
                         lat_p95_ms=round(float(np.percentile(lat, 95)), 1),
                         lat_p99_ms=round(float(np.percentile(lat, 99)), 1),
                         served_pct=round(100 * served, 3),
                         lost_per_1000=round(1000 * (1 - served), 2),
                         intervals=len(lat)))
    return rows


def run(family: str) -> None:
    hv.set_trace_family(family, "bb" if family == "bitbrains" else "azure")
    import knee_sensitivity as ks
    import train
    import eval_baselines as eb

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{family}.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["algo"], int(r["seed"])) for r in csv.DictReader(f)}
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    def emit(algo, seed, fn):
        LOG.clear()
        fn()
        rows = summarize_log(hv.SLA_MEAN_LATENCY_MS)
        for r in rows:
            w.writerow(dict(family=family, algo=algo, seed=seed, **r))
        f.flush()
        print(f"[{family}] {algo} seed{seed}: " + "  ".join(
            f"{r['workload']} slo {r['sla_pct']} p95 {r['lat_p95_ms']} served {r['served_pct']}"
            for r in rows), flush=True)

    def rl(algo, seed, tag, mode="discrete"):
        name = f"{algo}_all{('_' + tag) if tag else ''}"
        env = train.make_env("all", seed, action_mode=mode)
        model = train.load_model(algo, os.path.join(HERE, "models", name), env, seed=seed)
        return lambda: train.evaluate(model, seed, action_mode=mode)

    for algo, seed, tag in ks.rl_jobs(family):
        if (algo, seed) not in done:
            mode = "continuous" if algo.startswith("pandey") else "discrete"
            emit(algo, seed, rl(algo, seed, tag, mode))

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
    for name in ks.BASELINES:
        for seed in ks.SEEDS7:
            if (name, seed) in done:
                continue
            c = ctrl[name]()
            if c is None:
                break
            emit(name, seed, lambda: eb.evaluate_policy(c, seed))
    f.close()


def summarize() -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.csv"))):
        if not p.endswith("summary.csv"):
            with open(p) as f:
                rows += list(csv.DictReader(f))
    keys = sorted({(r["family"], r["algo"], r["workload"]) for r in rows})
    metrics = ["sla_pct", "lat_mean_ms", "lat_p95_ms", "lat_p99_ms", "served_pct", "lost_per_1000"]
    out = []
    for fam, algo, wl in keys:
        sel = [r for r in rows if (r["family"], r["algo"], r["workload"]) == (fam, algo, wl)]
        rec = dict(family=fam, algo=algo, workload=wl, seeds=len(sel))
        for m in metrics:
            v = np.array([float(r[m]) for r in sel])
            rec[f"{m}_mean"] = round(float(v.mean()), 3)
            rec[f"{m}_std"] = round(float(v.std(ddof=1)), 3) if len(v) > 1 else 0.0
        out.append(rec)
    path = os.path.join(OUT_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["azure", "bitbrains"])
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.family)
