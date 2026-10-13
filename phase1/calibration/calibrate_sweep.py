"""Calibration sweep (issue #1): measure latency vs CPU-saturation on the real cluster.

Pins the system to a FIXED configuration (no HPA, replicas=2, requests=limits so
the CFS limit that throttles equals the sim's notion of capacity), then steps the
offered load through a list of rps levels. At each level it discards a warmup
window, then records request latency stats (from load.py) and per-pod CPU usage
(metrics-server via kubectl top) over the measurement window.

Output: sweep_<ts>.csv with one row per level:
  rps, achieved_rps, success, mean/p50/p90/p95/p99 latency, usage_avg_m per pod,
  total_usage_m, rho (= total usage / total limit)

The fit (fit_constants.py) turns these rows into L0_MS + latency-curve shape and
mcores-per-rps for the sim.

Usage (from phase1/calibration/):
  python3 calibrate_sweep.py --levels 5,10,20,40,60,80,100,120 --measure 90 --warmup 30
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "phase0"))
from load import run_load, summarize  # noqa: E402

NS = "hv-phase0"


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def service_url() -> str:
    ip = sh(["kubectl", "get", "svc", "web-service", "-n", NS,
             "-o", "jsonpath={.spec.clusterIP}"]).strip()
    return f"http://{ip}:80/"


def pod_usage_m() -> tuple[int, list[int]]:
    """Total and per-pod CPU usage in millicores from metrics-server."""
    out = sh(["kubectl", "top", "pods", "-n", NS, "--no-headers"])
    per = []
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].endswith("m"):
            per.append(int(parts[1][:-1]))
    return sum(per), per


def total_limit_m() -> int:
    out = sh(["kubectl", "get", "pods", "-n", NS, "-o",
              "jsonpath={range .items[*]}{.spec.containers[0].resources.limits.cpu}{\"\\n\"}{end}"])
    tot = 0
    for line in out.strip().splitlines():
        tot += int(line[:-1]) if line.endswith("m") else int(float(line) * 1000)
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="5,10,20,40,60,80,100,120")
    ap.add_argument("--warmup", type=int, default=30, help="seconds discarded per level")
    ap.add_argument("--measure", type=int, default=90, help="measured seconds per level")
    ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=0,
                    help="drive load with phase1/loadgen.py across this many processes (needed above "
                         "~250 rps, where the single-process phase0 driver is the bottleneck); 0 = phase0 driver")
    args = ap.parse_args()
    global run_load
    if args.procs:
        sys.path.insert(0, os.path.join(HERE, ".."))
        from loadgen import run_load as _mp_run_load  # noqa: E402
        run_load = lambda url, rps, dur, quiet=True: _mp_run_load(url, rps, dur, procs=args.procs, quiet=True)  # noqa: E731
    levels = [int(x) for x in args.levels.split(",")]

    url = service_url()
    lim_m = total_limit_m()
    ts = time.strftime("%Y-%m-%d_%H%M")
    out_path = args.out or os.path.join(HERE, f"sweep_{ts}.csv")
    print(f"[sweep] url={url} total_limit={lim_m}m levels={levels} "
          f"({args.warmup}s warmup + {args.measure}s measure each)")

    rows = []
    for rps in levels:
        # warmup: drive load but discard samples
        run_load(url, rps, args.warmup, quiet=True)

        # measurement: latency samples + concurrent kubectl-top sampling
        usage_samples: list[int] = []
        per_pod_last: list[int] = []
        stop = threading.Event()

        def poll_top():
            while not stop.is_set():
                try:
                    tot, per = pod_usage_m()
                    usage_samples.append(tot)
                    per_pod_last[:] = per
                except subprocess.CalledProcessError:
                    pass
                stop.wait(15)

        poller = threading.Thread(target=poll_top, daemon=True)
        poller.start()
        samples = run_load(url, rps, args.measure, quiet=True)
        stop.set()
        poller.join(timeout=20)

        s = summarize(samples)
        # metrics-server lags ~15-30s; the level's steady state is best reflected
        # by the later samples, so average the second half of the window.
        half = usage_samples[len(usage_samples) // 2:] or usage_samples
        usage_m = sum(half) / len(half) if half else float("nan")
        rho = usage_m / lim_m if lim_m else float("nan")
        row = dict(rps=rps, achieved_rps=round(s["n"] / args.measure, 1),
                   success=round(s["success"], 4), mean_ms=round(s["mean_ms"], 2),
                   p50_ms=round(s["p50_ms"], 2), p90_ms=round(s["p90_ms"], 2),
                   p95_ms=round(s["p95_ms"], 2), p99_ms=round(s["p99_ms"], 2),
                   usage_total_m=round(usage_m, 1), limit_total_m=lim_m,
                   rho=round(rho, 4), per_pod_m=json.dumps(per_pod_last))
        rows.append(row)
        print(f"[sweep] rps={rps:4d} achieved={row['achieved_rps']:6.1f} "
              f"success={row['success']:.3f} mean={row['mean_ms']:7.1f}ms "
              f"p99={row['p99_ms']:8.1f}ms usage={row['usage_total_m']:6.1f}m rho={row['rho']:.2f}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
