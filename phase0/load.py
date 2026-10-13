"""Constant-rate HTTP load driver with per-request latency capture (stdlib only).

Phase 0 wants a STEADY load that parks per-pod utilization near the HPA target,
so any oscillation we see comes from the HPA<->VPA coupling rather than from a
changing workload. Hits the service ClusterIP (reachable from the control-plane
node this runs on).

Latency capture (issue #5): every request's wall-clock latency is recorded.
--out writes one CSV row per --interval seconds (mean/P50/P90/P95/P99, success
rate, achieved rps); a summary line prints at the end either way. The module is
also importable: run_load() returns the raw (latency_ms, ok) samples so the
calibration sweep can post-process them.
"""
import argparse
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = min(int(round((p / 100.0) * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return sorted_vals[k]


def summarize(samples: list[tuple[float, bool]]) -> dict:
    lats = sorted(l for l, ok in samples if ok)
    n = len(samples)
    ok_n = len(lats)
    return {
        "n": n,
        "success": (ok_n / n) if n else float("nan"),
        "mean_ms": (sum(lats) / ok_n) if ok_n else float("nan"),
        "p50_ms": percentile(lats, 50),
        "p90_ms": percentile(lats, 90),
        "p95_ms": percentile(lats, 95),
        "p99_ms": percentile(lats, 99),
    }


def run_load(url: str, rps: int, duration: int, workers: int = 64,
             timeout: float = 5.0, out_path: str | None = None,
             interval: int = 10, quiet: bool = False):
    """Drive constant-rate load; return list of (latency_ms, ok) samples."""
    samples: list[tuple[float, bool]] = []
    lock = threading.Lock()

    def hit() -> None:
        t0 = time.monotonic()
        ok = True
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                r.read()
        except Exception:
            ok = False
        lat_ms = (time.monotonic() - t0) * 1000.0
        with lock:
            samples.append((lat_ms, ok))

    out = open(out_path, "w") if out_path else None
    if out:
        out.write("elapsed_s,n,achieved_rps,success,mean_ms,p50_ms,p90_ms,p95_ms,p99_ms\n")

    t0 = time.time()
    last_flush, last_idx = t0, 0
    gap = 1.0 / rps  # pace uniformly: a synchronized burst overflows the server's
    with ThreadPoolExecutor(max_workers=workers) as pool:  # accept backlog and
        while time.time() - t0 < duration:  # measures TCP retransmits, not queueing
            tick = time.time()
            for i in range(rps):
                pool.submit(hit)
                nxt = tick + (i + 1) * gap
                pause = nxt - time.time()
                if pause > 0:
                    time.sleep(pause)
            if out and tick - last_flush >= interval:
                with lock:
                    window = samples[last_idx:]
                    last_idx = len(samples)
                s = summarize(window)
                out.write(f"{tick - t0:.0f},{s['n']},{s['n'] / (tick - last_flush):.1f},"
                          f"{s['success']:.4f},{s['mean_ms']:.2f},{s['p50_ms']:.2f},"
                          f"{s['p90_ms']:.2f},{s['p95_ms']:.2f},{s['p99_ms']:.2f}\n")
                out.flush()
                last_flush = tick
            elapsed = time.time() - tick
            time.sleep(max(0, 1.0 - elapsed))
    if out:
        out.close()

    if not quiet:
        s = summarize(samples)
        print(f"[load] done: n={s['n']} success={s['success']:.4f} "
              f"mean={s['mean_ms']:.1f}ms p50={s['p50_ms']:.1f} p90={s['p90_ms']:.1f} "
              f"p95={s['p95_ms']:.1f} p99={s['p99_ms']:.1f}")
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Constant-rate load generator with latency capture")
    ap.add_argument("--url", required=True, help="e.g. http://10.43.x.y:80/")
    ap.add_argument("--rps", type=int, default=40)
    ap.add_argument("--duration", type=int, default=900)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=None, help="per-interval latency CSV")
    ap.add_argument("--interval", type=int, default=10, help="CSV row every N seconds")
    args = ap.parse_args()

    print(f"[load] {args.url} @ {args.rps} rps for {args.duration}s")
    run_load(args.url, args.rps, args.duration, args.workers,
             out_path=args.out, interval=args.interval)


if __name__ == "__main__":
    main()
