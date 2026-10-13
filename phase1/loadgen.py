"""High-throughput load generator for the faithful sim-vs-real comparison (#4).

The Phase 0 driver (phase0/load.py) is a single threaded urllib process; it tops
out near ~260 rps because one Python process is the bottleneck. To reach the
sim's operating point (load_frac = 1 ~ 1200 rps = CPU_DEMAND_M / M_PER_RPS) this
fans that same, already-calibrated per-request loop across multiple processes and
aggregates their latency samples. Same measurement, more cores.

Drop-in for phase0.load.run_load: returns the list of (latency_ms, ok) samples,
so RealClusterBackend can swap driver with no other change.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

_PHASE0 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase0"))
if _PHASE0 not in sys.path:
    sys.path.insert(0, _PHASE0)
from load import run_load as _run_single, summarize  # noqa: E402,F401  (summarize re-exported)

RPS_PER_PROC = 250          # each worker process drives ~this many rps comfortably


def _worker(url, rps, duration, workers, q):
    q.put(_run_single(url, rps, duration, workers=workers, quiet=True))


def run_load(url: str, rps: int, duration: int, procs: int | None = None,
             workers_per_proc: int = 128, quiet: bool = True):
    """Drive `rps` total across `procs` processes; return aggregated samples.
    Single process (procs==1) delegates straight to the Phase 0 driver."""
    if procs is None:
        procs = max(1, min(os.cpu_count() or 1, -(-rps // RPS_PER_PROC)))
    if procs == 1:
        return _run_single(url, rps, duration, workers=workers_per_proc, quiet=quiet)

    shares = [rps // procs] * procs
    for i in range(rps % procs):
        shares[i] += 1
    q: mp.Queue = mp.Queue()
    ps = [mp.Process(target=_worker, args=(url, r, duration, workers_per_proc, q))
          for r in shares if r > 0]
    for p in ps:
        p.start()
    samples: list[tuple[float, bool]] = []
    for _ in ps:              # drain before join so the pipes don't fill and deadlock
        samples += q.get()
    for p in ps:
        p.join()
    if not quiet:
        s = summarize(samples)
        print(f"[loadgen] {procs} procs, target {rps} rps: achieved "
              f"{s['n'] / duration:.0f} rps, success {s['success']:.3f}, "
              f"mean {s['mean_ms']:.1f}ms p90 {s['p90_ms']:.1f}ms")
    return samples


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--rps", type=int, required=True)
    ap.add_argument("--duration", type=int, default=8)
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    run_load(args.url, args.rps, args.duration, procs=args.procs, quiet=False)
