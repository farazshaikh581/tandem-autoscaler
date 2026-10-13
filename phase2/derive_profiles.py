"""Derive a per-function service profile from measured Azure production data (#22).

Phase 1 gives every function the same service physics: one base latency (L0_MS
5.26), one saturation plateau (LAT_SAT_MS 403), one SLO (20ms), one working-set
size. Functions differ only in how much CPU/memory their load demands. That is
enough to call them different *workloads*, but not really different *functions* --
they are four parameterizations of a single service model.

This grounds the difference in measurement instead. Each workload class is already
driven by the invocation trace of a specific real Azure function (REP_HASHES in
phase1/rebuild_traces_azure.py). The same dataset also publishes, for that same
function, its measured execution-duration distribution, and for its owning app,
its measured allocated memory. So the service profile can be read off the real
function rather than invented:

    L0_ms        <- that function's measured p50 execution duration
    sla_ms       <- 3.8 x L0_ms
    lat_sat_ms   <- 76.6 x L0_ms
    mem_need_mi  <- its app's measured AverageAllocatedMb

The two multipliers are the ratios the k3s calibration already implies
(20 / 5.26 = 3.80 and 403 / 5.26 = 76.6). The split of sources is deliberate:
Azure measured what the *application* does (how long a call takes, how much it
allocates), the testbed sweep measured what a *saturated container* does (the
shape of the degradation curve, fitted at 0.8% latency MAPE). Each is used only
for the thing it actually observed, and at the Phase 1 values the whole mapping
reduces to the Phase 1 constants exactly.

One data gap: the app behind the memory-bound representative has no entry in the
memory files, so that class is reassigned to a candidate function that appears in
both the duration and the memory tables, chosen as the unused persistent candidate
whose day-1 burstiness (cv) is closest to the original -- preserving the traffic
character the class was selected for. Only cpu_norm is rewritten, exactly as
rebuild_traces_azure.py does; the Bitbrains mem_MB/io series that make the class
memory-bound are untouched.

    python derive_profiles.py            # derive + write fn_profiles.json
    python derive_profiles.py --rebuild  # also rebuild the memory-bound trace
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "phase1")
AZURE = os.path.join(P1, "data", "azure_functions_2019")
sys.path.insert(0, P1)

# Phase 1 calibrated constants -- the reference point the mapping reduces to.
L0_REF, SAT_REF, SLA_REF = 5.26, 403.0, 20.0
SLA_MULT = SLA_REF / L0_REF    # 3.80
SAT_MULT = SAT_REF / L0_REF    # 76.6
MINUTE_COLS = [str(i) for i in range(1, 1441)]
N_DAYS = 14

REP_HASHES = {
    "cpu-bound":    "b8b262f00cdff1708074af1370111644cc6d19a3b36e6495fe8c3e57fb32c86f",
    "memory-bound": "5e1ccf13b121cc9f63787e3a1bce01e2915935eae5274182200d3a0bdb243c05",
    "io-bound":     "538e03911ef39a51c19377b939e9d6122c08984f5230060ba09888eedec451c9",
    "mixed":        "2ed599a184ce05d91ca7c3516876d9d10fb6c041fbcc1287aca3aa772c031d30",
}


def load_durations() -> pd.DataFrame:
    """Per-HashFunction duration percentiles, averaged over the 14 daily files."""
    fs = sorted(glob.glob(os.path.join(AZURE, "function_durations_percentiles.anon.d*.csv")))
    if not fs:
        raise SystemExit("function_durations_percentiles.*.csv missing from " + AZURE)
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby("HashFunction").agg(
        HashApp=("HashApp", "first"), days=("Average", "size"),
        avg_ms=("Average", "mean"), p50=("percentile_Average_50", "mean"),
        p99=("percentile_Average_99", "mean"), count=("Count", "sum"))


def load_memory() -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(AZURE, "app_memory_percentiles.anon.d*.csv")))
    if not fs:
        raise SystemExit("app_memory_percentiles.*.csv missing from " + AZURE)
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby("HashApp").agg(days=("AverageAllocatedMb", "size"),
                                    avg_mb=("AverageAllocatedMb", "mean"),
                                    p99_mb=("AverageAllocatedMb_pct99", "mean"))


def taken_hashes() -> set[str]:
    """Every function already assigned to a representative, train or held-out
    trace, so a replacement never collides with the generalization splits."""
    used = set(REP_HASHES.values())
    a = os.path.join(P1, "azure_trace_assignment.csv")
    if os.path.exists(a):
        used |= set(pd.read_csv(a)["hash_prefix"].astype(str))
    return used


def pick_replacement(cls: str, dur: pd.DataFrame, mem: pd.DataFrame) -> dict:
    """Unused persistent candidate with both duration and memory data, closest in
    day-1 burstiness to the function it replaces."""
    cand = pd.read_csv(os.path.join(AZURE, "persistent_candidates.csv"),
                       usecols=["HashApp", "HashFunction", "Trigger", "cv_d01"])
    used = taken_hashes()
    cand = cand[~cand.HashFunction.isin(used)]
    cand = cand[~cand.HashFunction.str[:12].isin(used)]
    cand = cand[cand.HashFunction.isin(dur.index) & cand.HashApp.isin(mem.index)]
    if cand.empty:
        raise SystemExit("no candidate has both duration and memory data")
    orig_cv = pd.read_csv(os.path.join(AZURE, "persistent_candidates.csv"),
                          usecols=["HashFunction", "cv_d01"])
    row = orig_cv[orig_cv.HashFunction == REP_HASHES[cls]]
    target = float(row.cv_d01.iloc[0]) if len(row) else float(cand.cv_d01.median())
    best = cand.iloc[(cand.cv_d01 - target).abs().argsort().iloc[0]]
    return dict(hash=best.HashFunction, app=best.HashApp,
                cv=float(best.cv_d01), target_cv=target)


def rebuild_trace(cls: str, fn_hash: str) -> str:
    """Rewrite only cpu_norm for `cls` from this function's invocation counts,
    into phase2/traces/. Phase 1's own trace files are never touched, so every
    committed Phase 1 result stays reproducible."""
    src = os.path.join(P1, "traces", f"{cls}.npz")
    base = dict(np.load(src))
    n = len(base["cpu_norm"])
    chunks = []
    for d in range(1, N_DAYS + 1):
        df = pd.read_csv(os.path.join(AZURE, f"invocations_per_function_md.anon.d{d:02d}.csv"),
                         usecols=["HashFunction"] + MINUTE_COLS).set_index("HashFunction")
        if fn_hash not in df.index:
            raise SystemExit(f"{fn_hash[:10]} missing on day {d}")
        r = df.loc[fn_hash]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        chunks.append(r[MINUTE_COLS].to_numpy(dtype=np.float64))
        print(f"    day {d}/{N_DAYS}")
    series = np.concatenate(chunks)
    if len(series) < n:
        raise SystemExit(f"only {len(series)} samples, need {n}")
    base["cpu_norm"] = series[:n].astype(np.float32)
    out_dir = os.path.join(HERE, "traces")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{cls}.npz")
    np.savez_compressed(out, **base)
    nz = 100.0 * float(np.mean(base["cpu_norm"] > 0))
    print(f"    wrote {out}  ({n} samples, {nz:.1f}% non-zero)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the reassigned class's cpu_norm trace into phase2/traces/")
    args = ap.parse_args()

    dur, mem = load_durations(), load_memory()
    profiles, notes = {}, {}

    for cls, h in REP_HASHES.items():
        if h not in dur.index:
            raise SystemExit(f"{cls}: hash not in duration data")
        row = dur.loc[h]
        app = row.HashApp
        if app in mem.index:
            profiles[cls] = dict(hash=h, app=app, p50_ms=float(row.p50),
                                 mem_mb=float(mem.loc[app].avg_mb), reassigned=False)
        else:
            rep = pick_replacement(cls, dur, mem)
            r2 = dur.loc[rep["hash"]]
            profiles[cls] = dict(hash=rep["hash"], app=rep["app"], p50_ms=float(r2.p50),
                                 mem_mb=float(mem.loc[rep["app"]].avg_mb), reassigned=True)
            notes[cls] = (f"original app has no memory record; reassigned to "
                          f"{rep['hash'][:12]} (cv {rep['cv']:.3f} vs {rep['target_cv']:.3f})")
            print(f"[{cls}] {notes[cls]}")
            if args.rebuild:
                rebuild_trace(cls, rep["hash"])

    out = {"reference": dict(L0_ms=L0_REF, sla_ms=SLA_REF, lat_sat_ms=SAT_REF,
                             sla_mult=round(SLA_MULT, 4), sat_mult=round(SAT_MULT, 4)),
           "notes": notes, "profiles": {}}
    print(f"\n{'class':<14}{'L0_ms':>9}{'sla_ms':>9}{'lat_sat_ms':>12}{'mem_mi':>9}")
    for cls, p in profiles.items():
        l0 = round(p["p50_ms"], 2)
        prof = dict(l0_ms=l0, sla_ms=round(l0 * SLA_MULT, 2),
                    lat_sat_ms=round(l0 * SAT_MULT, 1),
                    mem_need_mi=round(p["mem_mb"], 1),
                    azure_function=p["hash"], azure_app=p["app"],
                    reassigned=p["reassigned"])
        out["profiles"][cls] = prof
        print(f"{cls:<14}{prof['l0_ms']:>9}{prof['sla_ms']:>9}"
              f"{prof['lat_sat_ms']:>12}{prof['mem_need_mi']:>9}")
    path = os.path.join(HERE, "fn_profiles.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
