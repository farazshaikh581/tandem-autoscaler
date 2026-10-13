"""Replace the cpu_norm (request-rate/CPU-demand) driver in every trace file with
genuine organic traffic instead of a repurposed Bitbrains VM CPU-utilization series.

Background: cpu_norm's own docstring in select_representatives.py already says
what it's FOR -- "request-rate / CPU-demand driver" -- but it has always been
sourced from Bitbrains VM CPU% (a resource-utilization proxy, not real requests).
This script swaps that source for the Azure Functions Trace 2019
(https://github.com/Azure/AzurePublicDataset), whose invocation-count-per-minute
files are a genuine, organic request-rate signal. mem_MB/io are left untouched
(the memory axis is a separate, later realism item) -- this only ever rewrites
cpu_norm, so the trace file format and every downstream env/train script are
unaffected.

Each output file's cpu_norm is truncated to that file's OWN existing length, so
SimBackend's shared modulo-cycle indexing between cpu_norm and mem_MB stays
aligned without touching envs/joint_hv_env.py. One class (the representative
"mixed" trace, 199 samples) is short enough that a raw per-minute slice is
~97% zero by chance -- it instead uses 336 hourly-aggregated bins (14 days)
and takes the first 199 hourly samples, which are >96% non-zero. Every other
file (all >7800 samples) uses raw per-minute resolution.

Selection is disjoint across representative/held-out/train-pool splits (a
function used for one is never reused for another), preserving the
generalization-eval methodology's train/test independence. Candidates are
drawn from a pre-scanned manifest of HTTP-triggered functions that appear in
all 14 daily files with >5000 invocations/day (persistent_candidates.csv),
spread across a range of real traffic burstiness (coefficient of variation)
the same way the original VM selection spread across Bitbrains intensity
quantiles -- same spirit (diversity across the pool), different measurement
basis, since the new source is decoupled from the old VM classification.

Usage:
  python rebuild_traces_azure.py            # rebuild held-out + train-pool (54 files)
  python rebuild_traces_azure.py --verify   # structural sanity check only, no writes
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
AZURE_DIR = os.path.join(HERE, "data", "azure_functions_2019")
CANDIDATES_CSV = os.path.join(AZURE_DIR, "persistent_candidates.csv")
N_DAYS = 14
MINUTE_COLS = [str(i) for i in range(1, 1441)]

# Already assigned to the 4 representative traces (excluded from further selection).
REP_HASHES = {
    "cpu-bound":    "b8b262f00cdff1708074af1370111644cc6d19a3b36e6495fe8c3e57fb32c86f",
    "memory-bound": "5e1ccf13b121cc9f63787e3a1bce01e2915935eae5274182200d3a0bdb243c05",
    "io-bound":     "538e03911ef39a51c19377b939e9d6122c08984f5230060ba09888eedec451c9",
    "mixed":        "2ed599a184ce05d91ca7c3516876d9d10fb6c041fbcc1287aca3aa772c031d30",
}


def select_pool_hashes(n_needed: int) -> list[str]:
    """n_needed distinct HashFunctions, spread across the candidate CV range,
    excluding the ones already used for the representatives."""
    cand = pd.read_csv(CANDIDATES_CSV)
    cand = cand[~cand.HashFunction.isin(REP_HASHES.values())].sort_values("cv_d01").reset_index(drop=True)
    if len(cand) < n_needed:
        raise SystemExit(f"only {len(cand)} candidates available, need {n_needed}")
    idx = np.linspace(0, len(cand) - 1, n_needed).round().astype(int)
    idx = pd.unique(idx)  # guard against collisions from rounding
    while len(idx) < n_needed:  # top up if rounding collapsed any duplicates
        extra = [i for i in range(len(cand)) if i not in idx]
        idx = np.concatenate([idx, extra[: n_needed - len(idx)]])
    return cand.iloc[idx[:n_needed]].HashFunction.tolist()


def extract_series(hashes: list[str]) -> dict[str, np.ndarray]:
    """One pass over the 14 daily files, pulling the per-minute series for every
    hash in `hashes` at once (avoids re-reading ~140MB/day per function)."""
    series = {h: [] for h in hashes}
    for d in range(1, N_DAYS + 1):
        df = pd.read_csv(os.path.join(AZURE_DIR, f"invocations_per_function_md.anon.d{d:02d}.csv"),
                          usecols=["HashFunction"] + MINUTE_COLS).set_index("HashFunction")
        for h in hashes:
            if h not in df.index:
                raise SystemExit(f"hash {h[:10]} missing on day {d} (should be persistent)")
            row = df.loc[h]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            series[h].append(row[MINUTE_COLS].to_numpy(dtype=np.float64))
        print(f"  day {d}/{N_DAYS} extracted")
    return {h: np.concatenate(chunks) for h, chunks in series.items()}


def sanity_check(cls: str, n: int) -> None:
    """Reset/step a short no-op rollout through the real env and confirm sane,
    non-degenerate physics -- same check used before spending compute on training."""
    from envs.joint_hv_env import JointHVEnv
    env = JointHVEnv(workload=cls, episode_len=min(50, n - 1), seed=0)
    loads = []
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(13)
            loads.append(info["metrics"]["load_frac"])
            done = term or trunc
    loads = np.array(loads)
    assert not np.isnan(loads).any(), f"{cls}: NaN in load_frac"
    assert loads.std() > 0, f"{cls}: degenerate all-constant load_frac"
    print(f"  [ok] {cls}: load_frac mean={loads.mean():.3f} std={loads.std():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="sanity-check existing traces only, no writes")
    args = ap.parse_args()

    if args.verify:
        for f in sorted(glob.glob(os.path.join(HERE, "traces", "*.npz"))):
            cls = os.path.basename(f)[:-4]
            n = len(np.load(f)["cpu_norm"])
            sanity_check(cls, n)
        for f in sorted(glob.glob(os.path.join(HERE, "traces_heldout", "*.npz"))) + \
                 sorted(glob.glob(os.path.join(HERE, "traces_train", "*.npz"))):
            d = np.load(f)
            assert not np.isnan(d["cpu_norm"]).any(), f"{f}: NaN cpu_norm"
            assert d["cpu_norm"].std() > 0, f"{f}: degenerate cpu_norm"
        print("[verify] all traces structurally sane")
        return

    heldout_files = sorted(glob.glob(os.path.join(HERE, "traces_heldout", "*.npz")))
    train_files = sorted(glob.glob(os.path.join(HERE, "traces_train", "*.npz")))
    targets = heldout_files + train_files
    print(f"rebuilding {len(targets)} held-out/train-pool files "
          f"({len(heldout_files)} held-out + {len(train_files)} train-pool)")

    hashes = select_pool_hashes(len(targets))
    print("extracting per-minute series for the selected pool...")
    series = extract_series(hashes)

    assignment = []
    for f, h in zip(targets, hashes):
        old = dict(np.load(f))
        # mem_MB is the untouched, authoritative length reference -- cpu_norm is
        # exactly the field this script rewrites, so on a second/idempotent run
        # its length can no longer be trusted as the original target length.
        n = len(old["mem_MB"])
        s = series[h]
        # one file (traces_train/memory-bound__1213.npz, n=23690) exceeds the
        # 20160-sample cap of a single 14-day Azure per-minute series -- tile
        # (wrap) rather than silently truncate mem_MB's usable range to match
        # a shorter cpu_norm, which would quietly discard part of the file's
        # existing (untouched-per-scope) memory trace.
        new_cpu = np.resize(s, n).astype(np.float64) if n > len(s) else s[:n].astype(np.float64)
        np.savez(f, cpu_norm=new_cpu, mem_MB=old["mem_MB"], io=old["io"])
        assignment.append(dict(file=os.path.relpath(f, HERE), hash_prefix=h[:12],
                               n=n, mean=round(float(new_cpu.mean()), 2),
                               zero_pct=round(float(np.mean(new_cpu == 0) * 100), 1)))
        print(f"  {os.path.relpath(f, HERE):40s} n={n:6d} mean={new_cpu.mean():7.2f} "
              f"zero%={np.mean(new_cpu == 0) * 100:5.1f}")

    pd.DataFrame(assignment).to_csv(os.path.join(HERE, "azure_trace_assignment.csv"), index=False)
    print(f"\nwrote azure_trace_assignment.csv ({len(assignment)} files)")


if __name__ == "__main__":
    main()
