"""Rebuild every workload trace used in the paper from the two public datasets.

The traces are not redistributed in this repository. This script recreates them,
array for array, from the official downloads:

  Bitbrains GWA-T-12 fastStorage   unpacked into phase1/data/fastStorage/2013-8/
  Azure Functions Trace 2019       unpacked into phase1/data/azure_functions_2019/
                                   (invocations_per_function_md.anon.d01..d14.csv)

It writes, for both trace families,
  traces{,_bitbrains}/<class>.npz                  representative machine per class
  traces_heldout{,_bitbrains}/<class>__<vm>.npz    held-out machines (Section 6.5)
  traces_train{,_bitbrains}/<class>__<vm>.npz      training pool (Section 6.5)

Every file holds cpu_norm (load), mem_MB (memory usage) and io. The memory and IO
series come from the Bitbrains machine in both families. cpu_norm is the machine's
CPU usage in the Bitbrains family, and in the Azure family the per-minute invocation
count of the function recorded in azure_trace_assignment.csv (the representatives use
the functions in rebuild_traces_azure.REP_HASHES; the 199-sample mixed representative
uses hourly counts). Machines come from representatives.csv, heldout_manifest.csv and
train_pool_manifest.csv.

Afterwards, run `python3 ../phase2/derive_profiles.py --rebuild` for the trace of
Section 6.7.

Usage (from phase1/):  python3 build_traces.py
"""
import glob
import os

import numpy as np
import pandas as pd

from rebuild_traces_azure import AZURE_DIR, REP_HASHES, extract_series
from select_representatives import DATA as BITBRAINS_DIR, extract

HERE = os.path.dirname(os.path.abspath(__file__))


def bitbrains_jobs() -> list[tuple[str, str]]:
    """(relative path without family suffix, Bitbrains machine id) for every trace."""
    jobs = [(f"traces/{r.bound}.npz", str(r.vm))
            for r in pd.read_csv(os.path.join(HERE, "representatives.csv")).itertuples()]
    for kind, manifest in (("traces_heldout", "heldout_manifest.csv"),
                           ("traces_train", "train_pool_manifest.csv")):
        for r in pd.read_csv(os.path.join(HERE, manifest)).itertuples():
            jobs.append((f"{kind}/{r.bound}__{r.vm}.npz", str(r.vm)))
    return jobs


def family_path(rel: str, family: str) -> str:
    kind, name = rel.split("/")
    return os.path.join(HERE, kind if family == "azure" else f"{kind}_bitbrains", name)


def main() -> None:
    if not os.path.isdir(BITBRAINS_DIR):
        raise SystemExit(f"missing {BITBRAINS_DIR}: download and unpack GWA-T-12 fastStorage")
    if not glob.glob(os.path.join(AZURE_DIR, "invocations_per_function_md.anon.d*.csv")):
        raise SystemExit(f"missing {AZURE_DIR}: download and unpack the Azure Functions 2019 trace")

    jobs = bitbrains_jobs()
    base = {rel: extract(vm) for rel, vm in jobs}
    print(f"extracted {len(base)} Bitbrains machines")

    # Azure load series: representatives by full hash, the rest by the recorded prefix
    asg = pd.read_csv(os.path.join(HERE, "azure_trace_assignment.csv"))
    first_day = pd.read_csv(os.path.join(AZURE_DIR, "invocations_per_function_md.anon.d01.csv"),
                            usecols=["HashFunction"]).HashFunction
    by_prefix = {}
    for h in first_day:
        if h[:12] in set(asg.hash_prefix):
            by_prefix.setdefault(h[:12], set()).add(h)
    if any(len(v) != 1 for v in by_prefix.values()) or len(by_prefix) != asg.hash_prefix.nunique():
        raise SystemExit("could not resolve every function prefix to exactly one function")
    file_hash = {r.file: next(iter(by_prefix[r.hash_prefix])) for r in asg.itertuples()}
    series = extract_series(sorted(set(file_hash.values()) | set(REP_HASHES.values())))

    for rel, _ in jobs:
        d = base[rel]
        n = len(d["mem_MB"])
        if rel.startswith("traces/"):
            s = series[REP_HASHES[os.path.basename(rel)[:-4]]]
            if rel == "traces/mixed.npz":
                s = s.reshape(-1, 60).sum(axis=1)          # hourly counts
        else:
            s = series[file_hash[rel]]
        azure_cpu = (np.resize(s, n) if n > len(s) else s[:n]).astype(np.float64)
        for family, cpu in (("bitbrains", d["cpu_norm"]), ("azure", azure_cpu)):
            out = family_path(rel, family)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, cpu_norm=np.asarray(cpu, dtype=np.float64),
                     mem_MB=np.asarray(d["mem_MB"], dtype=np.float64),
                     io=np.asarray(d["io"], dtype=np.float64))
    print(f"wrote {2 * len(jobs)} trace files")


if __name__ == "__main__":
    main()
