"""Classify Bitbrains GWA-T-12 VMs into resource-bound workload classes.

Goal for Phase 1: confirm a real MEMORY-BOUND cohort exists (the differentiator
that HPA-only cannot fix), plus CPU-bound / IO-bound / mixed classes to drive the
autoscaling experiments.

Per VM we compute p95 intensity on three axes:
  cpu_int = p95 CPU usage %                     (real utilization, 0-100)
  mem_int = p95 (mem_usage / mem_provisioned)%  (real utilization, 0-100)
  io_int  = 100 * p95(disk+net throughput) / population_p95   (relative, 0-100+)

Class = dominant axis if it is meaningful (>=20) and beats the runner-up by >=1.2x;
low activity -> "idle"; no clear winner -> "mixed".

Attribution: data (c) Bitbrains IT Services Inc.; cite the GWA-T-12 / CCGrid 2015 paper.
"""
import glob
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data", "fastStorage", "2013-8")
OUT_CSV = os.path.join(os.path.dirname(__file__), "workload_classes.csv")
OUT_PNG = os.path.join(os.path.dirname(__file__), "classification_summary.png")

# positional columns (semicolon-separated, values may have leading tabs)
C_CPU_PCT, C_MEM_PROV, C_MEM_USE = 4, 5, 6
C_DISK_R, C_DISK_W, C_NET_R, C_NET_T = 7, 8, 9, 10


def vm_stats(path: str) -> dict | None:
    try:
        df = pd.read_csv(path, sep=";", header=0, engine="python")
    except Exception:
        return None
    if df.shape[1] < 11 or len(df) < 10:
        return None
    g = lambda i: pd.to_numeric(df.iloc[:, i].astype(str).str.strip(), errors="coerce")
    cpu = g(C_CPU_PCT)
    mem_prov, mem_use = g(C_MEM_PROV), g(C_MEM_USE)
    mem_util = np.where(mem_prov > 0, mem_use / mem_prov * 100.0, np.nan)
    io = (g(C_DISK_R) + g(C_DISK_W) + g(C_NET_R) + g(C_NET_T))
    return dict(
        vm=os.path.splitext(os.path.basename(path))[0],
        cpu_p95=float(np.nanpercentile(cpu, 95)),
        cpu_mean=float(np.nanmean(cpu)),
        mem_util_p95=float(np.nanpercentile(mem_util, 95)) if np.isfinite(mem_util).any() else 0.0,
        mem_prov_mb=float(np.nanmean(mem_prov)) / 1024.0,
        io_p95=float(np.nanpercentile(io, 95)),
    )


def main() -> None:
    files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    rows = [s for s in (vm_stats(f) for f in files) if s]
    df = pd.DataFrame(rows).fillna(0.0)
    print(f"parsed {len(df)}/{len(files)} VMs")

    io_ref = np.percentile(df["io_p95"], 95) or 1.0
    df["io_int"] = (100.0 * df["io_p95"] / io_ref).clip(0, 200)
    df["cpu_int"], df["mem_int"] = df["cpu_p95"], df["mem_util_p95"]

    def classify(r):
        axes = {"cpu-bound": r.cpu_int, "memory-bound": r.mem_int, "io-bound": r.io_int}
        top, second = sorted(axes.values(), reverse=True)[:2]
        if top < 20:
            return "idle"
        if second > 0 and top / max(second, 1e-9) < 1.2:
            return "mixed"
        return max(axes, key=axes.get)

    df["bound"] = df.apply(classify, axis=1)

    print("\n== population p95 intensity quantiles ==")
    for c in ["cpu_int", "mem_int", "io_int"]:
        q = np.percentile(df[c], [50, 75, 90, 99])
        print(f"  {c:10s} p50={q[0]:6.1f} p75={q[1]:6.1f} p90={q[2]:6.1f} p99={q[3]:6.1f}")

    print("\n== class counts ==")
    print(df["bound"].value_counts().to_string())

    mem = df[df.bound == "memory-bound"]
    print(f"\nMEMORY-BOUND cohort: {len(mem)} VMs; "
          f"mem_util_p95 range {mem.mem_util_p95.min():.0f}-{mem.mem_util_p95.max():.0f}%, "
          f"median {mem.mem_util_p95.median():.0f}%, median prov {mem.mem_prov_mb.median():.0f} MB"
          if len(mem) else "\nMEMORY-BOUND cohort: NONE")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    # summary figure: class counts + cpu-vs-mem scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    vc = df["bound"].value_counts()
    a1.bar(vc.index, vc.values, color="steelblue"); a1.set_title("VM count per class")
    a1.tick_params(axis="x", rotation=20)
    colors = {"cpu-bound": "tab:red", "memory-bound": "tab:green", "io-bound": "tab:orange",
              "mixed": "gray", "idle": "lightgray"}
    for b, sub in df.groupby("bound"):
        a2.scatter(sub.cpu_int, sub.mem_int, s=10, alpha=0.5, label=b, color=colors.get(b))
    a2.set_xlabel("CPU intensity (p95 %)"); a2.set_ylabel("Memory intensity (p95 util %)")
    a2.set_title("CPU vs Memory intensity"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
