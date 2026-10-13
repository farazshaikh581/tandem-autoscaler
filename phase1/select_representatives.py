"""Pick one representative VM per workload class and extract its demand trace.

For each class (cpu-bound / memory-bound / io-bound / mixed) we take the VM whose
dominant-axis intensity is the median of that class (a "typical" member), and dump
its time series as a demand trace the env/simulator replays:
  cpu_norm  in [0,1]  -> request-rate / CPU-demand driver
  mem_MB              -> per-pod memory working set driver (the memory-bound axis)
  io                  -> disk+net throughput (context)
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "fastStorage", "2013-8")
CLASSES = ["cpu-bound", "memory-bound", "io-bound", "mixed"]
DOMINANT = {"cpu-bound": "cpu_int", "memory-bound": "mem_int",
            "io-bound": "io_int", "mixed": "cpu_int"}
C_CPU_PCT, C_MEM_PROV, C_MEM_USE, C_DISK_R, C_DISK_W, C_NET_R, C_NET_T = 4, 5, 6, 7, 8, 9, 10


def extract(vm: str) -> dict:
    df = pd.read_csv(os.path.join(DATA, f"{vm}.csv"), sep=";", header=0, engine="python")
    g = lambda i: pd.to_numeric(df.iloc[:, i].astype(str).str.strip(), errors="coerce").fillna(0.0).to_numpy()
    cpu_norm = np.clip(g(C_CPU_PCT) / 100.0, 0, 1.5)
    mem_MB = g(C_MEM_USE) / 1024.0
    io = g(C_DISK_R) + g(C_DISK_W) + g(C_NET_R) + g(C_NET_T)
    return dict(cpu_norm=cpu_norm, mem_MB=mem_MB, io=io)


def main() -> None:
    cls = pd.read_csv(os.path.join(HERE, "workload_classes.csv"))
    os.makedirs(os.path.join(HERE, "traces"), exist_ok=True)
    reps = []
    for c in CLASSES:
        sub = cls[cls.bound == c]
        if sub.empty:
            print(f"[skip] no VMs in class {c}")
            continue
        axis = DOMINANT[c]
        med = sub[axis].median()
        rep = sub.iloc[(sub[axis] - med).abs().argmin()]
        tr = extract(rep.vm)
        np.savez(os.path.join(HERE, "traces", f"{c}.npz"), **tr)
        reps.append(dict(bound=c, vm=rep.vm, n_samples=len(tr["cpu_norm"]),
                         cpu_p95=round(rep.cpu_p95, 1), mem_util_p95=round(rep.mem_util_p95, 1),
                         io_p95=round(rep.io_p95, 1), mem_prov_mb=round(rep.mem_prov_mb, 0)))
        print(f"{c:13s} -> VM {rep.vm}: {len(tr['cpu_norm'])} samples, "
              f"cpu_p95={rep.cpu_p95:.0f}% mem_util_p95={rep.mem_util_p95:.0f}% "
              f"mem_MB[min/med/max]={tr['mem_MB'].min():.0f}/{np.median(tr['mem_MB']):.0f}/{tr['mem_MB'].max():.0f}")
    pd.DataFrame(reps).to_csv(os.path.join(HERE, "representatives.csv"), index=False)
    print(f"\nwrote representatives.csv + traces/*.npz")


if __name__ == "__main__":
    main()
