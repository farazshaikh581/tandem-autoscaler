"""Build a domain-randomization training pool: many unseen VMs per class.

The held-out eval (select_heldout.py / eval_heldout.py) showed the clean-on-all-knobs
stability result is a per-representative-VM artifact -- training on ONE VM per class
does not make the policy stable on other same-class VMs. Domain randomization is the
fix: train across a POOL of VMs per class so the policy sees within-class variety.

This picks N VMs per class spanning the dominant-axis intensity range, EXCLUDING
both the original training rep (representatives.csv) and the 16 held-out VMs
(heldout_manifest.csv) -- so traces_train, traces (rep), and traces_heldout are
disjoint and the held-out eval stays a clean generalization test.

Output: traces_train/<class>__<vm>.npz  +  train_pool_manifest.csv

NOTE: the mixed class has only 8 VMs total; after excluding the rep + 4 held-out only
a few remain, so its pool is unavoidably thin (documented limitation).
"""
import os

import numpy as np
import pandas as pd

from select_representatives import CLASSES, DOMINANT, extract

HERE = os.path.dirname(__file__)
N = 12                     # target training VMs per class (fewer if the class is small)
MIN_SAMPLES = 202          # need > episode_len (200) + 1 for a valid start window


def main() -> None:
    cls = pd.read_csv(os.path.join(HERE, "workload_classes.csv"))
    reps = pd.read_csv(os.path.join(HERE, "representatives.csv"))
    held = pd.read_csv(os.path.join(HERE, "heldout_manifest.csv"))
    train_vm = dict(zip(reps.bound, reps.vm))
    held_by_cls = {c: set(held[held.bound == c].vm) for c in CLASSES}
    out_dir = os.path.join(HERE, "traces_train")
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for c in CLASSES:
        axis = DOMINANT[c]
        excl = held_by_cls.get(c, set()) | {train_vm.get(c)}
        sub = cls[(cls.bound == c) & (~cls.vm.isin(excl))]
        if sub.empty:
            print(f"[skip] no training VMs left in class {c}")
            continue
        # candidates nearest N evenly spaced intensity quantiles (dedup, keep spread)
        targets = sub[axis].quantile(np.linspace(0.05, 0.95, N)).to_numpy()
        cand = []
        for t in targets:
            vm = sub.iloc[(sub[axis] - t).abs().argmin()].vm
            if vm not in cand:
                cand.append(vm)
        kept = 0
        for vm in cand:
            if kept >= N:
                break
            tr = extract(vm)
            if len(tr["cpu_norm"]) < MIN_SAMPLES:
                continue
            np.savez(os.path.join(out_dir, f"{c}__{vm}.npz"), **tr)
            row = cls[cls.vm == vm].iloc[0]
            manifest.append(dict(bound=c, vm=int(vm), n_samples=len(tr["cpu_norm"]),
                                 cpu_p95=round(float(row.cpu_p95), 1),
                                 mem_util_p95=round(float(row.mem_util_p95), 1),
                                 io_p95=round(float(row.io_p95), 1)))
            kept += 1
        print(f"{c:13s} -> {kept} training VMs (pool of {len(sub)} after excluding "
              f"rep + {len(held_by_cls.get(c, set()))} held-out)")

    pd.DataFrame(manifest).to_csv(os.path.join(HERE, "train_pool_manifest.csv"), index=False)
    print(f"\nwrote train_pool_manifest.csv + traces_train/*.npz ({len(manifest)} VMs)")


if __name__ == "__main__":
    main()
