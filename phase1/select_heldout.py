"""Build a genuinely held-out eval set: unseen VMs, same class.

The trained models only ever saw ONE representative VM per class (the class-median
member picked by select_representatives.py). Sampling different start windows from
that same trace at eval time (train.py: seed+777) is a different *window* but the
same data-generating VM -> not a true held-out test.

Here we carve a proper train/test split at the VM level: for each class pick K
*distinct, unseen* VMs spanning the class's dominant-axis intensity distribution
(excluding the training rep), and dump each one's trace. Evaluating the trained
model on these (eval_heldout.py) measures generalization to workloads it never saw.

Output: traces_heldout/<class>__<vm>.npz  +  heldout_manifest.csv
"""
import os

import numpy as np
import pandas as pd

from select_representatives import CLASSES, DOMINANT, extract

HERE = os.path.dirname(__file__)
K = 4                      # held-out VMs per class
MIN_SAMPLES = 202          # need > episode_len (200) + 1 for a valid start window
QUANTILES = np.linspace(0.1, 0.9, 9)  # candidate spread across the class intensity range


def main() -> None:
    cls = pd.read_csv(os.path.join(HERE, "workload_classes.csv"))
    reps = pd.read_csv(os.path.join(HERE, "representatives.csv"))
    train_vm = dict(zip(reps.bound, reps.vm))  # exclude these (model trained on them)
    out_dir = os.path.join(HERE, "traces_heldout")
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for c in CLASSES:
        axis = DOMINANT[c]
        sub = cls[(cls.bound == c) & (cls.vm != train_vm.get(c))]
        if sub.empty:
            print(f"[skip] no unseen VMs in class {c}")
            continue
        # candidate VMs nearest each intensity quantile (dedup, preserve spread)
        targets = sub[axis].quantile(QUANTILES).to_numpy()
        cand = []
        for t in targets:
            vm = sub.iloc[(sub[axis] - t).abs().argmin()].vm
            if vm not in cand:
                cand.append(vm)
        # extract in spread order; keep the first K with a long-enough trace
        kept = 0
        for vm in cand:
            if kept >= K:
                break
            tr = extract(vm)
            n = len(tr["cpu_norm"])
            if n < MIN_SAMPLES:
                continue
            np.savez(os.path.join(out_dir, f"{c}__{vm}.npz"), **tr)
            row = cls[cls.vm == vm].iloc[0]
            manifest.append(dict(bound=c, vm=int(vm), n_samples=n,
                                 cpu_p95=round(float(row.cpu_p95), 1),
                                 mem_util_p95=round(float(row.mem_util_p95), 1),
                                 io_p95=round(float(row.io_p95), 1)))
            kept += 1
        print(f"{c:13s} -> {kept} held-out VMs (train rep {train_vm.get(c)} excluded), "
              f"pool of {len(sub)} candidates")

    pd.DataFrame(manifest).to_csv(os.path.join(HERE, "heldout_manifest.csv"), index=False)
    print(f"\nwrote heldout_manifest.csv + traces_heldout/*.npz ({len(manifest)} VMs)")


if __name__ == "__main__":
    main()
