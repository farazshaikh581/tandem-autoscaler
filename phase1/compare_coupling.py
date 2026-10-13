"""Solo vs co-located comparison for the multi-tenant coupling experiment (issue #14).

Reads the 4 real_*_spiritquest_{solo,coupled}.csv files (written by eval_real.py
run once alone and once concurrently with the other tenant) and reports the
same SLA/latency/reversal metrics used everywhere else in this project so the
coupling effect is directly comparable to every other real-cluster table.

  python compare_coupling.py
  python compare_coupling.py --family bitbrains   # *_spiritquest_bb_* -> coupling_summary_bitbrains.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from train import _reversals

HERE = os.path.dirname(__file__)
RUNS = [
    ("cpu-burn (QR-DQN, cpu-bound)", "real_qrdqn_cpu-bound_spiritquest_solo.csv",
     "real_qrdqn_cpu-bound_spiritquest_coupled.csv"),
    ("io-burn (Option B, io-bound)", "real_cvar-bellman-qrdqn_io-bound_spiritquest_solo.csv",
     "real_cvar-bellman-qrdqn_io-bound_spiritquest_coupled.csv"),
]


def summarize(path: str) -> dict:
    df = pd.read_csv(os.path.join(HERE, "eval", path))
    return dict(
        sla=100.0 * df["sla_ok"].mean(),
        lat=df["latency_ms"].mean(),
        rep_rev=_reversals(df["replicas"].tolist()),
        cpu_rev=_reversals(df["cpu_req_m"].tolist()),
        mem_rev=_reversals(df["mem_req_mi"].tolist()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["azure", "bitbrains"], default="azure")
    bb = ap.parse_args().family == "bitbrains"
    rows = []
    print(f"{'tenant':38}{'run':10}{'SLA%':>7}{'lat_ms':>8}{'rep_rev':>9}{'cpu_rev':>9}{'mem_rev':>9}")
    for name, solo_f, coupled_f in RUNS:
        if bb:
            # the Bitbrains runs carry eval_real.py's app suffix for io-burn
            app = "_io-burn" if "io-burn" in name else ""
            solo_f = solo_f.replace("_spiritquest_", f"{app}_spiritquest_bb_")
            coupled_f = coupled_f.replace("_spiritquest_", f"{app}_spiritquest_bb_")
        for tag, f in (("solo", solo_f), ("coupled", coupled_f)):
            s = summarize(f)
            print(f"{name:38}{tag:10}{s['sla']:>7.1f}{s['lat']:>8.2f}"
                  f"{s['rep_rev']:>9}{s['cpu_rev']:>9}{s['mem_rev']:>9}")
            rows.append(dict(tenant=name, run=tag, sla_pct=round(s["sla"], 1),
                              mean_lat_ms=round(s["lat"], 2), replica_reversals=s["rep_rev"],
                              cpu_reversals=s["cpu_rev"], mem_reversals=s["mem_rev"]))
    # committed summary (raw per-step real_*.csv stays gitignored, same convention
    # as every other real-cluster result in this project).
    out = os.path.join(HERE, "eval", f"coupling_summary{'_bitbrains' if bb else ''}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
