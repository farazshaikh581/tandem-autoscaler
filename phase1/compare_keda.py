"""KEDA baseline vs the RL headliners on the real cluster.

Reads the per-step real_*.csv written by eval_real_keda.py (`keda`, `keda-vpa`)
and eval_real.py (`qrdqn`, `cvar-bellman-qrdqn` / Option B) for the cpu-bound and
memory-bound windows, and reports the same SLA / latency / cost / reversal
metrics used in every other real-cluster table. RL rows prefer the matched
`_kedacmp` run (same cluster state and windows as the KEDA runs); if absent they
fall back to the historical real_<algo>_<workload>.csv and are flagged.

  python compare_keda.py
  python compare_keda.py --family bitbrains   # the *_bb runs -> keda_vs_rl_summary_bitbrains.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from train import _reversals

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "eval")
WORKLOADS = ["cpu-bound", "memory-bound"]
EXPECTED_STEPS = 20   # run_keda_real.sh STEPS; a shorter CSV means a crashed run,
                      # not a real result -- flag it rather than average it in
# label -> (filename stem, is_rl). RL stems get the _kedacmp / historical lookup.
CONTROLLERS = [
    ("keda", "real_keda", False),
    ("keda-vpa", "real_keda-vpa", False),
    ("QR-DQN", "real_qrdqn", True),
    ("Option B", "real_cvar-bellman-qrdqn", True),
]


def resolve(stem: str, workload: str, is_rl: bool, fam: str = "") -> tuple[str, bool]:
    """Return (path, matched). RL: prefer <stem>_<wl>_kedacmp.csv, else <stem>_<wl>.csv.
    fam is "" for azure or "_bb"; a Bitbrains RL row has no historical fallback."""
    if is_rl:
        matched = os.path.join(EVAL, f"{stem}_{workload}_kedacmp{fam}.csv")
        if os.path.exists(matched) or fam:
            return matched, True
    return os.path.join(EVAL, f"{stem}_{workload}{fam}.csv"), not is_rl


def summarize(path: str) -> dict:
    df = pd.read_csv(path)
    cost = (df["replicas"] * df["cpu_req_m"] / 1000.0).mean()
    return dict(
        sla=100.0 * df["sla_ok"].mean(),
        lat=df["latency_ms"].mean(),
        cost=cost,
        rep_rev=_reversals(df["replicas"].tolist()),
        mem_rev=_reversals(df["mem_req_mi"].tolist()),
        n=len(df),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["azure", "bitbrains"], default="azure")
    fam = "_bb" if ap.parse_args().family == "bitbrains" else ""
    rows = []
    hdr = (f"{'workload':13}{'controller':11}{'SLA%':>7}{'lat_ms':>9}{'cost':>7}"
           f"{'rep_rev':>9}{'mem_rev':>9}{'n':>4}  note")
    print(hdr)
    print("-" * len(hdr))
    for workload in WORKLOADS:
        for label, stem, is_rl in CONTROLLERS:
            path, matched = resolve(stem, workload, is_rl, fam)
            if not os.path.exists(path):
                print(f"{workload:13}{label:11}{'--':>7}{'(missing)':>9}")
                continue
            s = summarize(path)
            flags = []
            if not matched:
                flags.append("historical run (earlier testbed topology + load scale)")
            if s["n"] != EXPECTED_STEPS:
                flags.append(f"SHORT {s['n']}/{EXPECTED_STEPS} steps -- run likely crashed")
            note = "; ".join(flags)
            print(f"{workload:13}{label:11}{s['sla']:>7.1f}{s['lat']:>9.1f}{s['cost']:>7.2f}"
                  f"{s['rep_rev']:>9}{s['mem_rev']:>9}{s['n']:>4}  {note}")
            rows.append(dict(workload=workload, controller=label,
                             sla_pct=round(s["sla"], 1), mean_lat_ms=round(s["lat"], 1),
                             mean_cost_vcpu=round(s["cost"], 2),
                             replica_reversals=s["rep_rev"], mem_reversals=s["mem_rev"],
                             steps=s["n"], matched=matched,
                             short=s["n"] != EXPECTED_STEPS))
        print()

    out = os.path.join(EVAL, f"keda_vs_rl_summary{'_bitbrains' if fam else ''}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
