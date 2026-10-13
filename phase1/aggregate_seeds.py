"""Seed-aware aggregation: mean+-std per algo x workload over eval_<algo>_all_seed{0,1,2}.

Seed 0 is the original single-rep run (eval_<algo>_all.csv); seeds 1/2 are the
contenders-tier reruns from run_seed_lane.sh (issue #2). Only algos with all three
seeds present are aggregated (trpo/sac-discrete/hpa-* stay single-seed, dominated
or non-comparable).

  python aggregate_seeds.py                # -> comparison_seeded.csv
  python aggregate_seeds.py --suffix azure  # azure-tagged seeds -> comparison_azure_seeded.csv
"""
import argparse
import glob
import os

import pandas as pd

HERE = os.path.dirname(__file__)
METRICS = ["sla_pct", "mean_lat_ms", "mean_cost_vcpu", "churn",
           "replica_reversals", "cpu_reversals", "mem_reversals"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="e.g. 'azure' -> eval_<algo>_all_azure[_seed{1,2}].csv")
    args = ap.parse_args()
    tag0 = f"_{args.suffix}" if args.suffix else ""          # seed-0 filename suffix
    tagN = f"_{args.suffix}_seed" if args.suffix else "_seed"  # seed-1/2 filename prefix

    algos = []
    for f in sorted(glob.glob(os.path.join(HERE, "eval", f"eval_*_all{tag0}.csv"))):
        algo = os.path.basename(f)[len("eval_"):-len(f"_all{tag0}.csv")]
        if "smoke" in algo or algo in ("hpa-only", "hpa-vpa") or "_seed" in algo:
            continue
        s1 = os.path.join(HERE, "eval", f"eval_{algo}_all{tagN}1.csv")
        s2 = os.path.join(HERE, "eval", f"eval_{algo}_all{tagN}2.csv")
        if os.path.exists(s1) and os.path.exists(s2):
            algos.append(algo)

    if not algos:
        print("no multi-seed eval CSVs yet"); return

    frames = []
    for algo in algos:
        for seed in (0, 1, 2):
            fname = f"eval_{algo}_all{tag0}.csv" if seed == 0 else f"eval_{algo}_all{tagN}{seed}.csv"
            d = pd.read_csv(os.path.join(HERE, "eval", fname))
            d.insert(0, "seed", seed)
            d.insert(0, "algo", algo)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    agg = df.groupby(["algo", "workload"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    out_name = f"comparison_{args.suffix}_seeded.csv" if args.suffix else "comparison_seeded.csv"
    out = os.path.join(HERE, "eval", out_name)
    agg.to_csv(out, index=False)

    for algo in algos:
        print(f"\n== {algo} (mean +- std over seeds 0/1/2) ==")
        sub = agg[agg.algo == algo]
        for _, row in sub.iterrows():
            print(f"  {row.workload:14s} SLA {row.sla_pct_mean:5.1f}+-{row.sla_pct_std:4.1f}  "
                  f"reversals(rep/cpu/mem) "
                  f"{row.replica_reversals_mean:6.1f}+-{row.replica_reversals_std:5.1f} / "
                  f"{row.cpu_reversals_mean:6.1f}+-{row.cpu_reversals_std:5.1f} / "
                  f"{row.mem_reversals_mean:6.1f}+-{row.mem_reversals_std:5.1f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
