"""Aggregate per-algo eval CSVs into an algorithm x workload comparison.

  python aggregate_eval.py                  # eval_*_all.csv -> comparison.csv
  python aggregate_eval.py --suffix azure    # eval_*_all_azure.csv -> comparison_azure.csv
  python aggregate_eval.py --family bitbrains  # eval_*_all.csv + eval_*_all_bb.csv
                                               #   -> comparison_bitbrains.csv

The Bitbrains rows live under two names: the original Phase 1 runs (eval_*_all.csv,
tables frozen in comparison.csv) and the controllers added after the Azure switch,
run later on Bitbrains with a "bb" tag (eval_*_all_bb.csv). --family bitbrains joins
both into one complete table and leaves comparison.csv as it was.
"""
import argparse
import glob
import os

import pandas as pd

HERE = os.path.dirname(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="e.g. 'azure' to aggregate eval_*_all_azure.csv")
    ap.add_argument("--family", choices=["bitbrains"], default=None,
                    help="'bitbrains': join eval_*_all.csv and eval_*_all_bb.csv")
    args = ap.parse_args()

    if args.family == "bitbrains":
        tails, out_name = ["_all.csv", "_all_bb.csv"], "comparison_bitbrains.csv"
    else:
        tails = [f"_all_{args.suffix}.csv" if args.suffix else "_all.csv"]
        out_name = f"comparison_{args.suffix}.csv" if args.suffix else "comparison.csv"

    frames, seen = [], set()
    for tail in tails:
        for f in sorted(glob.glob(os.path.join(HERE, "eval", f"eval_*{tail}"))):
            if "smoke" in f:
                continue
            algo = os.path.basename(f)[len("eval_"):-len(tail)]
            if algo in seen:
                raise SystemExit(f"{algo} appears under more than one name, refusing to guess")
            seen.add(algo)
            d = pd.read_csv(f); d.insert(0, "algo", algo); frames.append(d)
    if not frames:
        print("no full-run eval CSVs yet"); return
    df = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(HERE, "eval", out_name)
    df.to_csv(out_path, index=False)

    for metric in ["sla_pct", "mean_lat_ms", "mean_cost_vcpu", "churn"]:
        print(f"\n== {metric} (algo x workload) ==")
        print(df.pivot(index="algo", columns="workload", values=metric).to_string())
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
