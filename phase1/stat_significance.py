"""Statistical significance testing on the multi-seed comparisons (issue #18).

Every multi-seed claim in the README up to this point (issues #2, the #11
multi-seed-azure followup) is reported as mean +- std, never a formal test.
This adds one, honestly bounded by what the available seeds can support.

Seed counts are NOT uniform across algos: the original pass had 3 seeds
(0-2) for all 5 multi-seed algos; a follow-up run (motivated by this file's
own power calculation) added 4 more seeds (3-6) for qrdqn and
cvar-bellman-qrdqn specifically, since that was the one pairwise comparison
cheap enough to formally nail down. Each pairwise test below uses the
INTERSECTION of the two algos' available seeds, discovered by scanning for
eval CSVs rather than assumed -- so qrdqn vs cvar-bellman-qrdqn now runs at
n=7, every other pair stays at n=3. This matters for what a Wilcoxon
signed-rank test can even do: it cannot reach p<0.05 two-sided at n=3 no
matter the effect size (min achievable p = 2/2^3 = 0.25, a combinatorial
floor), but at n=7 it can (min p = 2/2^7 = 0.0156). A paired t-test is
treated as the primary test throughout (works at any n>=2, though few
degrees of freedom need a large effect to reach significance at small n).
With ~280 pairwise tests per dataset (10 algo pairs x 4 classes x 7
metrics), raw p<0.05 would flag ~14 "significant" results by chance alone
even if nothing were real, so significance is flagged on the
Benjamini-Hochberg-corrected p-value, not the raw one.

  python stat_significance.py                # Bitbrains (comparison.csv family; seeds 3-6 as bb_seed<n>)
  python stat_significance.py --tag azure    # Azure-organic (comparison_azure.csv family)
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(__file__)
ALGOS = ["qrdqn", "cvar-qrdqn", "cvar-bellman-qrdqn", "ppo", "dqn"]
METRICS = ["sla_pct", "mean_lat_ms", "mean_cost_vcpu", "churn",
           "replica_reversals", "cpu_reversals", "mem_reversals"]
ALPHA = 0.05


def eval_path(algo: str, seed: int, suffix: str) -> str:
    tail = f"_all_{suffix}" if suffix else "_all"
    tail += f"_seed{seed}" if seed else ""
    path = os.path.join(HERE, "eval", f"eval_{algo}{tail}.csv")
    if not suffix and not os.path.exists(path):
        # Bitbrains seeds trained after the Azure switch carry a "bb" tag token
        # (eval_<algo>_all_bb_seed<n>.csv); seeds 0-2 keep their original names.
        bb = os.path.join(HERE, "eval", f"eval_{algo}_all_bb_seed{seed}.csv")
        if seed and os.path.exists(bb):
            return bb
    return path


def available_seeds(algo: str, suffix: str, max_seed: int = 20) -> list[int]:
    """Discovers which seeds actually exist for this algo, rather than
    assuming a fixed count -- seed availability is no longer uniform across
    algos (see module docstring)."""
    return [s for s in range(max_seed + 1) if os.path.exists(eval_path(algo, s, suffix))]


def load_seed(algo: str, seed: int, suffix: str) -> pd.DataFrame:
    return pd.read_csv(eval_path(algo, seed, suffix)).set_index("workload")


def paired_tests(xa: np.ndarray, xb: np.ndarray) -> tuple[float, float]:
    """Returns (t_test_p, wilcoxon_p). Handles the identical-values edge case
    both scipy functions reject with an error rather than a p=1.0 result."""
    if np.allclose(xa, xb):
        return 1.0, 1.0
    _, t_p = stats.ttest_rel(xa, xb)
    try:
        _, w_p = stats.wilcoxon(xa, xb)
    except ValueError:
        w_p = 1.0  # all-zero-difference / insufficient-rank edge case
    return float(t_p), float(w_p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="'azure' for the organic-traffic seed family")
    args = ap.parse_args()

    seeds_by_algo = {a: available_seeds(a, args.tag) for a in ALGOS}
    print("seeds discovered per algo: " +
          ", ".join(f"{a}={seeds_by_algo[a]}" for a in ALGOS))
    data = {a: {s: load_seed(a, s, args.tag) for s in seeds_by_algo[a]} for a in ALGOS}
    classes = data[ALGOS[0]][seeds_by_algo[ALGOS[0]][0]].index.tolist()

    rows = []
    for a, b in itertools.combinations(ALGOS, 2):
        common_seeds = sorted(set(seeds_by_algo[a]) & set(seeds_by_algo[b]))
        for wl in classes:
            for metric in METRICS:
                xa = np.array([data[a][s].loc[wl, metric] for s in common_seeds], dtype=float)
                xb = np.array([data[b][s].loc[wl, metric] for s in common_seeds], dtype=float)
                t_p, w_p = paired_tests(xa, xb)
                rows.append(dict(
                    algo_a=a, algo_b=b, workload=wl, metric=metric, n_seeds=len(common_seeds),
                    mean_a=round(float(xa.mean()), 3), mean_b=round(float(xb.mean()), 3),
                    std_a=round(float(xa.std(ddof=1)), 3), std_b=round(float(xb.std(ddof=1)), 3),
                    t_p=round(t_p, 4), wilcoxon_p=round(w_p, 4)))
    df = pd.DataFrame(rows)

    # Benjamini-Hochberg across the whole family of tests in this dataset --
    # the "raw p<0.05" count is not the number to trust with this many tests.
    df["t_p_bh"] = stats.false_discovery_control(df["t_p"].to_numpy(), method="bh")
    df["sig_raw"] = df["t_p"] < ALPHA
    df["sig_bh"] = df["t_p_bh"] < ALPHA

    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(HERE, "eval", f"stat_significance{suffix}.csv")
    df.to_csv(out, index=False)

    n = len(df)
    print(f"\n{n} pairwise tests ({len(ALGOS)} algos x {len(classes)} classes x "
          f"{len(METRICS)} metrics; n_seeds varies per pair, see column).")
    print(f"  raw t-test p<0.05:        {df['sig_raw'].sum():3d} ({df['sig_raw'].mean()*100:.1f}%) "
          f"-- includes ~{n*ALPHA:.0f} expected by chance alone at this many tests")
    print(f"  BH-corrected significant: {df['sig_bh'].sum():3d} ({df['sig_bh'].mean()*100:.1f}%)")
    for n_seeds, grp in df.groupby("n_seeds"):
        floor = 2.0 ** (1 - n_seeds)  # 2 * P(all n diffs same sign) = 2 * (1/2^n)
        w_sig = (grp["wilcoxon_p"] < ALPHA).sum()
        note = "structurally impossible" if floor > ALPHA else "reachable"
        print(f"  n={n_seeds}: Wilcoxon min achievable p={floor:.4f} ({note}) -- "
              f"{w_sig} of {len(grp)} tests at this n reach raw Wilcoxon p<0.05")
    print(f"wrote {out}\n")

    for a, b in [("qrdqn", "cvar-bellman-qrdqn"), ("cvar-qrdqn", "cvar-bellman-qrdqn")]:
        sub = df[(df.algo_a == a) & (df.algo_b == b)]
        n_seeds = sub["n_seeds"].iloc[0]
        print(f"== {a} vs {b} (n={n_seeds}, BH-significant only) ==")
        sig = sub[sub.sig_bh]
        if sig.empty:
            print("  (none reach BH-corrected significance)")
        else:
            print(sig[["workload", "metric", "mean_a", "mean_b", "t_p", "t_p_bh"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
