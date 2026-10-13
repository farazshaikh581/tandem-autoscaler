"""Plot the death spiral: replica count and CPU request over time.

Usage:
    # one run
    python plot_spiral.py --run results/monitor_inplace.csv:in-place
    # compare disruptive vs in-place
    python plot_spiral.py \
        --run results/monitor_disruptive.csv:disruptive \
        --run results/monitor_inplace.csv:in-place \
        --out results/death_spiral.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str) -> dict:
    cols: dict[str, list] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def fnum(xs):
    out = []
    for x in xs:
        try:
            out.append(float(x))
        except (ValueError, TypeError):
            out.append(float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="path:label (repeatable)")
    ap.add_argument("--out", default="results/death_spiral.png")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        path, _, label = spec.partition(":")
        runs.append((path, label or path))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for path, label in runs:
        d = load(path)
        t = [x / 60.0 for x in fnum(d["elapsed_s"])]  # minutes
        ax1.plot(t, fnum(d["desired_replicas"]), marker=".", ms=3, label=label)
        ax2.plot(t, fnum(d["req_avg_m"]), marker=".", ms=3, label=label)

    ax1.set_ylabel("Replicas (desired)")
    ax1.set_title("HPA/VPA coupling: replica count vs CPU request over time")
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_ylabel("Avg CPU request (m)")
    ax2.set_xlabel("Time (min)")
    ax2.grid(alpha=0.3); ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
