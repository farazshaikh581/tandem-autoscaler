"""Summary metrics for Phase 0 runs (monitor.py CSV + vertical-controller log).

Reproduces the figures in runs/2026-07-02_spiral_comparison/COMPARISON.md and
runs/2026-10-02_default_hpa_rps120/NOTES.md from their raw CSVs, and applies the
same definitions to later runs.

  replica changes / reversals   desired_replicas steps, and sign flips between
                                consecutive steps
  request changes / reversals   the same on the pods' mean CPU request (steps of
                                at least 1m)
  vertical actions              emulator: applied recommendations (and pods
                                resized in place); real VPA: InPlaceResizedByVPA
                                and EvictedByVPA events
  valid samples                 samples where the HPA reported a utilization
  under-provisioned             share of valid samples above 100% utilization
  total request                 mean of replicas x mean request (millicores)

  python summarize_runs.py results/monitor_inplace_tuned70.csv ...
  python summarize_runs.py --group results/monitor_*_tuned70*.csv ...   # mean+-std per cell
"""
import csv
import os
import re
import sys


def _steps(xs: list[float], min_step: float = 0.5) -> tuple[int, int]:
    ch = [b - a for a, b in zip(xs, xs[1:]) if abs(b - a) >= min_step]
    return len(ch), sum(1 for a, b in zip(ch, ch[1:]) if (a > 0) != (b > 0))


def summarize(monitor: str) -> dict:
    rows = list(csv.DictReader(open(monitor)))
    rep = [int(r["desired_replicas"]) for r in rows]
    req = [float(r["req_avg_m"]) for r in rows if r["req_avg_m"]]
    util = [float(r["hpa_cur_util_pct"]) for r in rows if r["hpa_cur_util_pct"]]
    tot = [int(r["desired_replicas"]) * float(r["req_avg_m"]) for r in rows if r["req_avg_m"]]
    rep_ch, rep_rev = _steps(rep)
    req_ch, req_rev = _steps(req, 1.0)
    out = dict(run=os.path.basename(monitor), replica_changes=rep_ch, replica_reversals=rep_rev,
               replicas=f"{min(rep)}-{max(rep)}", mean_replicas=round(sum(rep) / len(rep), 1),
               request_changes=req_ch, request_reversals=req_rev,
               mean_request_m=round(sum(req) / len(req)), total_request_m=round(sum(tot) / len(tot)),
               valid_samples=f"{len(util)}/{len(rows)}",
               valid_pct=round(100 * len(util) / len(rows), 1),
               under_provisioned_pct=round(100 * sum(u > 100 for u in util) / len(util), 1),
               peak_util_pct=round(max(util)))
    out.update(_vertical(monitor))
    out.update(_latency(monitor))
    return out


def _latency(monitor: str) -> dict:
    """Client-side latency from load.py's closing line (results/load_<stem>.txt)."""
    d, name = os.path.split(monitor)
    p = os.path.join(d, f"load_{name[len('monitor_'):-len('.csv')]}.txt")
    if not os.path.exists(p):
        return {}
    m = re.search(r"success=([\d.]+) mean=([\d.]+)ms .*p99=([\d.]+)", open(p).read())
    if not m:
        return {}
    return dict(success_pct=round(100 * float(m[1]), 2), mean_ms=float(m[2]), p99_ms=float(m[3]))


def _vertical(monitor: str) -> dict:
    d, name = os.path.split(monitor)
    stem = name[len("monitor_"):-len(".csv")]
    if stem.startswith("vpa-"):                       # real VPA: updater events
        mode, tag = stem[len("vpa-"):].split("_", 1)
        p = os.path.join(d, f"vpaevents_{mode}_{tag}.csv")
        if not os.path.exists(p):
            return {}
        ev = {r["reason"]: int(r["count"]) for r in csv.DictReader(open(p))}
        return dict(vertical_actions=ev.get("InPlaceResizedByVPA", 0) + ev.get("EvictedByVPA", 0),
                    pods_resized_in_place=ev.get("InPlaceResizedByVPA", 0),
                    pods_evicted=ev.get("EvictedByVPA", 0))
    p = os.path.join(d, f"vpa_{stem}.csv")            # emulator log
    if not os.path.exists(p):
        return {}
    v = list(csv.DictReader(open(p)))
    return dict(vertical_actions=sum(int(r["applied"]) for r in v),
                pods_resized_in_place=sum(int(r["n_pods_actuated"]) for r in v
                                          if int(r["n_pods_actuated"]) > 0))


# First runs recorded under an older tag, mapped to the cell they belong to.
ALIASES = {"disruptive_default2w": "disruptive_stock60"}


def cell(run: str) -> str:
    """Cell name of a run: the file stem without its _r<N> repetition suffix."""
    stem = re.sub(r"_r\d+$", "", run[len("monitor_"):-len(".csv")])
    return ALIASES.get(stem, stem)


def grouped(rows: list[dict]) -> list[dict]:
    """Mean and sample std per cell over its repetitions (std blank for n=1)."""
    cells: dict[str, list[dict]] = {}
    for r in rows:
        cells.setdefault(cell(r["run"]), []).append(r)
    out = []
    for name, rs in cells.items():
        g = dict(cell=name, n=len(rs))
        for k in rs[0]:
            vals = [r.get(k) for r in rs]
            if k == "run" or not all(isinstance(v, (int, float)) for v in vals):
                continue
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else None
            g[k] = f"{m:.1f}" if sd is None else f"{m:.1f}+-{sd:.1f}"
        out.append(g)
    return out


def main() -> None:
    args = sys.argv[1:]
    group = "--group" in args
    rows = [summarize(p) for p in args if p != "--group"]
    if group:
        rows = grouped(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(sys.stdout, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
