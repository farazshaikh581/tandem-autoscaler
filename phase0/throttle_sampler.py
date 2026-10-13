"""Sample CFS throttling of the web-service pods from each worker's cAdvisor endpoint.

Writes one row per (sample, pod): cumulative CFS periods, throttled periods and
throttled seconds of the `web` container, read through the API server's node proxy
(no exec into the pods). Counters are per container and start at 0 when the container
starts, so a run's totals are per-pod deltas; see summarize_throttle().

    python throttle_sampler.py --duration 900 --out results/throttle_<tag>.csv
    python throttle_sampler.py --summarize results/throttle_*.csv
"""
import argparse
import csv
import re
import subprocess
import sys
import time

METRICS = {
    "container_cpu_cfs_periods_total": "periods",
    "container_cpu_cfs_throttled_periods_total": "throttled_periods",
    "container_cpu_cfs_throttled_seconds_total": "throttled_s",
}
LINE = re.compile(r'^(\w+)\{(.*)\} (\S+)')
LABEL = re.compile(r'(\w+)="([^"]*)"')


def nodes() -> list[str]:
    out = subprocess.run(["kubectl", "get", "nodes", "-l", "!node-role.kubernetes.io/control-plane",
                          "-o", "jsonpath={.items[*].metadata.name}"],
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def scrape(node: str, ns: str, prefix: str, exclude: str) -> dict[str, dict]:
    raw = subprocess.run(["kubectl", "get", "--raw", f"/api/v1/nodes/{node}/proxy/metrics/cadvisor"],
                         capture_output=True, text=True, timeout=20).stdout
    pods: dict[str, dict] = {}
    for line in raw.splitlines():
        m = LINE.match(line)
        if not m or m.group(1) not in METRICS:
            continue
        lab = dict(LABEL.findall(m.group(2)))
        pod = lab.get("pod", "")
        if lab.get("namespace") != ns or lab.get("container") != "web" \
                or not pod.startswith(prefix) or pod.startswith(exclude):
            continue
        pods.setdefault(pod, {})[METRICS[m.group(1)]] = float(m.group(3))
    return pods


def sample(args) -> None:
    ws = nodes()
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "node", "pod", "periods", "throttled_periods", "throttled_s"])
        while time.time() - t0 < args.duration:
            el = int(time.time() - t0)
            for n in ws:
                try:
                    for pod, v in scrape(n, args.namespace, args.prefix, args.exclude).items():
                        w.writerow([el, n, pod, int(v.get("periods", 0)),
                                    int(v.get("throttled_periods", 0)), round(v.get("throttled_s", 0.0), 4)])
                except subprocess.SubprocessError:
                    pass
            f.flush()
            time.sleep(args.interval)


def summarize_throttle(path: str) -> dict:
    """Run totals: per pod, last minus first sample, except that a pod first seen after
    the first sample started during the run and is counted from 0."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {"run": path, "periods": 0, "throttled_periods": 0, "throttled_pct": "", "throttled_s": 0}
    t_first = min(int(r["elapsed_s"]) for r in rows)
    first, last = {}, {}
    for r in rows:
        first.setdefault(r["pod"], r)
        last[r["pod"]] = r
    tot = {"periods": 0, "throttled_periods": 0, "throttled_s": 0.0}
    for pod, r in last.items():
        base = first[pod] if int(first[pod]["elapsed_s"]) == t_first else None
        for k in tot:
            tot[k] += float(r[k]) - (float(base[k]) if base else 0.0)
    pct = 100.0 * tot["throttled_periods"] / tot["periods"] if tot["periods"] else 0.0
    return {"run": path.rsplit("/", 1)[-1], "periods": int(tot["periods"]),
            "throttled_periods": int(tot["throttled_periods"]), "throttled_pct": round(pct, 2),
            "throttled_s": round(tot["throttled_s"], 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--namespace", default="hv-phase0")
    ap.add_argument("--prefix", default="web-service-")
    ap.add_argument("--exclude", default="web-service-io-")
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--duration", type=int, default=900)
    ap.add_argument("--out", default="results/throttle.csv")
    ap.add_argument("--summarize", nargs="+", metavar="CSV")
    args = ap.parse_args()
    if args.summarize:
        w = csv.DictWriter(sys.stdout, ["run", "periods", "throttled_periods", "throttled_pct", "throttled_s"])
        w.writeheader()
        for p in args.summarize:
            w.writerow(summarize_throttle(p))
    else:
        sample(args)


if __name__ == "__main__":
    main()
