"""Samples the coupled system over time -> CSV for the death-spiral plot.

Records, per tick: desired/ready replicas, the HPA's own view (current CPU
utilization % and desired replicas), the CPU request spread across pods, and
mean CPU usage. The replica count and the CPU-request columns are the two lines
that oscillate against each other when the loop spirals.
"""
import argparse
import csv
import time

from common import (guard_namespace, kubectl_json, parse_cpu_millicores,
                    top_pods_cpu_millicores)


def sample(ns: str, deploy: str, hpa: str, selector: str) -> dict:
    d = kubectl_json(["get", "deploy", deploy, "-n", ns])
    desired = d["spec"].get("replicas", 0)
    ready = d.get("status", {}).get("readyReplicas", 0) or 0

    h = kubectl_json(["get", "hpa", hpa, "-n", ns])
    hstat = h.get("status", {})
    hpa_desired = hstat.get("desiredReplicas", 0)
    hpa_util = ""
    for m in hstat.get("currentMetrics", []) or []:
        cur = m.get("resource", {}).get("current", {})
        if "averageUtilization" in cur:
            hpa_util = cur["averageUtilization"]

    pods = kubectl_json(["get", "pods", "-n", ns, "-l", selector])
    reqs = []
    for p in pods.get("items", []):
        for c in p["spec"]["containers"]:
            r = c.get("resources", {}).get("requests", {}).get("cpu")
            if r:
                reqs.append(parse_cpu_millicores(r))
    usage = top_pods_cpu_millicores(ns, selector)

    return {
        "desired_replicas": desired,
        "ready_replicas": ready,
        "hpa_desired": hpa_desired,
        "hpa_cur_util_pct": hpa_util,
        "req_min_m": f"{min(reqs):.0f}" if reqs else "",
        "req_max_m": f"{max(reqs):.0f}" if reqs else "",
        "req_avg_m": f"{sum(reqs)/len(reqs):.0f}" if reqs else "",
        "usage_avg_m": f"{sum(usage)/len(usage):.0f}" if usage else "",
        "n_pods": len(pods.get("items", [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor the coupled HPA/VPA system")
    ap.add_argument("--namespace", default="hv-phase0")
    ap.add_argument("--deployment", default="web-service")
    ap.add_argument("--hpa", default="web-service")
    ap.add_argument("--selector", default="app=web-service")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--duration", type=int, default=900)
    ap.add_argument("--out", default="results/monitor.csv")
    args = ap.parse_args()

    guard_namespace(args.namespace)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "w", newline="")
    cols = ["ts", "elapsed_s", "desired_replicas", "ready_replicas", "hpa_desired",
            "hpa_cur_util_pct", "req_min_m", "req_max_m", "req_avg_m", "usage_avg_m", "n_pods"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()

    t0 = time.time()
    print(f"[monitor] ns={args.namespace} every {args.interval}s for {args.duration}s")
    while time.time() - t0 < args.duration:
        loop_start = time.time()
        try:
            row = sample(args.namespace, args.deployment, args.hpa, args.selector)
            row["ts"] = f"{time.time():.0f}"
            row["elapsed_s"] = f"{time.time()-t0:.0f}"
            w.writerow(row)
            f.flush()
        except Exception as e:
            print(f"  [warn] sample failed: {e}")
        time.sleep(max(0, args.interval - (time.time() - loop_start)))

    f.close()
    print(f"[monitor] done -> {args.out}")


if __name__ == "__main__":
    main()
