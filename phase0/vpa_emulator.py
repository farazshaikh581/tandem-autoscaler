"""Lightweight VPA emulator: the vertical half of the coupled control loop.

Real VPA is not installed on the cluster, and installing cluster-wide CRDs would
be invasive. This script reproduces VPA's essential behavior (recommend a CPU
request from observed usage, then actuate it) with the one knob Phase 0 needs:

    --mode disruptive : patch the Deployment template -> pods are recreated
    --mode inplace    : patch each Pod via the `resize` subresource -> no restart

The recommender keeps observed usage near --target-util of the request, exactly
the pressure that, coupled with a utilization-based HPA, produces the death
spiral. All actions are logged to a CSV so the actuation cost (disruptive vs
in-place) is measurable.
"""
import argparse
import csv
import time

from common import (guard_namespace, kubectl, kubectl_json, parse_cpu_millicores,
                    top_pods_cpu_millicores)


def current_request_millicores(ns: str, deploy: str, container: str) -> float:
    d = kubectl_json(["get", "deploy", deploy, "-n", ns])
    for c in d["spec"]["template"]["spec"]["containers"]:
        if c["name"] == container:
            return parse_cpu_millicores(c.get("resources", {}).get("requests", {}).get("cpu", "0"))
    return 0.0


def apply_disruptive(ns: str, deploy: str, container: str, req_m: int, lim_m: int) -> None:
    # Patches the Deployment template -> triggers a rolling recreate of all pods.
    kubectl(["set", "resources", f"deployment/{deploy}", "-n", ns, "-c", container,
             f"--requests=cpu={req_m}m", f"--limits=cpu={lim_m}m"])


def pod_requests_millicores(ns: str, selector: str, container: str) -> dict[str, float]:
    """Live CPU request of each running (non-terminating) pod, from its spec.

    In-place mode must compare the recommendation against these, not against the
    Deployment template: resize patches pods and never touches the template, so
    the template value stays at its manifest default and a deadband measured
    against it never engages."""
    d = kubectl_json(["get", "pods", "-n", ns, "-l", selector])
    out = {}
    for item in d.get("items", []):
        if item["metadata"].get("deletionTimestamp"):
            continue
        for c in item["spec"]["containers"]:
            if c["name"] == container:
                out[item["metadata"]["name"]] = parse_cpu_millicores(
                    c.get("resources", {}).get("requests", {}).get("cpu", "0"))
    return out


def apply_inplace(ns: str, pods: list[str], container: str, req_m: int, lim_m: int) -> int:
    # Patches live pods via the resize subresource (k8s 1.35 GA) -> no restart.
    patch = ('{"spec":{"containers":[{"name":"%s","resources":'
             '{"requests":{"cpu":"%dm"},"limits":{"cpu":"%dm"}}}]}}' % (container, req_m, lim_m))
    n = 0
    for p in pods:
        try:
            # strategic (not JSON merge): merges the container list by name and
            # the resources maps, so memory is preserved. JSON merge patch would
            # replace the whole containers array and be rejected by resize.
            kubectl(["patch", "pod", p, "-n", ns, "--subresource", "resize",
                     "--type", "strategic", "-p", patch])
            n += 1
        except RuntimeError as e:
            print(f"  [warn] resize {p} failed: {e}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="VPA emulator (disruptive vs in-place)")
    ap.add_argument("--namespace", default="hv-phase0")
    ap.add_argument("--deployment", default="web-service")
    ap.add_argument("--container", default="web")
    ap.add_argument("--selector", default="app=web-service")
    ap.add_argument("--mode", choices=["disruptive", "inplace"], required=True)
    ap.add_argument("--interval", type=int, default=15, help="seconds between recommendations")
    ap.add_argument("--target-util", type=float, default=0.70,
                    help="keep observed usage near this fraction of the request")
    ap.add_argument("--min-cpu", type=int, default=100, help="min request (millicores)")
    ap.add_argument("--max-cpu", type=int, default=800, help="max request (millicores)")
    ap.add_argument("--limit-ratio", type=float, default=2.0)
    ap.add_argument("--ema", type=float, default=0.5, help="smoothing on the recommendation")
    ap.add_argument("--min-change", type=int, default=30, help="millicores; deadband to actuate")
    ap.add_argument("--duration", type=int, default=900)
    ap.add_argument("--out", default="results/vpa_emulator.csv")
    args = ap.parse_args()

    guard_namespace(args.namespace)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["ts", "elapsed_s", "mode", "avg_usage_m", "cur_req_m",
                "recommended_m", "applied", "n_pods_actuated"])

    t0 = time.time()
    rec = None
    print(f"[vpa-emulator] mode={args.mode} ns={args.namespace} target_util={args.target_util}")
    while time.time() - t0 < args.duration:
        loop_start = time.time()
        usage = top_pods_cpu_millicores(args.namespace, args.selector)
        if args.mode == "inplace":
            pod_req = pod_requests_millicores(args.namespace, args.selector, args.container)
            cur_req = sum(pod_req.values()) / len(pod_req) if pod_req else 0.0
        else:
            cur_req = current_request_millicores(args.namespace, args.deployment, args.container)
        avg_usage = sum(usage) / len(usage) if usage else 0.0

        applied, n_act = 0, 0
        if avg_usage > 0:
            raw = avg_usage / args.target_util
            rec = raw if rec is None else args.ema * rec + (1 - args.ema) * raw
            target_req = int(max(args.min_cpu, min(args.max_cpu, rec)))
            if args.mode == "inplace":
                # same deadband as disruptive mode, applied per pod: only pods whose
                # live request is off target by >= min_change are resized
                off = [p for p, v in pod_req.items() if abs(target_req - v) >= args.min_change]
            due = bool(off) if args.mode == "inplace" else abs(target_req - cur_req) >= args.min_change
            if due:
                lim = int(min(args.max_cpu * args.limit_ratio, target_req * args.limit_ratio))
                try:
                    if args.mode == "disruptive":
                        apply_disruptive(args.namespace, args.deployment, args.container, target_req, lim)
                        n_act = -1  # rollout: all pods
                    else:
                        n_act = apply_inplace(args.namespace, off, args.container, target_req, lim)
                    applied = 1
                    print(f"  t={int(time.time()-t0)}s usage={avg_usage:.0f}m "
                          f"req {cur_req:.0f}->{target_req}m ({args.mode})")
                except RuntimeError as e:
                    print(f"  [warn] actuation failed: {e}")

        w.writerow([f"{time.time():.0f}", f"{time.time()-t0:.0f}", args.mode,
                    f"{avg_usage:.0f}", f"{cur_req:.0f}",
                    f"{rec:.0f}" if rec else "", applied, n_act])
        f.flush()
        time.sleep(max(0, args.interval - (time.time() - loop_start)))

    f.close()
    print(f"[vpa-emulator] done -> {args.out}")


if __name__ == "__main__":
    main()
