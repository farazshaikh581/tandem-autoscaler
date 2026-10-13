"""Shared helpers + safety guard for the Phase 0 HPA/VPA death-spiral experiment.

SAFETY: every script that mutates the cluster imports guard_namespace() and
refuses to operate on any system/shared namespace. Phase 0 lives ONLY in
hv-phase0, so it can never disturb default/workload-cpu|io or kube-system.
"""
import json
import subprocess
import sys

PROTECTED = {"default", "kube-system", "kube-public", "kube-node-lease"}


def guard_namespace(ns: str) -> None:
    if ns in PROTECTED:
        sys.exit(f"REFUSING to run against protected namespace '{ns}'. "
                 f"Phase 0 must target its own isolated namespace (hv-phase0).")


def kubectl(args: list[str], timeout: int = 15) -> str:
    """Run kubectl and return stdout (raises on non-zero)."""
    out = subprocess.run(["kubectl", *args, f"--request-timeout={timeout}s"],
                         capture_output=True, text=True, timeout=timeout + 5)
    if out.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def kubectl_json(args: list[str], timeout: int = 15) -> dict:
    return json.loads(kubectl([*args, "-o", "json"], timeout=timeout))


def parse_cpu_millicores(val: str) -> float:
    """Parse a k8s CPU quantity ('250m', '1', '1500m') into millicores (float)."""
    val = str(val).strip()
    if val.endswith("m"):
        return float(val[:-1])
    if val.endswith("n"):  # nanocores (kubectl top sometimes)
        return float(val[:-1]) / 1e6
    if val.endswith("u"):  # microcores
        return float(val[:-1]) / 1e3
    return float(val) * 1000.0


def pod_names(ns: str, selector: str) -> list[str]:
    out = kubectl(["get", "pods", "-n", ns, "-l", selector,
                   "-o", "jsonpath={.items[*].metadata.name}"]).strip()
    return out.split() if out else []


def top_pods_cpu_millicores(ns: str, selector: str) -> list[float]:
    """Per-pod CPU usage in millicores via the metrics API (kubectl top)."""
    try:
        out = kubectl(["top", "pods", "-n", ns, "-l", selector, "--no-headers"])
    except RuntimeError:
        return []
    vals = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals.append(parse_cpu_millicores(parts[1]))
            except ValueError:
                pass
    return vals
