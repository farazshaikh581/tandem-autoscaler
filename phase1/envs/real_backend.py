"""RealClusterBackend: actuate + measure the real hv-phase0 deployment.

Presents the SAME interface as SimBackend (`.s`, `.n`, reset/apply/step_physics)
so a policy trained in the simulator replays on the cluster with no change to
JointHVEnv. Horizontal scaling is `kubectl scale`; vertical scaling is the
in-place `resize` subresource (k8s 1.35 GA, strategic patch, no restart).
Per-step latency is measured by driving real HTTP load at the class demand;
CPU/memory come from metrics-server (`kubectl top`).

Reuses the Phase 0 primitives (common.py, load.py) so the actuation and the
latency capture are identical to the death-spiral experiment.

SCOPE / caveats (honest):
  * The memory axis is now exercised for real: the app holds a resident ballast
    (GET /alloc?mb=N) driven from the memory-bound trace, so an under-provisioned
    pod (working set > memory limit) is OOM-killed by the kernel and the resulting
    request failures are measured in the load window. `mem_need_now` is commanded
    (deterministic, like the CPU load_frac); its consequence is real. NOTE the
    real memory penalty is a binary OOM cliff, not the sim's gradual BETA_MEM
    latency inflation -- a genuine sim-vs-real difference, not a bug.
  * In-place CPU resize is restart-free (proven in Phase 0). In-place *memory*
    resize can require a restart depending on the container resizePolicy; failures
    are tolerated and `.s` is re-synced to the pods' actual requests so the
    observation never lies about cluster state.
  * Safety: guard_namespace() refuses any shared/system namespace. This backend
    only ever touches hv-phase0.
"""
from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np

# Phase 0 primitives (actuation + load) live one level up.
_PHASE1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PHASE0 = os.path.join(_PHASE1, "..", "phase0")
for _p in (_PHASE1, _PHASE0):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from common import guard_namespace, kubectl, kubectl_json, parse_cpu_millicores, pod_names  # noqa: E402
from loadgen import run_load, summarize  # noqa: E402  (multi-process; delegates to phase0 at low rps)

from envs.joint_hv_env import (  # noqa: E402
    CPU_MAX, CPU_MIN, MEM_MAX, MEM_MIN, MEM_NEED_MI, REPLICA_MAX, REPLICA_MIN, SysState,
)

# ---- cluster addressing (matches phase0/web-service*.yaml) ----
NS = "hv-phase0"       # both apps live in the same namespace

# Per-app deployment identity. "cpu-burn" is the original death-spiral app
# (phase0/web-service.yaml); "io-burn" is the app-diversity workstream's second
# app (phase0/web-service-io.yaml) -- same /health+/alloc surface, but its
# default path does blocking write+fsync IO instead of CPU hashing. Resource
# envelope is identical across presets on purpose: the variable under test is
# app behavior, not sizing.
APP_PRESETS = {
    "cpu-burn": dict(deploy="web-service", container="web", selector="app=web-service",
                      base_cpu_m=200, base_mem_mi=128, base_replicas=2),
    "io-burn": dict(deploy="web-service-io", container="web", selector="app=web-service-io",
                     base_cpu_m=200, base_mem_mi=128, base_replicas=2),
}

# ---- load mapping ----
# load_frac in [0,1] from the class trace -> real requests/sec. Per-class cpu
# intensity scales it so cpu-bound drives real traffic and memory-bound stays
# near-idle (the real app is CPU-only), mirroring the sim's intensity design.
# MAX_RPS calibrated on the testbed (baseline 2x200m): latency is flat ~5 ms to
# ~80 rps, crosses the knee ~100 rps, saturates ~420 ms by 160 rps. 180 puts
# load_frac=1 into saturation while staying recoverable by a ~2x scale-up, so the
# policy faces a solvable challenge instead of a hopeless or trivial one.
DEFAULT_MAX_RPS = 180                    # load_frac*cpu_int = 1 -> this many rps
DEFAULT_STEP_SECONDS = 8                 # HTTP load window per control step
LAT_MISS_MS = 5000.0                     # whole window failed -> treat as timeout latency


def _mem_mi(val: str) -> float:
    """Parse a k8s memory quantity from `kubectl top` ('13Mi', '512Ki', '1Gi')."""
    val = str(val).strip()
    for suf, mult in (("Gi", 1024.0), ("Mi", 1.0), ("Ki", 1 / 1024.0), ("Ei", 1024.0 ** 3)):
        if val.endswith(suf):
            return float(val[:-len(suf)]) * mult
    try:
        return float(val) / (1024.0 * 1024.0)   # raw bytes
    except ValueError:
        return 0.0


class RealClusterBackend:
    """Actuate + measure hv-phase0. Drop-in for SimBackend."""

    def __init__(self, trace: dict, rng: np.random.Generator, intensity: dict,
                 max_rps: int = DEFAULT_MAX_RPS, step_seconds: int = DEFAULT_STEP_SECONDS,
                 app: str = "cpu-burn"):
        guard_namespace(NS)
        preset = APP_PRESETS[app]
        self.deploy, self.container, self.selector = preset["deploy"], preset["container"], preset["selector"]
        self.base_cpu_m, self.base_mem_mi = preset["base_cpu_m"], preset["base_mem_mi"]
        self.base_replicas = preset["base_replicas"]
        self.rng = rng
        self.cpu_int, self.mem_int = intensity["cpu"], intensity["mem"]
        self.max_rps, self.step_seconds = max_rps, step_seconds
        cpu = np.nan_to_num(trace["cpu_norm"]).astype(np.float32)
        self.load = np.clip(cpu / (np.percentile(cpu, 99) or 1.0), 0, 1.5)
        # memory-bound working-set trace: same normalization as SimBackend, so the
        # ballast the pods hold each step tracks the same demand the sim replays.
        mem = np.nan_to_num(trace["mem_MB"]).astype(np.float32)
        self.memf = np.clip(mem / (np.percentile(mem, 99) or 1.0), 0, 1.0)
        self.n = len(self.load)
        self.t = 0
        self.s = SysState()
        self.url = self._service_url()

    # ---- cluster helpers ----
    def _service_url(self) -> str:
        svc = kubectl_json(["get", "svc", self.deploy, "-n", NS])
        ip = svc["spec"]["clusterIP"]
        port = svc["spec"]["ports"][0]["port"]
        return f"http://{ip}:{port}/"

    def _ready_replicas(self) -> int:
        d = kubectl_json(["get", "deploy", self.deploy, "-n", NS])
        return int(d.get("status", {}).get("readyReplicas", 0) or 0)

    def _pod_ips(self) -> list[str]:
        out = kubectl(["get", "pods", "-n", NS, "-l", self.selector,
                       "-o", "jsonpath={.items[*].status.podIP}"]).strip()
        return out.split() if out else []

    def _set_ballast_all(self, mb: int) -> None:
        """Drive every live pod's resident working set to `mb` MiB via /alloc. This
        is the memory-bound demand (the CPU-burn path allocates nothing). Failures
        are tolerated on purpose: a pod whose ballast exceeds its memory limit is
        OOM-killed and drops the connection -- that under-provisioning shows up as
        the real request failures the load window measures."""
        for ip in self._pod_ips():
            try:
                urllib.request.urlopen(f"http://{ip}:8080/alloc?mb={int(mb)}",
                                       timeout=self.step_seconds).read()
            except Exception:
                pass

    def _top(self) -> list[tuple[float, float]]:
        """Per-pod (cpu_millicores, mem_MiB) from metrics-server; [] if unavailable."""
        try:
            out = kubectl(["top", "pods", "-n", NS, "-l", self.selector, "--no-headers"])
        except RuntimeError:
            return []
        rows = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append((parse_cpu_millicores(parts[1]), _mem_mi(parts[2])))
                except ValueError:
                    pass
        return rows

    def _container_resources(self, pod: str) -> dict:
        """The web container's resources map ({requests:{}, limits:{}}) for a pod.
        Returns {} if the pod vanished mid-query -- expected under memory-bound OOM
        churn, where a pod named by pod_names() can be gone a moment later."""
        try:
            spec = kubectl_json(["get", "pod", pod, "-n", NS])
        except RuntimeError:
            return {}
        for c in spec["spec"]["containers"]:
            if c["name"] == self.container:
                return c.get("resources", {})
        return {}

    def _live_requests(self) -> tuple[int, int]:
        """The web container's actual (cpu_m, mem_Mi) request on the first live pod.
        Skips pods that vanished (empty resources) so OOM churn can't spoof a base."""
        for p in pod_names(NS, self.selector):
            req = self._container_resources(p).get("requests", {})
            if not req:
                continue
            cpu = int(round(parse_cpu_millicores(req.get("cpu", f"{self.base_cpu_m}m"))))
            mem = int(round(_mem_mi(req.get("memory", f"{self.base_mem_mi}Mi"))))
            return cpu, mem
        return self.base_cpu_m, self.base_mem_mi

    def _qos_ratios(self) -> tuple[float, float]:
        """(cpu, mem) limit/request ratios of the live pods. Resize may not change a
        pod's QoS class, so we preserve this ratio: Guaranteed pods (ratio 1.0) keep
        limits == requests, Burstable pods keep their spread."""
        for p in pod_names(NS, self.selector):
            res = self._container_resources(p)
            if not res:
                continue
            req, lim = res.get("requests", {}), res.get("limits", {})
            rc = parse_cpu_millicores(req.get("cpu", "1"))
            rm = _mem_mi(req.get("memory", "1"))
            cpu_r = parse_cpu_millicores(lim["cpu"]) / rc if lim.get("cpu") and rc else 1.0
            mem_r = _mem_mi(lim["memory"]) / rm if lim.get("memory") and rm else 1.0
            return cpu_r, mem_r
        return 1.0, 1.0

    def _resize_all(self, cpu_m: int, mem_mi: int) -> None:
        """In-place resize every live pod to (cpu_m, mem_mi) via strategic patch.
        Idempotent: also normalizes pods freshly created by a scale-up. Preserves the
        QoS class by keeping the pods' request:limit ratio. CPU resize is restart-free;
        memory failures are tolerated (see module scope)."""
        cpu_r, mem_r = self._qos_ratios()
        lim_cpu, lim_mem = int(round(cpu_m * cpu_r)), int(round(mem_mi * mem_r))
        patch = ('{"spec":{"containers":[{"name":"%s","resources":'
                 '{"requests":{"cpu":"%dm","memory":"%dMi"},'
                 '"limits":{"cpu":"%dm","memory":"%dMi"}}}]}}'
                 % (self.container, cpu_m, mem_mi, lim_cpu, lim_mem))
        for p in pod_names(NS, self.selector):
            try:
                kubectl(["patch", "pod", p, "-n", NS, "--subresource", "resize",
                         "--type", "strategic", "-p", patch])
            except RuntimeError as e:
                print(f"  [warn] resize {p} failed (kept prior): {e}")

    # ---- SimBackend-compatible API ----
    def reset(self, start: int) -> None:
        self.t = start
        kubectl(["scale", "deploy", self.deploy, "-n", NS, f"--replicas={self.base_replicas}"], timeout=30)
        self._resize_all(self.base_cpu_m, self.base_mem_mi)
        self._set_ballast_all(0)          # clear any residual working set from a prior run
        self.s = SysState()
        self.s.replicas = self.s.target_replicas = self._ready_replicas() or self.base_replicas
        self.s.cpu_req, self.s.mem_req = self._live_requests()

    def apply(self, d_rep: int, d_cpu: int, d_mem: int) -> None:
        s = self.s
        new_rep = int(np.clip(s.target_replicas + d_rep, REPLICA_MIN, REPLICA_MAX))
        new_cpu = int(np.clip(s.cpu_req + d_cpu, CPU_MIN, CPU_MAX))
        new_mem = int(np.clip(s.mem_req + d_mem, MEM_MIN, MEM_MAX))
        if new_rep != s.target_replicas:
            kubectl(["scale", "deploy", self.deploy, "-n", NS, f"--replicas={new_rep}"], timeout=30)
            s.target_replicas = new_rep
        # resize every step so scale-up pods inherit the current request too
        self._resize_all(new_cpu, new_mem)
        # re-sync `.s` to what the cluster actually accepted (memory may not shrink)
        s.cpu_req, s.mem_req = self._live_requests()

    def step_physics(self) -> dict:
        s = self.s
        s.replicas = self._ready_replicas() or s.replicas
        load_frac = float(self.load[self.t % self.n])
        # memory-bound demand: hold this working set on every pod before the load
        # window, so pods under-provisioned on memory OOM-kill during measurement.
        mem_need = float(self.memf[self.t % self.n]) * MEM_NEED_MI * self.mem_int
        self._set_ballast_all(int(round(mem_need)))
        rps = max(1, int(round(load_frac * self.cpu_int * self.max_rps)))
        samples = run_load(self.url, rps, self.step_seconds, quiet=True)
        summ = summarize(samples)
        latency = summ["mean_ms"] if summ["n"] and not np.isnan(summ["mean_ms"]) else LAT_MISS_MS
        success = summ["success"] if summ["n"] else 0.0

        top = self._top()
        cpu_used = sum(c for c, _ in top)
        req_total = max(s.replicas * s.cpu_req, 1.0)
        rho = cpu_used / req_total
        # working set is commanded (like the sim's load_frac), so mem_pressure is
        # deterministic; its *consequence* -- OOM-driven failures -- is what the real
        # load window above actually measures in `success`/`latency`.
        s.mem_need_now = mem_need
        mem_pressure = max(0.0, (mem_need - s.mem_req) / max(s.mem_req, 1))

        self.t += 1
        return dict(load_frac=load_frac, rho=rho, latency=latency, success=success,
                    mem_pressure=mem_pressure, cpu_util=min(rho, 1.5))
