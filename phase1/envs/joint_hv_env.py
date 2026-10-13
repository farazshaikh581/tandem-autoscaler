"""JointHVEnv: Gymnasium env for joint horizontal + vertical + memory autoscaling.

Action: Discrete(27) = dReplicas{-1,0,+1} x dCPU{-100,0,+100}m x dMem{-64,0,+64}Mi.
Obs (normalized [0,1]): memory-inclusive, with the causal `pending = target-ready`.
Reward: SLA (mean latency < 20 ms) - cost - churn, with an overload/OOM penalty.

Two interchangeable backends behind one interface:
  SimBackend         - fast queueing simulator for training (default)
  RealClusterBackend - kubectl scale + in-place resize on the real cluster (eval); stub

CALIBRATION: the CPU axis (rho -> latency/success) is calibrated against the real
cluster (phase1/calibration, issue #1): a knee model with flat base latency and a
saturation plateau, fitted at 0.8% latency MAPE. Constants still marked [CAL]
(memory axis, per-class intensity) await real measurements. The phase0 app is
CPU-only, so the memory axis has no ground truth yet.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # allow import without gymnasium for inspection
    gym = None

# ---- bounds & steps (from PARAMETERS.md) ----
REPLICA_MIN, REPLICA_MAX = 1, 12
CPU_MIN, CPU_MAX = 100, 800          # millicores per pod
MEM_MIN, MEM_MAX = 64, 512           # MiB per pod
D_REPLICA, D_CPU, D_MEM = (-1, 0, 1), (-100, 0, 100), (-64, 0, 64)
ACTIONS = list(itertools.product(D_REPLICA, D_CPU, D_MEM))  # 27

# ---- trace families ----
# "azure" (default): cpu_norm is the organic Azure Functions 2019 per-minute
# invocation count (issue #11), in traces/, traces_heldout/, traces_train/.
# "bitbrains": the original Bitbrains VM CPU-utilization proxy for request rate,
# restored from the pre-#11 history into traces*_bitbrains/. mem_MB and io are
# identical across the two families; only cpu_norm differs. Selected process-wide
# through HV_TRACE_FAMILY so that every JointHVEnv built with default paths (the
# eval loops, MPC, the baselines) reads the same family as the run that drives it.
TRACE_FAMILIES = ("azure", "bitbrains")
_PHASE1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def trace_family() -> str:
    fam = os.environ.get("HV_TRACE_FAMILY", "azure")
    if fam not in TRACE_FAMILIES:
        raise ValueError(f"HV_TRACE_FAMILY={fam!r}, expected one of {TRACE_FAMILIES}")
    return fam


def trace_dir(kind: str = "traces") -> str:
    """kind is 'traces' (class representatives), 'traces_heldout' or 'traces_train'."""
    fam = trace_family()
    return os.path.join(_PHASE1_DIR, kind if fam == "azure" else f"{kind}_{fam}")


def family_suffix() -> str:
    """'' for azure, '_bitbrains' otherwise: for per-family model artifacts."""
    fam = trace_family()
    return "" if fam == "azure" else f"_{fam}"


def set_trace_family(family: str, tag: str | None = None) -> None:
    """Select the trace family for this process and check the output tag carries it.

    Bitbrains runs must have a 'bb' token in their tag and azure runs must not, so
    a job launched without the family flag cannot write azure numbers under a
    Bitbrains name (or the reverse). tag=None skips the check, for tools whose
    outputs are named per family by family_suffix() instead of by a tag.
    """
    if family not in TRACE_FAMILIES:
        raise SystemExit(f"--trace-family {family!r}, expected one of {TRACE_FAMILIES}")
    tokens = tag.split("_") if tag is not None else []
    if tag is not None and (("bb" in tokens) != (family == "bitbrains")
                            or (family == "bitbrains" and "azure" in tokens)):
        raise SystemExit(f"tag {tag!r} does not match --trace-family {family}: "
                         "bitbrains runs need a 'bb' tag token, azure runs must not have one")
    os.environ["HV_TRACE_FAMILY"] = family

# per-class axis intensity: the trace gives the temporal SHAPE, these set which
# resource axis the class actually stresses (so memory-bound stresses memory, not CPU). [CAL]
INTENSITY = {
    "cpu-bound":    {"cpu": 1.0,  "mem": 0.2},
    "memory-bound": {"cpu": 0.05, "mem": 1.0},
    "io-bound":     {"cpu": 0.5,  "mem": 0.3},
    "mixed":        {"cpu": 0.7,  "mem": 0.7},
}

DECISION_INTERVAL_S = 15
SCALE_DELAY_STEPS = 2                 # ~30s for a new replica to become ready
SLA_MEAN_LATENCY_MS = 20.0           # primary SLA (Azure-trace convention)

# ---- simulator calibration ----
# CPU axis fitted from the real-cluster sweep (phase1/calibration/fit_constants.py
# on sweep_v2.csv, latency MAPE 0.8%). The old smooth M/M/1 form L0/(1-rho) was
# wrong: measured latency stays flat (~5 ms) up to rho ~0.8 and then jumps to a
# ~400 ms plateau (CPU-limit throttling, timeout-bounded tail), so the model is
# a sigmoid knee between those two regimes. rho is demand-based and can exceed 1;
# at 3.27 mcores/rps, CPU_DEMAND_M = 4000 corresponds to a ~1200 rps peak.
CPU_DEMAND_M = 4000.0                 # required millicores at load_frac = 1
MEM_NEED_MI = 480.0                   # [CAL] per-pod working set at mem_frac=1 (< MEM_MAX so scaling CAN cover it)
L0_MS = 5.26                          # base service latency below the knee (fitted)
LAT_SAT_MS = 403.0                    # saturation plateau latency (fitted)
RHO_KNEE = 0.91                       # knee position (fitted)
KNEE_WIDTH = 0.015                    # knee sharpness (fitted)
SAT_DROP = 0.052                      # success loss on the plateau (fitted)
OVERLOAD_DROP = 0.2                   # [CAL] extra success slope beyond the measured range (rho > 1.31)
BETA_MEM = 6.0                        # [CAL] latency multiplier from memory pressure
OOM_DROP = 0.5                        # [CAL] success loss slope from memory pressure
LAT_CAP_MS = 5000.0                   # client timeout

# ---- reward weights (from PARAMETERS.md, tunable) ----
W_COST, W_CHURN, W_VIOL = 0.3, 0.5, 1.0


@dataclass
class SysState:
    replicas: int = 2
    target_replicas: int = 2
    cpu_req: int = 200
    mem_req: int = 128
    mem_need_now: float = 0.0         # current per-pod working set (for the ratchet)
    steps_to_ready: int = 0
    # What the cluster scheduler actually granted, as opposed to what this deployment
    # asked for. None means no scheduler is in the loop -- every single-function call
    # site -- and every expression below then reduces to the Phase 1 arithmetic
    # exactly, which is what keeps the multi-function N=1 case bit-identical.
    admitted_replicas: int | None = None
    cpu_granted: int | None = None
    mem_granted: int | None = None


class SimBackend:
    """Queueing simulator. Replays a class demand trace; models memory ratchet."""

    def __init__(self, trace: dict, rng: np.random.Generator, intensity: dict,
                 l0_ms: float | None = None, lat_sat_ms: float | None = None,
                 mem_need_mi: float | None = None):
        self.rng = rng
        # Per-function service profile. Phase 1 gives every function the same
        # latency curve and working set, so functions differ only in how much load
        # they draw; these let a function also differ in what a single call costs,
        # which is what actually distinguishes two real services. None keeps the
        # calibrated Phase 1 constants, so every existing call site is unchanged.
        self.l0 = L0_MS if l0_ms is None else float(l0_ms)
        self.lat_sat = LAT_SAT_MS if lat_sat_ms is None else float(lat_sat_ms)
        self.mem_need_mi = MEM_NEED_MI if mem_need_mi is None else float(mem_need_mi)
        self.cpu_int, self.mem_int = intensity["cpu"], intensity["mem"]
        cpu = np.nan_to_num(trace["cpu_norm"]).astype(np.float32)
        mem = np.nan_to_num(trace["mem_MB"]).astype(np.float32)
        # per-trace normalize to [0,1] (robust): each class then stresses its axis
        self.load = np.clip(cpu / (np.percentile(cpu, 99) or 1.0), 0, 1.5)
        self.memf = np.clip(mem / (np.percentile(mem, 99) or 1.0), 0, 1.0)
        self.n = len(self.load)
        self.t = 0
        self.s = SysState()

    def reset(self, start: int) -> None:
        self.t = start
        self.s = SysState()

    def apply(self, d_rep: int, d_cpu: int, d_mem: int) -> None:
        s = self.s
        s.target_replicas = int(np.clip(s.target_replicas + d_rep, REPLICA_MIN, REPLICA_MAX))
        if s.target_replicas != s.replicas and s.steps_to_ready == 0:
            s.steps_to_ready = SCALE_DELAY_STEPS
        s.cpu_req = int(np.clip(s.cpu_req + d_cpu, CPU_MIN, CPU_MAX))  # in-place: immediate
        new_mem = int(np.clip(s.mem_req + d_mem, MEM_MIN, MEM_MAX))
        # memory ratchet: an in-place *shrink* below the current working set is not
        # allowed (would need a restart). Holding (d_mem=0) does NOT auto-grow -> a
        # too-small mem_req stays too small and the pod suffers memory pressure.
        if d_mem < 0 and new_mem < s.mem_need_now:
            new_mem = int(np.clip(min(s.mem_req, np.ceil(s.mem_need_now)), MEM_MIN, MEM_MAX))
        s.mem_req = new_mem

    def step_physics(self) -> dict:
        s = self.s
        # advance replica readiness (actuation delay)
        if s.steps_to_ready > 0:
            s.steps_to_ready -= 1
            if s.steps_to_ready == 0:
                s.replicas = s.target_replicas
        if s.admitted_replicas is not None:
            # A scheduler is in the loop (multi-function). Losing capacity to a
            # neighbour takes effect at once -- a pod that cannot be placed is not
            # serving -- but *gaining* it still has to wait out SCALE_DELAY_STEPS,
            # so a Pending pod retries as room appears instead of popping in ready.
            granted = min(s.target_replicas, s.admitted_replicas)
            if granted < s.replicas or s.steps_to_ready == 0:
                s.replicas = granted
        load_frac = float(self.load[self.t % self.n])
        mem_frac = float(self.memf[self.t % self.n])
        s.mem_need_now = mem_frac * self.mem_need_mi * self.mem_int

        # the scheduler may defer an in-place growth that does not fit on the pod's
        # incumbent node, in which case the pod keeps running at its previous size
        cpu_eff = s.cpu_req if s.cpu_granted is None else s.cpu_granted
        mem_eff = s.mem_req if s.mem_granted is None else s.mem_granted
        required_cpu = load_frac * CPU_DEMAND_M * self.cpu_int
        total_cpu = max(s.replicas * cpu_eff, 1.0)
        rho = required_cpu / total_cpu
        knee = 1.0 / (1.0 + np.exp(-(rho - RHO_KNEE) / KNEE_WIDTH))
        lat_cpu = self.l0 + (self.lat_sat - self.l0) * knee
        mem_pressure = max(0.0, (s.mem_need_now - mem_eff) / max(mem_eff, 1))
        latency = min(lat_cpu * (1.0 + BETA_MEM * mem_pressure), LAT_CAP_MS)
        success = 1.0 - min(0.95, SAT_DROP * knee + OVERLOAD_DROP * max(0.0, rho - 1.31)
                            + OOM_DROP * mem_pressure)

        self.t += 1
        return dict(load_frac=load_frac, rho=rho, latency=latency, success=success,
                    mem_pressure=mem_pressure, cpu_util=min(rho, 1.5))


class JointHVEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self, traces_dir: str | None = None, workload: str = "memory-bound",
                 episode_len: int = 200, backend: str = "sim", seed: int | None = None,
                 backend_kwargs: dict | None = None, trace_paths: dict | None = None,
                 trace_pool: dict | None = None, action_mode: str = "discrete",
                 reward_fn: str = "default", pandey_beta: float = 0.5,
                 scaling_mode: str = "joint", w_cost: float | None = None,
                 w_churn: float | None = None, w_viol: float | None = None,
                 profile: dict | None = None):
        # profile = this function's measured service characteristics
        # (l0_ms, lat_sat_ms, sla_ms, mem_need_mi), see phase2/derive_profiles.py.
        # None keeps the single calibrated profile every Phase 1 result was produced
        # with, so those results are unaffected.
        profile = profile or {}
        self.sla_ms = float(profile.get("sla_ms", SLA_MEAN_LATENCY_MS))
        self._profile = profile
        self.traces_dir = traces_dir or trace_dir("traces")
        self.episode_len = episode_len
        self.backend_kind = backend
        self.backend_kwargs = backend_kwargs or {}
        self.rng = np.random.default_rng(seed)
        # action_mode="continuous": Box(-1,1)^3 in place of the Discrete(27) every
        # other algo in this project uses -- built for the Pandey et al. SOTA
        # reproduction (thread #SOTA-1), whose action design is continuous
        # target-CPU/mem + a replica delta, which DDPG requires (DDPG has no
        # discrete-action formulation). Each dim maps to the SAME per-step
        # magnitude budget as the discrete env (+-1 replica, +-100m cpu, +-64Mi
        # mem) so this isolates representation/algorithm, not a bigger action
        # budget, from the discrete comparators.
        self.action_mode = action_mode
        # scaling_mode restricts the discrete action axis for the H-only / V-only
        # ablations: "horizontal" keeps only the replica delta (CPU/mem pinned at
        # the shared SysState init), "vertical" keeps only in-place CPU/mem resize
        # (replicas pinned). "joint" is the full 27-action space every headline
        # result uses. Obs and reward are identical across modes, so this isolates
        # the action space and nothing else -- the controlled ablation for the
        # "joint scaling is necessary" claim (quantifies the Phase 0 coupling story).
        self.scaling_mode = scaling_mode
        # reward_fn="pandey": r_t = beta*SLO_score_t + (1-beta)*cost_efficiency_t,
        # reproducing the reward formula in Sec 3.2 of "Reinforcement Learning-Based
        # Autoscaling for Cost and Performance Optimization in Kubernetes Clusters"
        # (Pandey, 2026) in place of this env's own W_COST/W_CHURN/W_VIOL reward.
        self.reward_fn = reward_fn
        self.pandey_beta = pandey_beta
        # reward-weight overrides for the cost/SLA Pareto sweep (#20). None keeps the
        # PARAMETERS.md defaults (W_COST/W_CHURN/W_VIOL), so every existing call site
        # and every committed result is unchanged. Only the "default" reward reads
        # these; _reward_pandey has its own formula.
        self.w_cost = W_COST if w_cost is None else float(w_cost)
        self.w_churn = W_CHURN if w_churn is None else float(w_churn)
        self.w_viol = W_VIOL if w_viol is None else float(w_viol)
        # workload can be a single class or "all" (generalist: random class per episode)
        self._wls = list(INTENSITY) if workload == "all" else [workload]
        # trace_paths overrides the default `<class>.npz` lookup with an explicit file
        # per class -> lets held-out eval replay a VM the model never trained on
        # (same class, different data-generating VM). Falls back to traces_dir otherwise.
        trace_paths = trace_paths or {}
        self._traces = {w: dict(np.load(trace_paths.get(w, os.path.join(self.traces_dir, f"{w}.npz"))))
                        for w in self._wls}
        # trace_pool = domain randomization over VMs: {class: [path, ...]}. When set,
        # each reset resamples the active trace for the chosen class from a preloaded
        # pool, so the policy trains across many same-class VMs instead of one rep.
        self._pool = {w: [dict(np.load(p)) for p in trace_pool[w]]
                      for w in self._wls if trace_pool and w in trace_pool} if trace_pool else {}
        self._cur_wl = self._wls[0]
        self.workload = self._cur_wl
        if scaling_mode == "horizontal":
            self._actions = [a for a in ACTIONS if a[1] == 0 and a[2] == 0]  # replica delta only
        elif scaling_mode == "vertical":
            self._actions = [a for a in ACTIONS if a[0] == 0]                # in-place resize only
        elif scaling_mode == "joint":
            self._actions = ACTIONS
        else:
            raise ValueError(f"unknown scaling_mode {scaling_mode!r}")
        if self.action_mode == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self._actions))
        # 12 normalized observation features
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32)
        self._prev_lat = 0.0
        self._prev_load = 0.0
        self._t_since_action = 0
        self._backend: SimBackend | None = None
        self._step = 0

    # ---- helpers ----
    def _make_backend(self):
        intensity = INTENSITY.get(self._cur_wl, {"cpu": 0.7, "mem": 0.7})
        trace = self._traces[self._cur_wl]
        if self.backend_kind == "sim":
            return SimBackend(trace, self.rng, intensity,
                              l0_ms=self._profile.get("l0_ms"),
                              lat_sat_ms=self._profile.get("lat_sat_ms"),
                              mem_need_mi=self._profile.get("mem_need_mi"))
        from envs.real_backend import RealClusterBackend  # lazy: keeps kubectl out of training
        return RealClusterBackend(trace, self.rng, intensity, **self.backend_kwargs)

    def _obs(self, m: dict) -> np.ndarray:
        s = self._backend.s
        pending = s.target_replicas - s.replicas
        # Latency features are scaled by this function's OWN SLO rather than by a
        # global constant, so a fast and a slow service present the same normalized
        # picture to the policy. At the Phase 1 SLO of 20ms these reduce to exactly
        # the original /20 and /200 divisors, so nothing about Phase 1 changes.
        sla = self.sla_ms
        lat_trend = np.tanh((m["latency"] - self._prev_lat) / sla) * 0.5 + 0.5
        load_trend = np.tanh((m["load_frac"] - self._prev_load) * 2.0) * 0.5 + 0.5
        headroom = np.clip((sla - m["latency"]) / sla, -1, 1) * 0.5 + 0.5
        return np.array([
            np.clip(m["load_frac"], 0, 1),
            np.clip(m["latency"] / (10.0 * sla), 0, 1),
            np.clip(m["cpu_util"], 0, 1),
            (s.replicas - REPLICA_MIN) / (REPLICA_MAX - REPLICA_MIN),
            (s.cpu_req - CPU_MIN) / (CPU_MAX - CPU_MIN),
            (s.mem_req - MEM_MIN) / (MEM_MAX - MEM_MIN),
            np.clip(s.mem_need_now / getattr(self._backend, "mem_need_mi", MEM_NEED_MI), 0, 1),
            np.clip((pending + REPLICA_MAX) / (2 * REPLICA_MAX), 0, 1),
            lat_trend, load_trend, headroom,
            np.clip(self._t_since_action / 10.0, 0, 1),
        ], dtype=np.float32)

    def _reward(self, m: dict, changed: int) -> float:
        if self.reward_fn == "pandey":
            return self._reward_pandey(m)
        s = self._backend.s
        sla_ok = m["latency"] < self.sla_ms and m["success"] >= 0.99
        cost = (s.replicas * s.cpu_req / (REPLICA_MAX * CPU_MAX)
                + s.replicas * s.mem_req / (REPLICA_MAX * MEM_MAX)) / 2.0
        viol = max(0.0, (m["latency"] - self.sla_ms) / self.sla_ms) + (1.0 - m["success"])
        churn = changed / 3.0
        return (1.0 if sla_ok else 0.0) - self.w_cost * cost - self.w_churn * churn - self.w_viol * viol

    def _reward_pandey(self, m: dict) -> float:
        """r_t = beta*SLO_score_t + (1-beta)*cost_efficiency_t (Pandey Sec 3.2).
        SLO_score: the paper defines it as "the fraction of requests below a
        latency threshold"; our sim reports one scalar latency/success per
        control interval rather than a per-request stream, so it collapses to
        the same binary SLA check used elsewhere in this env. cost_efficiency:
        the paper states "the inverse of allocated vCPU x RAM" -- taken
        literally, 1/(cpu*mem) diverges as allocation shrinks, which no real
        implementation would leave unbounded, so we use the bounded proxy
        1 - normalized_alloc_fraction (same direction: reward for using less
        combined CPU*mem, capped in [0,1] to stay commensurate with SLO_score)."""
        s = self._backend.s
        slo_score = 1.0 if (m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99) else 0.0
        alloc_frac = ((s.replicas * s.cpu_req) / (REPLICA_MAX * CPU_MAX)) * \
                     ((s.replicas * s.mem_req) / (REPLICA_MAX * MEM_MAX))
        cost_efficiency = 1.0 - alloc_frac
        return self.pandey_beta * slo_score + (1.0 - self.pandey_beta) * cost_efficiency

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._cur_wl = self._wls[int(self.rng.integers(len(self._wls)))]
        self.workload = self._cur_wl
        pool = self._pool.get(self._cur_wl)
        if pool:  # domain randomization: pick a VM from this class's pool for the episode
            self._traces[self._cur_wl] = pool[int(self.rng.integers(len(pool)))]
        self._backend = self._make_backend()
        if options and options.get("start") is not None:
            start = int(options["start"]) % max(1, self._backend.n - self.episode_len - 1)
        else:
            start = int(self.rng.integers(0, max(1, self._backend.n - self.episode_len - 1)))
        self._backend.reset(start)
        self._step = 0
        self._t_since_action = 0
        m = self._backend.step_physics()
        self._prev_lat, self._prev_load = m["latency"], m["load_frac"]
        return self._obs(m), {"metrics": m}

    def step(self, action):
        if self.action_mode == "continuous":
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            d_rep = int(round(float(a[0])))     # pods aren't fractional -> round to {-1,0,1}
            d_cpu = float(a[1]) * 100.0          # continuous within +-100m (matches discrete budget)
            d_mem = float(a[2]) * 64.0           # continuous within +-64Mi
            changed = (d_rep != 0) + (abs(d_cpu) > 1.0) + (abs(d_mem) > 1.0)
        else:
            d_rep, d_cpu, d_mem = self._actions[int(action)]
            changed = (d_rep != 0) + (d_cpu != 0) + (d_mem != 0)
        self._t_since_action = 0 if changed else self._t_since_action + 1
        self._backend.apply(d_rep, d_cpu, d_mem)
        m = self._backend.step_physics()
        reward = self._reward(m, changed)
        obs = self._obs(m)
        self._prev_lat, self._prev_load = m["latency"], m["load_frac"]
        self._step += 1
        truncated = self._step >= self.episode_len
        return obs, float(reward), False, truncated, {"metrics": m}
