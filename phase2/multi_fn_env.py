"""MultiFnEnv: N functions, each with its own controller, sharing one node pool.

Phase 1 measures one function managed by one controller in isolation, and its env
has no cluster capacity model at all -- a controller can always get the replicas
it asks for. This env adds the missing piece and nothing else: a kube-scheduler-
style admission step between `apply()` and `step_physics()`, so that

    desired != granted

whenever aggregate demand exceeds the node pool. Every calibrated constant, the
latency knee, the memory ratchet, the observation vector and the reward come from
phase1/envs/joint_hv_env.py unchanged -- this file composes N JointHVEnv instances
and drives their internals, so an N=1 run at infinite capacity is bit-identical to
Phase 1 (see smoke_test_multifn.py).

Starvation therefore needs no new physics: a function that cannot place its pods
just has a smaller denominator in rho, which slides it up the already-fitted knee.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# phase1 is the single source of physics; import it rather than duplicating it.
_P1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase1"))
if _P1 not in sys.path:
    sys.path.insert(0, _P1)

from envs.joint_hv_env import (  # noqa: E402
    ACTIONS, JointHVEnv, SLA_MEAN_LATENCY_MS,
)

NODES = 3                 # matches the 3-node k3s testbed
NODE_MEM_MI = 8192        # memory is deliberately non-binding by default (see #22)

# One function per workload class: a realistic co-located mix, and each function's
# row stays directly comparable to its own class row in comparison_azure.csv.
# These are genuinely different functions, not duplicates -- the classes span a 20x
# spread in peak CPU demand (cpu-bound 4000m vs memory-bound 200m) and a 5x spread
# in memory, each driven by its own VM trace.
#
# Order matters: the N-sweep takes nested prefixes, so N=2 is the most contrasting
# pair (the biggest CPU consumer against the smallest), and growing N only ever
# *adds* a function rather than swapping the mix. N is then the only variable.
DEFAULT_CLASSES = ["cpu-bound", "memory-bound", "io-bound", "mixed"]


def _load_profiles(mode: str) -> tuple[dict, dict]:
    """(per-class profile, per-class trace override) for the requested mode.

    A class whose Azure function had to be reassigned (no memory record for the
    original app) reads its trace from phase2/traces/, never from phase1/traces/ --
    Phase 1's own inputs are left exactly as the committed results were produced.
    """
    if mode == "legacy":
        return {}, {}
    if mode != "azure":
        raise ValueError(f"unknown profiles mode {mode!r}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fn_profiles.json")
    if not os.path.exists(path):
        raise SystemExit("fn_profiles.json missing -- run: python derive_profiles.py --rebuild")
    with open(path) as f:
        blob = json.load(f)
    prof, tpaths = {}, {}
    for cls, p in blob["profiles"].items():
        prof[cls] = {k: p[k] for k in ("l0_ms", "lat_sat_ms", "sla_ms", "mem_need_mi")}
        if p.get("reassigned"):
            t = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", f"{cls}.npz")
            if not os.path.exists(t):
                raise SystemExit(f"{cls} was reassigned but {t} is missing -- "
                                 "run: python derive_profiles.py --rebuild")
            tpaths[cls] = t
    return prof, tpaths


class ClusterScheduler:
    """kube-scheduler's default NodeResourcesFit/LeastAllocated, deliberately dumb.

    Running pods keep their node. New pods are placed one at a time onto the
    least-allocated node with room. There is no priority and no preemption: the
    scheduler must not be capable of anything that could be mistaken for
    coordination between controllers, because coordination is the thing under
    study. It produces exactly two denial modes, both real Kubernetes behaviours:

      Pending         a new pod fits on no node  -> admitted_replicas < target
      Deferred resize an in-place *growth* does not fit on the pod's incumbent
                      node -> cpu_granted stays at the previous size

    The second is the one worth having. Node fragmentation silently blocks VPA
    growth: the same H+V coupling failure phase0 characterized, but caused by the
    cluster rather than by two controllers fighting each other.
    """

    def __init__(self, n_fns: int, n_nodes: int = NODES, node_cpu_m: float | None = None,
                 node_mem_mi: float = NODE_MEM_MI, strict: bool = False):
        # node_cpu_m=None means infinite capacity -> the scheduler is inert and the
        # env takes the exact Phase 1 code path (granted fields stay None).
        self.enabled = node_cpu_m is not None
        self.n_fns, self.n_nodes = n_fns, n_nodes
        self.node_cpu = float(node_cpu_m) if self.enabled else float("inf")
        self.node_mem = float(node_mem_mi)
        self.strict = strict
        self.placed = [[0] * n_fns for _ in range(n_nodes)]
        self.gcpu = [0] * n_fns
        self.gmem = [0] * n_fns
        self.denied = [0] * n_fns      # pods left Pending this step
        self.deferred = [False] * n_fns  # resize refused this step
        self._tick = 0

    # ---- capacity bookkeeping ----
    def _alloc(self, node: int) -> tuple[float, float]:
        c = sum(self.placed[node][f] * self.gcpu[f] for f in range(self.n_fns))
        m = sum(self.placed[node][f] * self.gmem[f] for f in range(self.n_fns))
        return c, m

    def _fits(self, node: int, cpu: float, mem: float) -> bool:
        c, m = self._alloc(node)
        return c + cpu <= self.node_cpu + 1e-9 and m + mem <= self.node_mem + 1e-9

    def _place_one(self, fn: int) -> bool:
        """LeastAllocated: lowest CPU utilisation among nodes that fit."""
        cands = [n for n in range(self.n_nodes)
                 if self._fits(n, self.gcpu[fn], self.gmem[fn])]
        if not cands:
            return False
        best = min(cands, key=lambda n: (self._alloc(n)[0] / self.node_cpu, n))
        self.placed[best][fn] += 1
        return True

    def _remove(self, fn: int, k: int) -> None:
        """Scale-in frees the most-allocated nodes first (best defragmentation)."""
        for _ in range(k):
            hosts = [n for n in range(self.n_nodes) if self.placed[n][fn] > 0]
            if not hosts:
                return
            worst = max(hosts, key=lambda n: (self._alloc(n)[0], n))
            self.placed[worst][fn] -= 1

    def _total(self, fn: int) -> int:
        return sum(self.placed[n][fn] for n in range(self.n_nodes))

    # ---- the admission step ----
    def reset(self, states) -> None:
        self.placed = [[0] * self.n_fns for _ in range(self.n_nodes)]
        self.gcpu = [s.cpu_req for s in states]
        self.gmem = [s.mem_req for s in states]
        self._tick = 0
        if not self.enabled:
            return
        for fn, s in enumerate(states):
            for _ in range(s.replicas):
                self._place_one(fn)
        self._writeback(states)

    def schedule(self, states) -> None:
        if not self.enabled:
            return  # capacity never binds -> leave the granted fields None
        self._tick += 1
        self.denied = [0] * self.n_fns
        self.deferred = [False] * self.n_fns
        # Rotate which function the scheduler considers first. A fixed order would
        # hand function 0 a systematic advantage and show up as an indexing
        # artefact in the fairness numbers rather than a property of the workload.
        order = [(i + self._tick) % self.n_fns for i in range(self.n_fns)]

        for fn in order:  # vertical first: resizes change what a pod costs
            self._resize(fn, states[fn])
        for fn in order:  # then horizontal
            self._scale(fn, states[fn])

        self._writeback(states)
        if self.strict:
            self._assert_conserved(states)

    def _resize(self, fn: int, s) -> None:
        want_c, want_m = float(s.cpu_req), float(s.mem_req)
        d_c, d_m = want_c - self.gcpu[fn], want_m - self.gmem[fn]
        if d_c <= 0 and d_m <= 0:
            self.gcpu[fn], self.gmem[fn] = want_c, want_m  # shrink always lands
            return
        # a growth must fit on *every* node already hosting this function's pods
        for n in range(self.n_nodes):
            k = self.placed[n][fn]
            if k and not self._fits(n, k * max(d_c, 0.0), k * max(d_m, 0.0)):
                self.deferred[fn] = True
                return
        self.gcpu[fn], self.gmem[fn] = want_c, want_m

    def _scale(self, fn: int, s) -> None:
        have, want = self._total(fn), int(s.target_replicas)
        if want < have:
            self._remove(fn, have - want)
            return
        for _ in range(want - have):
            if not self._place_one(fn):
                self.denied[fn] = want - self._total(fn)
                break

    def _writeback(self, states) -> None:
        for fn, s in enumerate(states):
            s.admitted_replicas = self._total(fn)
            s.cpu_granted = self.gcpu[fn]
            s.mem_granted = self.gmem[fn]

    def _assert_conserved(self, states) -> None:
        for n in range(self.n_nodes):
            c, m = self._alloc(n)
            assert c <= self.node_cpu + 1e-6, f"node {n} cpu {c} > {self.node_cpu}"
            assert m <= self.node_mem + 1e-6, f"node {n} mem {m} > {self.node_mem}"
        for s in states:
            assert s.admitted_replicas <= s.target_replicas

    # ---- signals the env reports (controllers are free to ignore them) ----
    def cluster_cpu_alloc_frac(self) -> float:
        if not self.enabled:
            return 0.0
        total = sum(self._alloc(n)[0] for n in range(self.n_nodes))
        return total / (self.node_cpu * self.n_nodes)


class MultiFnEnv:
    """N functions stepped in lockstep against one shared node pool.

    Not a gym.Env: this is the evaluation harness for the #22 ablation, where each
    function is driven by its *own* controller, so a single flat action space would
    be the wrong shape. `step` takes one action per function. (The sequential
    single-agent wrapper an RL algorithm would need is a separate concern and is
    not built here, because #22 trains nothing.)

    Observations are the unmodified Phase 1 12-feature vector, which is what makes
    the committed single-function checkpoints droppable straight in.
    """

    def __init__(self, classes: list[str] | None = None, node_cpu_m: float | None = None,
                 n_nodes: int = NODES, node_mem_mi: float = NODE_MEM_MI,
                 episode_len: int = 200, seed: int | None = None, strict: bool = False,
                 traces_dir: str | None = None, profiles: str = "legacy"):
        # profiles="legacy": every function shares the one calibrated service profile,
        #   so functions differ only in how much load they draw. This is the Phase 1
        #   model, and it is what the bit-exactness gates check against.
        # profiles="azure": each function additionally gets the measured service time,
        #   SLO and working set of the real Azure function driving its trace
        #   (derive_profiles.py), which is a ~23x spread in base service time.
        # Separating the two keeps contention and heterogeneity independently
        # attributable instead of confounded in one number.
        self.classes = list(classes or DEFAULT_CLASSES)
        self.n = len(self.classes)
        self.episode_len = episode_len
        self.profiles_mode = profiles
        prof, tpaths = _load_profiles(profiles)
        self.envs = [JointHVEnv(workload=c, episode_len=episode_len,
                                seed=(seed or 0) + 1000 * i, traces_dir=traces_dir,
                                profile=prof.get(c),
                                trace_paths={c: tpaths[c]} if c in tpaths else None)
                     for i, c in enumerate(self.classes)]
        self.sched = ClusterScheduler(self.n, n_nodes=n_nodes, node_cpu_m=node_cpu_m,
                                      node_mem_mi=node_mem_mi, strict=strict)
        self._step = 0

    @property
    def states(self):
        return [e._backend.s for e in self.envs]

    def reset(self, seed: int | None = None, start: int | None = None):
        out = []
        for i, e in enumerate(self.envs):
            o, info = e.reset(seed=None if seed is None else seed + 1000 * i,
                              options={"start": start} if start is not None else None)
            out.append((o, info))
        self.sched.reset(self.states)
        self._step = 0
        # the scheduler may already have denied the initial replicas, so recompute
        # each function's first observation against what it actually got
        if self.sched.enabled:
            out = [(e._obs(info["metrics"]), info) for e, (_, info) in zip(self.envs, out)]
        return [o for o, _ in out], [self._info(i, inf) for i, (_, inf) in enumerate(out)]

    def step(self, actions: list[int]):
        changed = []
        for e, a in zip(self.envs, actions):
            d_rep, d_cpu, d_mem = ACTIONS[int(a)]
            changed.append((d_rep != 0) + (d_cpu != 0) + (d_mem != 0))
            e._t_since_action = 0 if changed[-1] else e._t_since_action + 1
            e._backend.apply(d_rep, d_cpu, d_mem)

        self.sched.schedule(self.states)

        obs, rews, infos = [], [], []
        for i, (e, ch) in enumerate(zip(self.envs, changed)):
            m = e._backend.step_physics()
            rews.append(float(e._reward(m, ch)))
            obs.append(e._obs(m))
            e._prev_lat, e._prev_load = m["latency"], m["load_frac"]
            e._step += 1
            infos.append(self._info(i, {"metrics": m}))
        self._step += 1
        trunc = self._step >= self.episode_len
        return obs, rews, False, trunc, infos

    def _info(self, i: int, info: dict) -> dict:
        m = dict(info["metrics"])
        s = self.states[i]
        # contention signals ride along as extra metrics keys. Every Phase 1
        # controller ignores unknown keys, which is exactly right for #22: they
        # were trained with no concept of a neighbour.
        m["denied_replicas"] = self.sched.denied[i] if self.sched.enabled else 0
        m["resize_deferred"] = bool(self.sched.deferred[i]) if self.sched.enabled else False
        m["cluster_cpu_alloc_frac"] = self.sched.cluster_cpu_alloc_frac()
        # judged against THIS function's SLO: in azure-profile mode a 237ms service
        # and a 10ms service cannot share one latency target
        m["sla_ms"] = self.envs[i].sla_ms
        m["sla_ok"] = m["latency"] < m["sla_ms"] and m["success"] >= 0.99
        m["cost_vcpu"] = s.replicas * (s.cpu_granted if s.cpu_granted is not None
                                       else s.cpu_req) / 1000.0
        return {"metrics": m}


def jain(xs) -> float:
    """Jain fairness index over per-function SLA. 1.0 = perfectly equal; a mean
    alone hides one totally starved function, which at N=4 is the failure that
    matters."""
    xs = np.asarray(xs, dtype=float)
    d = float(np.sum(xs ** 2)) * len(xs)
    return float(np.sum(xs) ** 2 / d) if d > 0 else 1.0
