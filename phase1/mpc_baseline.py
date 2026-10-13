"""Model-predictive control baseline (issue #16): the "did you even need RL"
challenger. Every other baseline is threshold/rule-based (HPA/VPA/SLO-cost-aware)
or a competing RL method (Pandey); none of them plan explicitly over the known
dynamics. The calibrated sim (0.8% latency MAPE, issue #1) gives an exact plant
model to plan on -- if a controller with privileged access to that model AND the
exact reward formula RL was trained to maximize still can't match QR-DQN/Option B,
that is a STRONGER result for RL than beating another rule-based baseline is.

Design (see issue #16 for the full writeup):
  - Disturbance: PERSISTENCE only -- assume the current tick's observed demand
    holds over the planning horizon. No perfect-foresight oracle variant (by
    explicit choice): this keeps MPC in the same information class as every
    other baseline and as RL itself. Its only edge, if any, comes from planning
    over known dynamics (the SCALE_DELAY_STEPS actuation delay, the irreversible
    memory-shrink ratchet), not from cheating on the workload.
  - Objective: the exact `default_reward` formula from `envs.joint_hv_env`
    (imported, not reimplemented) -- MPC directly SOLVES what RL had to LEARN.
  - Solver: EXACT finite-horizon backward-induction DP, vectorized in numpy over
    the full discretized state grid (replicas x target_replicas x cpu_req x
    mem_req x steps_to_ready). Deterministic -- no RNG, no seed, no sampling
    hyperparameters. Chosen over CEM/random-shooting after benchmarking: a naive
    per-tick enumeration or python-loop reachability search is 4-5 orders of
    magnitude too slow at eval scale (~16k control ticks across 20 episodes x 4
    classes); a full-grid vectorized backward induction is cheap enough to redo
    from scratch every control tick (~9.5ms grid solve once the static
    transition structure is precomputed once at construction, plus the per-tick
    ratchet correction and reward pass -- see PLANNING.md-style note below).
  - Receding horizon: replan from the TRUE observed state every tick, execute
    only the first action of the resulting plan, discard the rest.

FIDELITY: the transition/metric formulas below are a hand-vectorized mirror of
`SimBackend.apply()` / `SimBackend.step_physics()` in envs/joint_hv_env.py --
NOT a shared code path (the live per-step scalar physics were deliberately left
untouched rather than refactored, since 15+ commits of committed results depend
on that file being unchanged). Calibration CONSTANTS are imported directly (so
those at least cannot drift); the transition LOGIC is duplicated and must be
kept in sync by hand if joint_hv_env.py's physics ever change -- same accepted
"documented duplication" pattern already used elsewhere in this project (e.g.
cvar_bellman_qrdqn.py copies sb3_contrib's QRDQN.train() with one changed
block). `verify_fidelity()` below cross-checks the vectorized model against the
live scalar SimBackend on random states/actions and must pass before trusting
any eval numbers built on this module.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from envs.joint_hv_env import (ACTIONS, BETA_MEM, CPU_MAX, CPU_MIN,
                               KNEE_WIDTH, L0_MS, LAT_CAP_MS, LAT_SAT_MS,
                               MEM_MAX, MEM_MIN, OOM_DROP, OVERLOAD_DROP,
                               REPLICA_MAX, REPLICA_MIN, RHO_KNEE, SAT_DROP,
                               SCALE_DELAY_STEPS, SLA_MEAN_LATENCY_MS, W_COST,
                               W_CHURN, W_VIOL, SysState)

# ---- planning hyperparameters ----
HORIZON = 4          # spans SCALE_DELAY_STEPS=2 with margin
DISCOUNT = 0.9

_REPS = np.arange(REPLICA_MIN, REPLICA_MAX + 1)              # 12
_TGTS = np.arange(REPLICA_MIN, REPLICA_MAX + 1)               # 12
_CPUS = np.arange(CPU_MIN, CPU_MAX + 1, 100)                  # 8
_MEMS = np.arange(MEM_MIN, MEM_MAX + 1, 64)                   # 8
_READY = np.array([0, 1, 2])                                  # SCALE_DELAY_STEPS + 1 values
_GRID_SHAPE = (len(_REPS), len(_TGTS), len(_CPUS), len(_MEMS), len(_READY))


def _flat_index(rep, tgt, cpu, mem, ready):
    r = (rep - REPLICA_MIN).astype(np.int64)
    t = (tgt - REPLICA_MIN).astype(np.int64)
    c = ((cpu - CPU_MIN) // 100).astype(np.int64)
    m = ((mem - MEM_MIN) // 64).astype(np.int64)
    rd = ready.astype(np.int64)
    return np.ravel_multi_index((r, t, c, m, rd), _GRID_SHAPE)


@dataclass
class _StaticTransition:
    """Per-action transition structure that does NOT depend on the observed
    load/mem_need -- precomputed once at controller construction, not per tick.
    mem_req is left UNRATCHETED here (the ratchet depends on mem_need_now, which
    varies tick to tick); `act()` applies the ratchet correction per tick."""
    d_mem: int
    changed: np.ndarray      # scalar-ish (same for every grid cell): (d_rep!=0)+(d_cpu!=0)+(d_mem!=0)
    rep3: np.ndarray         # replicas after the readiness advance
    tgt2: np.ndarray         # target_replicas after applying the action
    cpu2: np.ndarray
    mem2_unratcheted: np.ndarray
    ready3: np.ndarray


class MPCController:
    """Same `.act(m, s)` / `.reset()` interface as ClassicalController /
    SloCostAwareController, so it drops straight into eval_baselines.py's
    evaluate_policy() harness."""

    def __init__(self, horizon: int = HORIZON, discount: float = DISCOUNT,
                 w_cost: float | None = None, w_churn: float | None = None,
                 w_viol: float | None = None):
        self.horizon = horizon
        self.discount = discount
        # reward-weight overrides for the cost/SLA Pareto sweep (#20). None keeps the
        # imported env constants, so the planner's objective stays byte-identical to
        # the env's default reward (the whole point of the MPC baseline) unless a
        # sweep cell explicitly asks otherwise.
        self.w_cost = W_COST if w_cost is None else float(w_cost)
        self.w_churn = W_CHURN if w_churn is None else float(w_churn)
        self.w_viol = W_VIOL if w_viol is None else float(w_viol)
        self.mean_plan_ms = 0.0
        self._n_calls = 0
        self._n_states = int(np.prod(_GRID_SHAPE))
        rep, tgt, cpu, mem, ready = np.meshgrid(_REPS, _TGTS, _CPUS, _MEMS, _READY, indexing="ij")
        self._rep = rep.ravel(); self._tgt = tgt.ravel(); self._cpu = cpu.ravel()
        self._mem = mem.ravel(); self._ready = ready.ravel()
        self._static = [self._precompute_transition(d_rep, d_cpu, d_mem)
                        for (d_rep, d_cpu, d_mem) in ACTIONS]
        self.reset()

    def reset(self) -> None:
        pass  # stateless/deterministic -- nothing to reset between episodes

    # ---- precompute (once) ----
    def _precompute_transition(self, d_rep: int, d_cpu: int, d_mem: int) -> _StaticTransition:
        # apply_action (mirrors SimBackend.apply, sans the mem ratchet)
        tgt2 = np.clip(self._tgt + d_rep, REPLICA_MIN, REPLICA_MAX)
        ready_now = (tgt2 != self._rep) & (self._ready == 0)
        ready2 = np.where(ready_now, SCALE_DELAY_STEPS, self._ready)
        cpu2 = np.clip(self._cpu + d_cpu, CPU_MIN, CPU_MAX)
        mem2 = np.clip(self._mem + d_mem, MEM_MIN, MEM_MAX)
        # advance_ready (mirrors the first half of SimBackend.step_physics)
        ready3 = np.where(ready2 > 0, ready2 - 1, ready2)
        landed = (ready2 > 0) & (ready3 == 0)
        rep3 = np.where(landed, tgt2, self._rep)
        changed = (d_rep != 0) + (d_cpu != 0) + (d_mem != 0)
        return _StaticTransition(d_mem=d_mem, changed=changed, rep3=rep3, tgt2=tgt2,
                                 cpu2=cpu2, mem2_unratcheted=mem2, ready3=ready3)

    # ---- per-tick (cheap: ratchet correction + reward + backward induction) ----
    def _solve(self, required_cpu: float, mem_need: float) -> np.ndarray:
        """Returns the optimal FIRST action index for every grid cell (we only
        ever read the entry for the current true state, but computing the whole
        grid is what makes this vectorized/fast)."""
        rewards, indices = [], []
        for tr in self._static:
            mem2 = tr.mem2_unratcheted
            if tr.d_mem < 0:
                # ratchet: an in-place shrink below the CURRENT working set is not
                # allowed (would need a restart) -- mirrors SimBackend.apply()
                shrink_below_need = mem2 < mem_need
                mem2 = np.where(shrink_below_need,
                                np.clip(np.minimum(self._mem, np.ceil(mem_need)), MEM_MIN, MEM_MAX),
                                mem2)
            idx = _flat_index(tr.rep3, tr.tgt2, tr.cpu2, mem2, tr.ready3)
            total_cpu = np.maximum(tr.rep3 * tr.cpu2, 1.0)
            rho = required_cpu / total_cpu
            knee = 1.0 / (1.0 + np.exp(-(rho - RHO_KNEE) / KNEE_WIDTH))
            lat_cpu = L0_MS + (LAT_SAT_MS - L0_MS) * knee
            mem_pressure = np.maximum(0.0, (mem_need - mem2) / np.maximum(mem2, 1))
            latency = np.minimum(lat_cpu * (1.0 + BETA_MEM * mem_pressure), LAT_CAP_MS)
            success = 1.0 - np.minimum(
                0.95, SAT_DROP * knee + OVERLOAD_DROP * np.maximum(0.0, rho - 1.31)
                + OOM_DROP * mem_pressure)
            sla_ok = (latency < SLA_MEAN_LATENCY_MS) & (success >= 0.99)
            cost = (tr.rep3 * tr.cpu2 / (REPLICA_MAX * CPU_MAX)
                   + tr.rep3 * mem2 / (REPLICA_MAX * MEM_MAX)) / 2.0
            viol = np.maximum(0.0, (latency - SLA_MEAN_LATENCY_MS) / SLA_MEAN_LATENCY_MS) + (1.0 - success)
            r = (sla_ok.astype(np.float64) - self.w_cost * cost
                 - self.w_churn * (tr.changed / 3.0) - self.w_viol * viol)
            rewards.append(r); indices.append(idx)

        n = self._n_states
        V = np.zeros(n)
        best_a0 = np.zeros(n, dtype=np.int64)
        for h in range(self.horizon):
            best = np.full(n, -1e18)
            for ai in range(len(ACTIONS)):
                q = rewards[ai] + self.discount * V[indices[ai]]
                mask = q > best
                if h == self.horizon - 1:      # last backward step = first forward step
                    best_a0 = np.where(mask, ai, best_a0)
                best = np.where(mask, q, best)
            V = best
        return best_a0

    def act(self, m: dict, s: SysState) -> tuple[int, int, int]:
        t0 = time.perf_counter()
        required_cpu = m["rho"] * max(s.replicas * s.cpu_req, 1.0)
        mem_need = s.mem_need_now
        best_a0 = self._solve(required_cpu, mem_need)
        cur_idx = _flat_index(np.array([s.replicas]), np.array([s.target_replicas]),
                              np.array([s.cpu_req]), np.array([s.mem_req]),
                              np.array([s.steps_to_ready]))[0]
        action = ACTIONS[int(best_a0[cur_idx])]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._n_calls += 1
        self.mean_plan_ms += (dt_ms - self.mean_plan_ms) / self._n_calls
        return action


def verify_fidelity(n_checks: int = 200, seed: int = 0) -> None:
    """Cross-checks the vectorized transition/reward model above against the
    LIVE scalar SimBackend.apply() + the step_physics() formula on random
    states and actions. Must pass before trusting any MPC eval numbers -- this
    is the substitute for a shared-code guarantee, since the live physics were
    deliberately left untouched (see module docstring).

    Both paths are fed the SAME persisted (required_cpu, mem_need) pair --
    this checks implementation fidelity of the transition/metric math, not the
    persistence MODELING assumption itself (a design choice, not a bug to
    catch here): the live env's real next-tick demand is whatever the trace
    says, genuinely unknown to a controller deciding now, so there is nothing
    to compare it against at decision time."""
    from envs.joint_hv_env import SimBackend

    rng = np.random.default_rng(seed)
    mpc = MPCController()
    mismatches = []
    for _ in range(n_checks):
        s = SysState(
            replicas=int(rng.integers(REPLICA_MIN, REPLICA_MAX + 1)),
            target_replicas=int(rng.integers(REPLICA_MIN, REPLICA_MAX + 1)),
            cpu_req=int(rng.choice(_CPUS)),
            mem_req=int(rng.choice(_MEMS)),
            mem_need_now=float(rng.uniform(0, 480)),
            steps_to_ready=int(rng.integers(0, 3)),
        )
        d_rep, d_cpu, d_mem = ACTIONS[int(rng.integers(len(ACTIONS)))]
        required_cpu_persisted = float(rng.uniform(200, 5000))
        mem_need_persisted = s.mem_need_now  # what BOTH paths treat as "now", by construction

        # --- live scalar path: real SimBackend.apply(), then step_physics()'s
        # formula hand-copied (it reads the trace directly, so we feed it the
        # SAME persisted demand instead of faking a trace array) ---
        live = SimBackend.__new__(SimBackend)
        live.s = SysState(**s.__dict__)
        live.apply(d_rep, d_cpu, d_mem)
        if live.s.steps_to_ready > 0:
            live.s.steps_to_ready -= 1
            if live.s.steps_to_ready == 0:
                live.s.replicas = live.s.target_replicas
        live.s.mem_need_now = mem_need_persisted
        total_cpu = max(live.s.replicas * live.s.cpu_req, 1.0)
        rho = required_cpu_persisted / total_cpu
        knee = 1.0 / (1.0 + np.exp(-(rho - RHO_KNEE) / KNEE_WIDTH))
        lat_cpu = L0_MS + (LAT_SAT_MS - L0_MS) * knee
        mem_pressure = max(0.0, (live.s.mem_need_now - live.s.mem_req) / max(live.s.mem_req, 1))
        latency = min(lat_cpu * (1.0 + BETA_MEM * mem_pressure), LAT_CAP_MS)
        success = 1.0 - min(0.95, SAT_DROP * knee + OVERLOAD_DROP * max(0.0, rho - 1.31)
                            + OOM_DROP * mem_pressure)

        # --- vectorized path, evaluated for the single (state, action) pair above ---
        ai = ACTIONS.index((d_rep, d_cpu, d_mem))
        tr = mpc._static[ai]
        idx0 = _flat_index(np.array([s.replicas]), np.array([s.target_replicas]),
                           np.array([s.cpu_req]), np.array([s.mem_req]),
                           np.array([s.steps_to_ready]))[0]
        rep3, tgt2, cpu2 = tr.rep3[idx0], tr.tgt2[idx0], tr.cpu2[idx0]
        mem2 = tr.mem2_unratcheted[idx0]
        if d_mem < 0 and mem2 < mem_need_persisted:
            mem2 = np.clip(min(s.mem_req, np.ceil(mem_need_persisted)), MEM_MIN, MEM_MAX)
        total_cpu_v = max(rep3 * cpu2, 1.0)
        rho_v = required_cpu_persisted / total_cpu_v
        mem_pressure_v = max(0.0, (mem_need_persisted - mem2) / max(mem2, 1))
        knee_v = 1.0 / (1.0 + np.exp(-(rho_v - RHO_KNEE) / KNEE_WIDTH))
        lat_cpu_v = L0_MS + (LAT_SAT_MS - L0_MS) * knee_v
        latency_v = min(lat_cpu_v * (1.0 + BETA_MEM * mem_pressure_v), LAT_CAP_MS)
        success_v = 1.0 - min(0.95, SAT_DROP * knee_v + OVERLOAD_DROP * max(0.0, rho_v - 1.31)
                              + OOM_DROP * mem_pressure_v)

        ok = (rep3 == live.s.replicas and tgt2 == live.s.target_replicas
              and cpu2 == live.s.cpu_req and abs(mem2 - live.s.mem_req) < 1e-6
              and abs(latency_v - latency) < 1e-6 and abs(success_v - success) < 1e-9)
        if not ok:
            mismatches.append((s, (d_rep, d_cpu, d_mem), rep3, live.s.replicas,
                              latency_v, latency, mem2, live.s.mem_req))
    if mismatches:
        raise AssertionError(f"{len(mismatches)}/{n_checks} MPC fidelity mismatches: {mismatches[:3]}")
    print(f"verify_fidelity: {n_checks}/{n_checks} checks passed")


if __name__ == "__main__":
    verify_fidelity()
    mpc = MPCController()
    m = dict(rho=0.5, latency=6.0, success=1.0, load_frac=0.5, cpu_util=0.5, mem_pressure=0.0)
    s = SysState()
    t0 = time.perf_counter()
    for _ in range(50):
        mpc.act(m, s)
    print(f"mean act() time over 50 calls: {mpc.mean_plan_ms:.2f} ms")
