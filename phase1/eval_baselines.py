"""Classical-autoscaler baselines for the headline comparison table (#1, #SOTA-1).

Evaluates hand-coded HPA / HPA+VPA / SLO-Driven-Cost-Aware controllers on the SAME
sim, classes, episodes and metrics as the trained RL policies (train.evaluate), so
their rows drop straight into eval/comparison.csv via aggregate_eval.py (which globs
eval_*_all.csv).

  hpa-only : Kubernetes HPA replica control on CPU utilisation (target 70%,
             10% tolerance, scale-down stabilisation). Never touches requests ->
             cannot relieve memory pressure (the death-spiral-prone control).
  hpa-vpa  : the same HPA on CPU for replicas, PLUS a VPA that right-sizes the
             MEMORY request to the working set with headroom. This is the
             recommended HPA+VPA split (don't put both on CPU); it is the
             strongest classical joint controller, so the RL policy beating it
             is the meaningful result.
  slo-cost-aware      : reproduction of "An SLO Driven and Cost-Aware Autoscaling
             Framework for Kubernetes" (arXiv 2512.23415, Dec 2025) -- see
             SloCostAwareController docstring for the faithful-reconstruction
             notes (the paper publishes stage descriptions, not formulas). Purely
             horizontal, like the paper's own proposed controller; the paper
             itself explicitly defers in-place vertical integration to future
             work, which is the gap this project closes.
  slo-cost-aware-vpa  : the same SLO-driven horizontal decision engine paired with
             the same VPA memory right-sizing as hpa-vpa -- our fairest classical
             hybrid, isolating whether SLO-violation-driven horizontal decisions
             (vs plain CPU-ratio HPA) change the memory-bound outcome once paired
             with a vertical assist.
  mpc                 : model-predictive control challenger (issue #16) -- see
             mpc_baseline.MPCController.
  predictive          : forecast-then-scale (issue #17) -- Holt's linear-trend
             extrapolation of the demand signal, scaling ahead of a predicted
             ramp instead of reacting to the current tick. Same threshold/
             tolerance/stabilisation discipline as ClassicalController, so the
             comparison isolates forecast-vs-react, nothing else. See
             PredictiveController docstring.
  predictive-vpa      : the same forecast-driven horizontal engine paired with
             the same VPA memory right-sizing as hpa-vpa/slo-cost-aware-vpa.
  coscal              : reproduction of CoScal (Xu et al., IEEE TNSM 19(4), 2022)
             -- GRU-forecast workload-level state, tabular Q-learning, CoScal's
             r=R_qos/R_util training reward, event trigger, vertical-first
             execution, auto-adapter. Joint H+V and RL like this project's
             method but on a minimal 5-level state. See coscal_baseline.py /
             coscal_gru.py and issue #21. Needs models/coscal_qtable.pkl +
             models/coscal_gru.pt (falls back to a Holt predictor if the GRU
             weights are missing; skipped entirely if the Q-table is missing).

  python eval_baselines.py            # writes eval_<name>_all.csv for every controller
  python aggregate_eval.py            # folds them into comparison.csv
"""
import argparse
import csv
import math
import os
import time

import numpy as np

from envs.joint_hv_env import (ACTIONS, CPU_MAX, CPU_MIN, JointHVEnv, MEM_MAX,
                               MEM_MIN, REPLICA_MAX, REPLICA_MIN,
                               SCALE_DELAY_STEPS, SLA_MEAN_LATENCY_MS,
                               TRACE_FAMILIES, set_trace_family)
from coscal_baseline import CoScalController, load_qtable
from mpc_baseline import MPCController
from train import CLASSES, _reversals

HERE = os.path.dirname(__file__)

HPA_TARGET_UTIL = 0.70   # standard HPA CPU target
HPA_TOLERANCE = 0.10     # HPA default: no action within 10% of target
HPA_SCALEDOWN_STAB = 3   # steps util must stay low before scaling in (anti-flap)
VPA_MEM_HEADROOM = 1.15  # keep the memory request ~15% above the working set


class ClassicalController:
    """HPA replica control (+ optional VPA memory right-sizing). Emits an ACTIONS
    (d_rep, d_cpu, d_mem) tuple, clamped so a knob already at its bound is not
    'moved' (which would inflate churn/reversals unfairly)."""

    def __init__(self, vpa: bool, target_util: float = HPA_TARGET_UTIL):
        self.vpa = vpa
        self.target_util = target_util   # swept for the classical cost/SLA frontier (#20)
        self.reset()

    def reset(self) -> None:
        self._down = 0

    def act(self, m, s) -> tuple[int, int, int]:
        d_rep = 0
        # --- HPA: desired replicas from CPU utilisation ratio ---
        ratio = m["cpu_util"] / self.target_util
        if s.steps_to_ready > 0:
            self._down = 0                       # a scale is in flight; HPA waits
        elif ratio > 1.0 + HPA_TOLERANCE:
            d_rep, self._down = +1, 0            # scale out immediately
        elif ratio < 1.0 - HPA_TOLERANCE:
            self._down += 1                      # scale in only after stabilisation
            if self._down >= HPA_SCALEDOWN_STAB:
                d_rep, self._down = -1, 0
        else:
            self._down = 0

        # --- VPA: right-size the MEMORY request to the working set (recommended split) ---
        d_mem = 0
        if self.vpa:
            target_mem = s.mem_need_now * VPA_MEM_HEADROOM
            if target_mem > s.mem_req:
                d_mem = +64                      # under-provisioned -> grow
            elif target_mem < s.mem_req - 64:
                d_mem = -64                      # comfortably over -> shrink (ratchet-guarded in env)

        # clamp deltas at the actuator bounds so no phantom action is issued
        if (d_rep > 0 and s.replicas >= REPLICA_MAX) or (d_rep < 0 and s.replicas <= REPLICA_MIN):
            d_rep = 0
        if (d_mem > 0 and s.mem_req >= MEM_MAX) or (d_mem < 0 and s.mem_req <= MEM_MIN):
            d_mem = 0
        return (d_rep, 0, d_mem)


SLO_SCALE_DOWN_UTIL = 0.5   # scale in only when comfortably under the SLO, not just "not violating"
SLO_SCALEDOWN_STAB = 3      # same stabilisation window as the HPA baseline (fair comparison)


class SloCostAwareController:
    """Reproduction of the 5-stage pipeline in "An SLO Driven and Cost-Aware
    Autoscaling Framework for Kubernetes" (arXiv 2512.23415, Dec 2025).

    The paper names five stages (Signal Aggregation -> SLO-Driven Demand
    Estimation -> Workload Sensitivity Adjustment -> Guardrail Enforcement ->
    Coordinated Actuation) and a state tuple s=(L,Q,U,P), but publishes no
    formulas -- it is explicitly "qualitative and rule-based." This is our
    documented, faithful-to-the-stage-descriptions reconstruction, run through
    the SAME per-step ternary action interface every other controller/RL policy
    in this comparison uses (so "bounding the queue-draining rate" collapses to
    a gate rather than a magnitude, since no controller here can move more than
    one replica per step):

      L (latency)      -> m["latency"]
      Q (queue/backlog) -> max(0, rho-1): our sim has no explicit queue counter,
                           so we use demand-exceeds-capacity as the closest
                           analogue of queue buildup.
      U (utilisation)   -> m["cpu_util"]
      P (pending pods)  -> s.target_replicas - s.replicas (already in the env's
                           own observation as "pending")

    Stage 2 (SLO-Driven Demand Estimation): "initial replica requirement
    proportional to SLO violation severity" -> we reuse the env's own SLO
    violation measure, viol = max(0, (L-SLA)/SLA) + (1-success).

    Stage 3 (Workload Sensitivity Adjustment, "queue-aware... bound queue
    draining"): a violation only triggers a scale-up if Q confirms it is a
    genuine capacity shortfall (rho>1) rather than e.g. a memory-pressure-driven
    latency spike replicas cannot fix -- the cost-aware guardrail against
    wasteful horizontal scale-out.

    Stage 4 (Guardrail Enforcement: hard bounds, cost-aware selection,
    stability controls): hard replica bounds (global), a pending-scale wait
    (mirrors HPA baseline discipline), and a stabilisation window before
    scaling down so a controller isn't credited/debited churn from a
    differently-parameterised anti-flap window than its HPA competitor.

    Stage 5 (Coordinated Actuation): this controller is purely horizontal, like
    the paper's own proposed framework -- vertical scaling is explicitly out of
    its scope (the paper's future-work section defers "tighter integration of
    vertical resource adjustments into real-time control loops"). Node-capacity
    hints are out of scope too (our env is single-node). `vpa=True` pairs it
    with the same memory VPA as `hpa-vpa` for a fair hybrid comparison.
    """

    def __init__(self, vpa: bool = False):
        self.vpa = vpa
        self.reset()

    def reset(self) -> None:
        self._down = 0

    def act(self, m, s) -> tuple[int, int, int]:
        d_rep = 0
        if s.steps_to_ready > 0:
            self._down = 0                       # Stage 4: a scale is in flight, wait
        else:
            viol = max(0.0, (m["latency"] - SLA_MEAN_LATENCY_MS) / SLA_MEAN_LATENCY_MS) + (1.0 - m["success"])
            backlog = max(0.0, m["rho"] - 1.0)   # Q
            if viol > 0 and backlog > 0:
                # Stage 2+3: SLO violation confirmed by a genuine capacity shortfall
                d_rep, self._down = +1, 0
            elif viol == 0 and m["cpu_util"] < SLO_SCALE_DOWN_UTIL:
                # Stage 4: stability control before committing a scale-down
                self._down += 1
                if self._down >= SLO_SCALEDOWN_STAB:
                    d_rep, self._down = -1, 0
            else:
                self._down = 0

        d_mem = 0
        if self.vpa:
            target_mem = s.mem_need_now * VPA_MEM_HEADROOM
            if target_mem > s.mem_req:
                d_mem = +64
            elif target_mem < s.mem_req - 64:
                d_mem = -64

        if (d_rep > 0 and s.replicas >= REPLICA_MAX) or (d_rep < 0 and s.replicas <= REPLICA_MIN):
            d_rep = 0
        if (d_mem > 0 and s.mem_req >= MEM_MAX) or (d_mem < 0 and s.mem_req <= MEM_MIN):
            d_mem = 0
        return (d_rep, 0, d_mem)


HOLT_ALPHA = 0.3     # level smoothing
HOLT_BETA = 0.1       # trend smoothing (slower -- avoid chasing noise in the slope)
FORECAST_LEAD = SCALE_DELAY_STEPS  # scale ahead exactly far enough that a fresh
                                    # replica lands when the predicted spike arrives


class PredictiveController:
    """Forecast-then-scale (issue #17): every other baseline, including MPC,
    reacts to (or, for MPC, assumes persistence of) the CURRENT observed state.
    This one extrapolates a TREND and scales ahead of a predicted ramp instead
    of waiting for it to arrive.

    Forecast: Holt's linear-trend method (level + trend double exponential
    smoothing) on `m["load_frac"]` -- the raw demand signal, not utilisation,
    so the forecast isn't contaminated by this controller's own capacity
    changes. `FORECAST_LEAD = SCALE_DELAY_STEPS` ticks ahead: exactly long
    enough that a replica requested now is ready by the time the predicted
    spike materialises.

    The forecasted load_frac is converted to a forecasted utilisation ratio
    using the CURRENT tick's observed (rho / load_frac) ratio, recomputed
    fresh every tick -- this self-corrects for any capacity change this same
    controller just made (no stale coupling to an old capacity), and avoids
    needing per-class intensity constants (INTENSITY is private to SimBackend;
    every controller in this file only ever reads `m`/`s`, by design).

    Trigger/stabilisation discipline is otherwise IDENTICAL to
    ClassicalController (same HPA_TARGET_UTIL/HPA_TOLERANCE/
    HPA_SCALEDOWN_STAB), so the comparison isolates forecast-vs-react and
    nothing else -- the predictive analogue of how hpa-vpa isolates the VPA
    assist. `vpa=True` pairs it with the same reactive VPA memory
    right-sizing every other -vpa variant uses (the memory axis is not made
    predictive here, matching how VPA is a bolted-on fair pairing everywhere
    else in this file, not itself the object of study)."""

    def __init__(self, vpa: bool, target_util: float = HPA_TARGET_UTIL):
        self.vpa = vpa
        self.target_util = target_util   # swept for the classical cost/SLA frontier (#20)
        self.reset()

    def reset(self) -> None:
        self._down = 0
        self._level = None
        self._trend = 0.0

    def _forecast_load_frac(self, load_frac: float) -> float:
        if self._level is None:
            self._level = load_frac  # no history yet -> forecast collapses to persistence
            return load_frac
        prev_level = self._level
        self._level = HOLT_ALPHA * load_frac + (1 - HOLT_ALPHA) * (prev_level + self._trend)
        self._trend = HOLT_BETA * (self._level - prev_level) + (1 - HOLT_BETA) * self._trend
        return self._level + FORECAST_LEAD * self._trend

    def act(self, m, s) -> tuple[int, int, int]:
        load_frac_hat = self._forecast_load_frac(m["load_frac"])
        k_now = m["rho"] / max(m["load_frac"], 1e-6)  # implied demand->rho factor, this tick
        forecast_util = min(max(k_now * load_frac_hat, 0.0), 1.5)

        d_rep = 0
        ratio = forecast_util / self.target_util
        if s.steps_to_ready > 0:
            self._down = 0
        elif ratio > 1.0 + HPA_TOLERANCE:
            d_rep, self._down = +1, 0
        elif ratio < 1.0 - HPA_TOLERANCE:
            self._down += 1
            if self._down >= HPA_SCALEDOWN_STAB:
                d_rep, self._down = -1, 0
        else:
            self._down = 0

        d_mem = 0
        if self.vpa:
            target_mem = s.mem_need_now * VPA_MEM_HEADROOM
            if target_mem > s.mem_req:
                d_mem = +64
            elif target_mem < s.mem_req - 64:
                d_mem = -64

        if (d_rep > 0 and s.replicas >= REPLICA_MAX) or (d_rep < 0 and s.replicas <= REPLICA_MIN):
            d_rep = 0
        if (d_mem > 0 and s.mem_req >= MEM_MAX) or (d_mem < 0 and s.mem_req <= MEM_MIN):
            d_mem = 0
        return (d_rep, 0, d_mem)


def _make_coscal():
    """CoScalController wired with the trained Q-table + GRU predictor, or None
    if the Q-table has not been trained yet (see coscal_baseline.py --train-qtable)."""
    try:
        q = load_qtable()
    except FileNotFoundError:
        print("  [coscal] no models/coscal_qtable.pkl -- run "
              "`python coscal_baseline.py --train-qtable` first; skipping the coscal row")
        return None
    from coscal_gru import make_predictor   # torch import deferred to here
    return CoScalController(predictor=make_predictor(), q_table=q,
                            epsilon=0.0, train=False)


def evaluate_policy(controller, seed: int, episodes: int = 20) -> list[dict]:
    """Mirror of train.evaluate() so metrics are identical to the RL rows."""
    rows = []
    for wl in CLASSES:
        env = JointHVEnv(workload=wl, episode_len=200, seed=seed + 777)
        sla, lat, cost, churn = [], [], [], 0
        reversals = dict(replica=0, cpu=0, mem=0)
        plan_ms = []
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + 777 + ep)
            controller.reset()
            m = info["metrics"]
            reps, cpus, mems, prev_a = [], [], [], None
            done = False
            while not done:
                t0 = time.perf_counter()
                d_action = controller.act(m, env._backend.s)
                plan_ms.append((time.perf_counter() - t0) * 1000.0)
                a = ACTIONS.index(d_action)
                obs, r, term, trunc, info = env.step(a)
                m = info["metrics"]; s = env._backend.s
                lat.append(m["latency"])
                sla.append(m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99)
                cost.append(s.replicas * s.cpu_req / 1000.0)
                if prev_a is not None and a != prev_a:
                    churn += 1
                prev_a = a
                reps.append(s.replicas); cpus.append(s.cpu_req); mems.append(s.mem_req)
                done = term or trunc
            reversals["replica"] += _reversals(reps)
            reversals["cpu"] += _reversals(cpus)
            reversals["mem"] += _reversals(mems)
        rows.append(dict(workload=wl, sla_pct=round(100 * np.mean(sla), 1),
                         mean_lat_ms=round(float(np.mean(lat)), 1),
                         mean_cost_vcpu=round(float(np.mean(cost)), 2),
                         churn=churn, replica_reversals=reversals["replica"],
                         cpu_reversals=reversals["cpu"], mem_reversals=reversals["mem"],
                         mean_plan_ms=round(float(np.mean(plan_ms)), 3)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="match the RL rows (seed 0)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tag", default="", help="output filename suffix, e.g. 'azure' -> eval_<name>_all_azure.csv")
    ap.add_argument("--w-cost", type=float, default=None,
                    help="MPC objective cost weight for the Pareto sweep (#20); default keeps the env constant")
    ap.add_argument("--w-viol", type=float, default=None,
                    help="MPC objective SLA-violation weight for the Pareto sweep (#20); default keeps the env constant")
    ap.add_argument("--hpa-target", type=float, default=None,
                    help="CPU utilisation target for the HPA/predictive controllers (#20 classical frontier); "
                         "default 0.70. Does not affect slo-cost-aware (viol-gated) or mpc.")
    ap.add_argument("--only", default="",
                    help="comma-separated controller names to run (default: all). e.g. 'mpc' or "
                         "'hpa-only,hpa-vpa,predictive,predictive-vpa'")
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure",
                    help="trace family to evaluate on; bitbrains runs need a 'bb' tag token")
    args = ap.parse_args()
    set_trace_family(args.trace_family, args.tag)
    suffix = f"_{args.tag}" if args.tag else ""
    tu = args.hpa_target if args.hpa_target is not None else HPA_TARGET_UTIL

    fields = ["workload", "sla_pct", "mean_lat_ms", "mean_cost_vcpu", "churn",
              "replica_reversals", "cpu_reversals", "mem_reversals", "mean_plan_ms"]
    controllers = [
        ("hpa-only", ClassicalController(vpa=False, target_util=tu)),
        ("hpa-vpa", ClassicalController(vpa=True, target_util=tu)),
        ("slo-cost-aware", SloCostAwareController(vpa=False)),
        ("slo-cost-aware-vpa", SloCostAwareController(vpa=True)),
        ("mpc", MPCController(w_cost=args.w_cost, w_viol=args.w_viol)),
        ("predictive", PredictiveController(vpa=False, target_util=tu)),
        ("predictive-vpa", PredictiveController(vpa=True, target_util=tu)),
    ]
    want = {n.strip() for n in args.only.split(",") if n.strip()} if args.only else None
    if want is None or "coscal" in want:
        coscal = _make_coscal()
        if coscal is not None:
            controllers.append(("coscal", coscal))
    for name, controller in controllers:
        if want is not None and name not in want:
            continue
        rows = evaluate_policy(controller, args.seed, args.episodes)
        out = os.path.join(HERE, "eval", f"eval_{name}_all{suffix}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(f"\n== {name} ==")
        for r in rows:
            print(f"  {r['workload']:13s} SLA {r['sla_pct']:5.1f}%  lat {r['mean_lat_ms']:6.1f}ms  "
                  f"cost {r['mean_cost_vcpu']:5.2f}  churn {r['churn']:4d}  "
                  f"rev(r/c/m) {r['replica_reversals']}/{r['cpu_reversals']}/{r['mem_reversals']}  "
                  f"plan {r['mean_plan_ms']:.2f}ms")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
