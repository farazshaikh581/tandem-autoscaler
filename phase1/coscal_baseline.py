"""CoScal: a multi-faceted RL autoscaler baseline (issue #21).

Reproduction of the controller in Xu et al., "CoScal: Multifaceted Scaling of
Microservices With Reinforcement Learning", IEEE Transactions on Network and
Service Management 19(4), Dec 2022 (DOI 10.1109/TNSM.2022.3210211). Reference
code: https://github.com/Kminassch/CoScal

CoScal is the third reproduced literature method, after the SLO-Driven
Cost-Aware autoscaler (#7) and Pandey PPO/DDPG (#10). It is directly comparable
to this project's method -- joint horizontal + vertical scaling, RL decisions,
an explicit response-time-vs-cost reward -- but differs on every implementation
axis, which is the point of running it: tabular Q-learning on a 5-level workload
state, vs deep distributional RL on the full observation vector.

The CoScal method (paper Sec. IV-V):
  state      workload level in {0..N_LEVELS-1}, binned from the demand signal
             against a 75% overload threshold (Table III)
  predictor  a GRU (MTFS preprocessing, Algorithm 1) forecasts the next
             interval's workload level
  trigger    event-driven: act only when the predicted level differs from the
             previous level (the source of CoScal's low churn)
  action     per node {h, v_i, b}: replica delta, per-resource delta, brownout
             on/off, executed cheapest-first (vertical, brownout, horizontal)
  learner    tabular Q-learning (eq. 1), epsilon-greedy, offline pretrain +
             online updates
  reward     r = R_qos(rt) / R_util(u)   (eqs. 2-4)
  adapter    auto-adapter: a vertical correction when the observed level at the
             start of an interval differs from the predicted one

Documented interpretive choices for this harness (full rationale in issue #21):
  1. Brownout omitted -- the single-service sim has no optional sub-components.
     CoScal's action set reduces to {h, v} = this env's joint H+V axis. Results
     are a lower bound on full CoScal.
  2. Workload-level bins: LEVEL_EDGES below, 0.75 as one boundary (paper's
     overload threshold); on `load_frac`, the demand signal, not utilisation
     (same choice as PredictiveController), so the level isn't contaminated by
     this controller's own capacity changes.
  3. GRU predictor ported separately (coscal_gru.py, trained on the azure
     traces); HoltLevelPredictor is the runnable fallback if the weights are
     absent.
  4. Scoring reward != training reward: coscal_reward() trains the Q-table only;
     evaluation uses the standard SLA/cost/churn/reversal metrics.
  5. Action budget pinned to this env's per-step magnitude (+-1 replica,
     +-100 mcpu, +-64 MiB) to isolate algorithm + state representation.
  6. State stays minimal -- only the workload level (+ optional utilisation
     bucket). The minimalism is the method.
  7. Global action a_g: N/A for a single-service homogeneous replica pool.
  8. Q-learning hyperparameters: repo values if available, else the defaults
     below.
  9. Azure organic traces only (matches #15-#20).

STATUS: complete (issue #21 steps 1-4). GRU trained (coscal_gru.py ->
models/coscal_gru.pt); Q-table trained offline by train_qtable() ->
models/coscal_qtable.pkl; wired into eval_baselines.py as the `coscal` row and
folded into comparison_azure.csv.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from envs.joint_hv_env import (CPU_MAX, CPU_MIN, MEM_MAX, MEM_MIN, REPLICA_MAX,
                               REPLICA_MIN, SLA_MEAN_LATENCY_MS, TRACE_FAMILIES,
                               family_suffix, set_trace_family)

def qtable_path() -> str:
    """models/coscal_qtable.pkl for azure, coscal_qtable_<family>.pkl otherwise."""
    return os.path.join(os.path.dirname(__file__), "models", f"coscal_qtable{family_suffix()}.pkl")
CLASSES = ("cpu-bound", "memory-bound", "io-bound", "mixed")   # train.CLASSES, kept local

# ---- workload-level discretisation (choice 2) ----
N_LEVELS = 5
# five bins on load_frac; 0.75 is the paper's overload threshold (Table III).
# TODO(#21 step 2): replace with the exact edges from the reference repo if it
# publishes them.
LEVEL_EDGES = (0.25, 0.50, 0.75, 1.00)   # -> levels 0..4

# ---- CoScal reward constants (eqs. 2-4) ----
RT_MAX_MS = SLA_MEAN_LATENCY_MS          # maximum acceptable response time
U_MAX = 0.75                             # overload / max-utilisation threshold

# ---- Q-learning hyperparameters (choice 8; repo values TBD, issue #21 step 3) ----
ALPHA = 0.1                              # learning rate
GAMMA = 0.9                              # discount factor
EPSILON = 0.1                            # epsilon-greedy exploration

# ---- CoScal action sub-space (choice 1: brownout dropped; choice 5: unit steps) ----
# {h in {-1,0,1}} x {v_cpu in {-100,0,100}} x {v_mem in {-64,0,64}}
_H = (-1, 0, 1)
_V_CPU = (-100, 0, 100)
_V_MEM = (-64, 0, 64)
COSCAL_ACTIONS = [(h, c, m) for h in _H for c in _V_CPU for m in _V_MEM]  # 27
NOOP = (0, 0, 0)


def workload_level(load_frac: float) -> int:
    """Bin the demand signal into a CoScal workload level in {0..N_LEVELS-1}."""
    lvl = 0
    for e in LEVEL_EDGES:
        if load_frac >= e:
            lvl += 1
    return lvl


def coscal_reward(latency_ms: float, cpu_util: float) -> float:
    """CoScal reward r = R_qos(rt) / R_util(u)  (paper eqs. 2-4).

    Single-machine reduction of eq. 3 (K = 1, a_g collapsed -- choice 7):
      R_util(u) = (U_MAX - u) + 1        if u <= U_MAX
                = (u - U_MAX) + 1        if u >  U_MAX
    Higher R_qos and lower R_util raise the total (eq. 4).
    """
    if latency_ms > RT_MAX_MS:
        r_qos = float(np.exp(-(((latency_ms - RT_MAX_MS) / RT_MAX_MS) ** 2)))
    else:
        r_qos = 1.0
    if cpu_util <= U_MAX:
        r_util = (U_MAX - cpu_util) + 1.0
    else:
        r_util = (cpu_util - U_MAX) + 1.0
    return r_qos / r_util


class WorkloadLevelPredictor:
    """Forecasts the next interval's workload level from the demand history."""

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, load_frac: float) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class HoltLevelPredictor(WorkloadLevelPredictor):
    """Runnable fallback (choice 3): Holt's linear-trend extrapolation of
    `load_frac`, one interval ahead, then binned to a level. Mirrors the
    forecaster in PredictiveController (#17) so the fallback is a known
    quantity; swapped for coscal_gru.GRULevelPredictor once that is trained.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1) -> None:
        self.alpha, self.beta = alpha, beta
        self.reset()

    def reset(self) -> None:
        self._level_v: float | None = None
        self._trend = 0.0

    def predict(self, load_frac: float) -> int:
        if self._level_v is None:
            self._level_v = load_frac
            return workload_level(load_frac)
        prev = self._level_v
        self._level_v = self.alpha * load_frac + (1 - self.alpha) * (prev + self._trend)
        self._trend = self.beta * (self._level_v - prev) + (1 - self.beta) * self._trend
        return workload_level(self._level_v + self._trend)


class CoScalController:
    """CoScal controller for eval_baselines.evaluate_policy().

    act(m, s) -> (d_rep, d_cpu, d_mem), one of COSCAL_ACTIONS, clamped at the
    actuator bounds so a knob already at its limit is not 'moved'.

    Parameters
    ----------
    predictor : WorkloadLevelPredictor or None
        None -> HoltLevelPredictor() (fallback).
    q_table : dict or None
        {(level_from, level_to): np.ndarray[len(COSCAL_ACTIONS)]}. None -> empty
        (every state's Q starts at zeros; with train=False the controller then
        plays the argmax of zeros = COSCAL_ACTIONS[0], so a trained table is
        required for a meaningful run -- issue #21 step 3).
    epsilon : float
        Exploration rate. Kept at 0.0 for deterministic evaluation; ALPHA-style
        exploration only matters during train_qtable().
    train : bool
        If True, act() records (state, action) for a later update() call
        (online updates, paper Algorithm 2 lines 9-14).
    rng : np.random.Generator or None
        Only used when epsilon > 0.
    """

    def __init__(self, predictor: WorkloadLevelPredictor | None = None,
                 q_table: dict | None = None, epsilon: float = 0.0,
                 train: bool = False, rng: np.random.Generator | None = None) -> None:
        self.predictor = predictor if predictor is not None else HoltLevelPredictor()
        self.q = q_table if q_table is not None else {}
        self.epsilon = epsilon
        self.train = train
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.reset()

    def reset(self) -> None:
        self.predictor.reset()
        self._prev_level: int | None = None
        self._last_key: tuple[int, int] | None = None
        self._last_action_idx: int | None = None

    # -- Q-table access -------------------------------------------------------
    def _q_row(self, key: tuple[int, int]) -> np.ndarray:
        row = self.q.get(key)
        if row is None:
            row = np.zeros(len(COSCAL_ACTIONS), dtype=np.float64)
            self.q[key] = row
        return row

    def _select(self, key: tuple[int, int]) -> int:
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(COSCAL_ACTIONS)))
        return int(np.argmax(self._q_row(key)))

    def update(self, key: tuple[int, int], action_idx: int, reward: float,
               next_key: tuple[int, int]) -> None:
        """Tabular Q-update, paper eq. 1 / Algorithm 2 line 12. Called by
        train_qtable() (offline) and, when train=True, once per acted interval
        during evaluation (online)."""
        q_sa = self._q_row(key)[action_idx]
        target = reward + GAMMA * float(np.max(self._q_row(next_key)))
        self._q_row(key)[action_idx] = q_sa + ALPHA * (target - q_sa)

    # -- auto-adapter (paper Sec. V-D) -------------------------------------
    @staticmethod
    def _auto_adapter(observed_level: int, predicted_level: int,
                      s) -> tuple[int, int]:
        """When the observed level disagrees with the prediction, apply a
        vertical CPU correction toward the gap (CoScal corrects with vertical
        scaling only). Returns (d_cpu, d_mem)."""
        if observed_level == predicted_level:
            return 0, 0
        d_cpu = 100 if observed_level > predicted_level else -100
        return d_cpu, 0

    # -- main entry point --------------------------------------------------
    def act(self, m, s) -> tuple[int, int, int]:
        load_frac = m["load_frac"]
        observed = workload_level(load_frac)
        predicted = self.predictor.predict(load_frac)

        # first tick: nothing to compare against yet
        if self._prev_level is None:
            self._prev_level = predicted
            return NOOP

        # event trigger (Algorithm 2 line 6): no level change -> no RL action,
        # only the lightweight auto-adapter correction.
        if predicted == self._prev_level:
            d_cpu, d_mem = self._auto_adapter(observed, predicted, s)
            d_rep = 0
            self._last_key = None
            self._last_action_idx = None
        else:
            key = (self._prev_level, predicted)
            a_idx = self._select(key)
            d_rep, d_cpu, d_mem = COSCAL_ACTIONS[a_idx]
            self._last_key = key
            self._last_action_idx = a_idx
            self._prev_level = predicted

        # clamp at actuator bounds (same discipline as the other controllers)
        if (d_rep > 0 and s.replicas >= REPLICA_MAX) or (d_rep < 0 and s.replicas <= REPLICA_MIN):
            d_rep = 0
        if (d_cpu > 0 and s.cpu_req >= CPU_MAX) or (d_cpu < 0 and s.cpu_req <= CPU_MIN):
            d_cpu = 0
        if (d_mem > 0 and s.mem_req >= MEM_MAX) or (d_mem < 0 and s.mem_req <= MEM_MIN):
            d_mem = 0
        return (d_rep, d_cpu, d_mem)


def save_qtable(q: dict, path: str | None = None) -> None:
    path = path or qtable_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({k: np.asarray(v, dtype=np.float64) for k, v in q.items()}, f)


def load_qtable(path: str | None = None) -> dict:
    with open(path or qtable_path(), "rb") as f:
        return pickle.load(f)


def train_qtable(predictor, episodes: int = 6000, seed: int = 0,
                 episode_len: int = 200, eps_start: float = EPSILON,
                 eps_end: float = 0.01) -> dict:
    """Offline Q-table training on the azure traces (paper Algorithm 2, offline
    branch + the online-update rule from lines 9-14).

    Rolls one CoScalController(train=True) with linearly decaying epsilon
    through JointHVEnv, cycling the four classes (the `--workload all`
    generalist posture). CoScal's event trigger means only some intervals carry
    an RL action; this accumulates coscal_reward(latency, cpu_util) across the
    auto-adapter-only intervals between two RL actions and applies the tabular
    Q-update (eq. 1) over that semi-MDP transition, bootstrapping off the key of
    the next RL action. The learner is tiny -- state is (prev_level, pred_level)
    in {0..4}^2, so <=25 keys x 27 actions -- and converges well inside the
    default 6000 episodes (the paper's ~16.5k epochs is for a far larger
    multi-node state).

    `predictor` is a WorkloadLevelPredictor instance (GRULevelPredictor or
    HoltLevelPredictor); its history is reset with every episode. Returns the
    learned `.q` dict; caller pickles it via save_qtable().
    """
    from envs.joint_hv_env import ACTIONS, JointHVEnv

    rng = np.random.default_rng(seed)
    q: dict = {}
    ctrl = CoScalController(predictor=predictor, q_table=q, epsilon=eps_start,
                            train=True, rng=rng)
    envs = {c: JointHVEnv(workload=c, episode_len=episode_len, seed=seed + 777)
            for c in CLASSES}
    decay_over = max(1.0, 0.8 * episodes)

    for ep in range(episodes):
        env = envs[CLASSES[ep % len(CLASSES)]]
        ctrl.epsilon = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / decay_over)
        _, info = env.reset(seed=seed + 777 + ep)
        ctrl.reset()                              # also resets predictor history
        m = info["metrics"]
        pend_key = pend_idx = None
        pend_rew = 0.0
        done = False
        while not done:
            d_action = ctrl.act(m, env._backend.s)
            took_rl = ctrl._last_key is not None
            _, _, term, trunc, info = env.step(ACTIONS.index(d_action))
            m = info["metrics"]
            step_rew = coscal_reward(m["latency"], m["cpu_util"])
            if took_rl:
                new_key, new_idx = ctrl._last_key, ctrl._last_action_idx
                if pend_key is not None:
                    ctrl.update(pend_key, pend_idx, pend_rew, new_key)
                pend_key, pend_idx, pend_rew = new_key, new_idx, step_rew
            else:
                pend_rew += step_rew
            done = term or trunc
        if pend_key is not None:                  # episode end -> bootstrap off self
            ctrl.update(pend_key, pend_idx, pend_rew, pend_key)

    return q


def verify_reward() -> None:
    """Cheap sanity check on coscal_reward() against the paper's eqs. 2-4.
    Safe to run (no env, no training)."""
    assert coscal_reward(RT_MAX_MS - 1.0, 0.5) == 1.0 / (U_MAX - 0.5 + 1.0)
    assert coscal_reward(RT_MAX_MS, U_MAX) == 1.0 / 1.0
    over = coscal_reward(2 * RT_MAX_MS, 0.9)
    assert 0.0 < over < 1.0
    assert workload_level(0.0) == 0 and workload_level(2.0) == N_LEVELS - 1
    assert workload_level(0.75) == 3   # overload threshold lands in level 3
    print("coscal_baseline: reward + level checks OK")


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="CoScal baseline: sanity check / Q-table training")
    ap.add_argument("--train-qtable", action="store_true",
                    help="run offline Q-table training and pickle it to models/coscal_qtable.pkl")
    ap.add_argument("--predictor", choices=["gru", "holt"], default="gru",
                    help="workload-level predictor for training (default: the trained GRU)")
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure")
    args = ap.parse_args()
    set_trace_family(args.trace_family)

    verify_reward()
    if args.train_qtable:
        if args.predictor == "gru":
            from coscal_gru import GRULevelPredictor
            pred = GRULevelPredictor()
        else:
            pred = HoltLevelPredictor()
        t0 = time.perf_counter()
        q = train_qtable(pred, episodes=args.episodes, seed=args.seed)
        save_qtable(q)
        print(f"wrote {qtable_path()}  ({len(q)} states, "
              f"{sum(int(np.any(r != 0.0)) for r in q.values())} non-zero rows, "
              f"{time.perf_counter() - t0:.1f}s)")
