"""Unified train + eval scaffold for joint H+V autoscaling on JointHVEnv.

Algorithms (SB3 / sb3-contrib):
  qrdqn  - QR-DQN  (HEADLINE; distributional). NOTE: standard QR-DQN selects actions
           by the MEAN of quantiles. The risk-averse (CVaR) action selection that is
           our novelty is a TODO -- see cvar note below.
  dqn    - DQN (Double-DQN-style value baseline / ablation)
  ppo    - PPO (on-policy reference to prior work)
  trpo   - TRPO (trust-region comparator)
  sac-discrete - custom discrete SAC (Christodoulou 2019); see sac_discrete.py

Usage:
  python train.py --algo qrdqn --workload all --timesteps 300000
  python train.py --algo ppo   --workload memory-bound --timesteps 200000
Eval runs all four workload classes and reports SLA%, latency, churn, oscillation.
"""
import argparse
import csv
import os

import numpy as np
import torch as th
th.set_num_threads(1)  # tiny nets; avoids CPU contention when 5 algos train concurrently
from stable_baselines3 import DDPG, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import QRDQN, TRPO

from envs.joint_hv_env import (ACTIONS, TRACE_FAMILIES, JointHVEnv, SLA_MEAN_LATENCY_MS,
                               set_trace_family, trace_dir)

HERE = os.path.dirname(__file__)
CLASSES = ["cpu-bound", "memory-bound", "io-bound", "mixed"]
ALGOS = {"qrdqn": QRDQN, "dqn": DQN, "ppo": PPO, "trpo": TRPO}
OFF_POLICY = {"qrdqn", "dqn"}
# Pandey SOTA reproduction (thread #SOTA-1): continuous action space + the
# paper's own reward (Sec 3.2), since DDPG has no discrete-action formulation.
# PPO is retrained on the SAME continuous env/reward for a fair PPO-vs-DDPG
# comparison under identical conditions, matching what the paper itself compares.
PANDEY_ALGOS = {"pandey-ppo": PPO, "pandey-ddpg": DDPG}


def hyperparams(algo: str, seed: int, tb: bool = True) -> dict:
    common = dict(policy="MlpPolicy", gamma=0.99, seed=seed, verbose=0,
                  tensorboard_log=os.path.join(HERE, "tb") if tb else None)
    if algo == "pandey-ddpg":
        action_noise = NormalActionNoise(mean=np.zeros(3), sigma=0.1 * np.ones(3))
        return dict(common, learning_rate=1e-3, buffer_size=100_000, learning_starts=2_000,
                    batch_size=128, tau=0.005, action_noise=action_noise,
                    policy_kwargs=dict(net_arch=[256, 256]))
    if algo in OFF_POLICY:
        return dict(common, learning_rate=3e-4, buffer_size=100_000, learning_starts=2_000,
                    batch_size=128, train_freq=4, target_update_interval=1_000,
                    exploration_fraction=0.2, exploration_final_eps=0.05,
                    policy_kwargs=dict(net_arch=[256, 256]))
    return dict(common, learning_rate=3e-4, n_steps=2_048, batch_size=256,
                gae_lambda=0.95, policy_kwargs=dict(net_arch=[256, 256]))


def make_env(workload: str, seed: int, episode_len: int = 200, trace_pool: dict | None = None,
             action_mode: str = "discrete", reward_fn: str = "default",
             scaling_mode: str = "joint", w_cost: float | None = None,
             w_churn: float | None = None, w_viol: float | None = None):
    return Monitor(JointHVEnv(workload=workload, episode_len=episode_len, seed=seed,
                              trace_pool=trace_pool, action_mode=action_mode, reward_fn=reward_fn,
                              scaling_mode=scaling_mode, w_cost=w_cost, w_churn=w_churn, w_viol=w_viol))


def build_train_pool() -> dict:
    """Map each class -> its list of traces_train/*.npz paths (domain randomization)."""
    import glob
    pool_dir = trace_dir("traces_train")
    pool: dict[str, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(pool_dir, "*.npz"))):
        cls = os.path.basename(p).split("__")[0]
        pool.setdefault(cls, []).append(p)
    if not pool:
        raise SystemExit("no traces_train/*.npz -- run select_train_pool.py first")
    return pool


def load_model(algo: str, mpath: str, env, cvar_alpha: float = 0.25, seed: int = 0):
    """Load a saved model, reconstructing the custom cvar/sac types correctly.
    Shared by --eval-only and the real-cluster eval (eval_real.py)."""
    if algo in ("cvar-qrdqn", "cvar-bellman-qrdqn"):
        # Both act CVaR-greedy via the same policy; the Bellman variant differs only
        # in the training target, which is irrelevant at eval -> load as QRDQN.
        import cvar_qrdqn  # registers the custom policy classes for unpickling
        cvar_qrdqn.CVaRQuantileNetwork.cvar_alpha = cvar_alpha
        return QRDQN.load(mpath, env=env)
    if algo == "sac-discrete":
        from sac_discrete import SACDiscrete
        model = SACDiscrete(env, seed=seed)
        ckpt = th.load(mpath + ".zip", map_location="cpu", weights_only=False)
        model.actor.load_state_dict(ckpt["actor"])
        return model
    if algo in PANDEY_ALGOS:
        return PANDEY_ALGOS[algo].load(mpath, env=env)
    return ALGOS[algo].load(mpath, env=env)


def _reversals(series: list[int]) -> int:
    """Direction changes of an actuator over an episode. Compress consecutive
    equal values to isolate moves, then count sign flips between successive
    moves. A flat or monotonic series has 0 reversals (this is why an untrained
    policy pinned at min replicas correctly reads 0, not a broken metric)."""
    d = np.diff([x for i, x in enumerate(series) if i == 0 or series[i] != series[i - 1]])
    return int(np.sum(np.diff(np.sign(d)) != 0)) if len(d) > 1 else 0


def evaluate(model, seed: int, episodes: int = 20, action_mode: str = "discrete",
             scaling_mode: str = "joint") -> list[dict]:
    continuous = action_mode == "continuous"
    rows = []
    for wl in CLASSES:
        env = JointHVEnv(workload=wl, episode_len=200, seed=seed + 777, action_mode=action_mode,
                         scaling_mode=scaling_mode)
        sla, lat, cost, churn = [], [], [], 0
        reversals = dict(replica=0, cpu=0, mem=0)
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + 777 + ep)
            reps, cpus, mems, prev_a = [], [], [], None
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a if continuous else int(a))
                m = info["metrics"]; s = env._backend.s
                lat.append(m["latency"]); sla.append(m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99)
                cost.append(s.replicas * s.cpu_req / 1000.0)
                if continuous:
                    # no discrete action index to compare -- treat a material
                    # move in action-space (matches the env's own "changed" gate
                    # of +-1 replica / >1m cpu / >1Mi mem) as one churn event.
                    if prev_a is not None and float(np.max(np.abs(np.asarray(a) - prev_a))) > 0.05:
                        churn += 1
                    prev_a = np.asarray(a, dtype=np.float32).copy()
                else:
                    if prev_a is not None and int(a) != prev_a:
                        churn += 1
                    prev_a = int(a)
                reps.append(s.replicas); cpus.append(s.cpu_req); mems.append(s.mem_req)
                done = term or trunc
            reversals["replica"] += _reversals(reps)
            reversals["cpu"] += _reversals(cpus)
            reversals["mem"] += _reversals(mems)
        rows.append(dict(workload=wl, sla_pct=round(100 * np.mean(sla), 1),
                         mean_lat_ms=round(float(np.mean(lat)), 1),
                         mean_cost_vcpu=round(float(np.mean(cost)), 2),
                         churn=churn, replica_reversals=reversals["replica"],
                         cpu_reversals=reversals["cpu"], mem_reversals=reversals["mem"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGOS) + list(PANDEY_ALGOS)
                    + ["cvar-qrdqn", "cvar-bellman-qrdqn", "sac-discrete"],
                    required=True)
    ap.add_argument("--workload", default="all", help="a class or 'all' (generalist)")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cvar-alpha", type=float, default=0.25, help="risk-aversion for cvar-qrdqn (1.0 = vanilla)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load the saved model and re-run eval (e.g. to backfill new metrics)")
    ap.add_argument("--domain-rand", action="store_true",
                    help="train over the traces_train/ VM pool (domain randomization), not the single rep")
    ap.add_argument("--scaling-mode", choices=["joint", "horizontal", "vertical"], default="joint",
                    help="restrict the action axis: horizontal=replicas only, vertical=in-place "
                         "resize only (H-only/V-only action-space ablation). discrete-only.")
    ap.add_argument("--w-cost", type=float, default=None,
                    help="reward cost weight for the Pareto sweep (#20); default keeps the env constant 0.3")
    ap.add_argument("--w-churn", type=float, default=None,
                    help="reward churn weight; default keeps the env constant 0.5")
    ap.add_argument("--w-viol", type=float, default=None,
                    help="reward SLA-violation weight for the Pareto sweep (#20); default keeps the env constant 1.0")
    ap.add_argument("--no-tb", action="store_true",
                    help="disable tensorboard logging (sweep runs don't need per-run tb dirs, and lets a "
                         "tb-less venv run training)")
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure",
                    help="trace family to train and evaluate on; bitbrains runs need a 'bb' tag token")
    args = ap.parse_args()
    set_trace_family(args.trace_family, args.tag)
    tb = not args.no_tb

    pandey = args.algo in PANDEY_ALGOS
    if pandey and args.scaling_mode != "joint":
        raise SystemExit("--scaling-mode ablation is discrete-only (not for the pandey continuous env)")

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    # the H-only/V-only ablation is a distinct row in the comparison table -> fold the
    # mode into the algo label so aggregate_eval picks it up as its own algorithm.
    mode_suffix = {"horizontal": "-h-only", "vertical": "-v-only"}.get(args.scaling_mode, "")
    label = f"{args.algo}{mode_suffix}"
    name = f"{label}_{args.workload}{('_' + args.tag) if args.tag else ''}"
    mpath = os.path.join(HERE, "models", name)
    trace_pool = build_train_pool() if args.domain_rand else None
    if trace_pool:
        print(f"[domain-rand] training pool: "
              + ", ".join(f"{c}={len(v)}" for c, v in trace_pool.items()))
    action_mode = "continuous" if pandey else "discrete"
    reward_fn = "pandey" if pandey else "default"
    env = make_env(args.workload, args.seed, trace_pool=trace_pool,
                    action_mode=action_mode, reward_fn=reward_fn,
                    scaling_mode=args.scaling_mode,
                    w_cost=args.w_cost, w_churn=args.w_churn, w_viol=args.w_viol)

    if args.eval_only:
        model = load_model(args.algo, mpath, env, cvar_alpha=args.cvar_alpha, seed=args.seed)
        print(f"[eval-only] loaded {mpath}")
    else:
        if args.algo == "cvar-qrdqn":
            from cvar_qrdqn import make_cvar_qrdqn
            model = make_cvar_qrdqn(env, cvar_alpha=args.cvar_alpha, **hyperparams("qrdqn", args.seed, tb))
        elif args.algo == "cvar-bellman-qrdqn":
            from cvar_bellman_qrdqn import make_cvar_bellman_qrdqn
            model = make_cvar_bellman_qrdqn(env, cvar_alpha=args.cvar_alpha, **hyperparams("qrdqn", args.seed, tb))
        elif args.algo == "sac-discrete":
            from sac_discrete import SACDiscrete
            model = SACDiscrete(env, seed=args.seed)
        elif pandey:
            model = PANDEY_ALGOS[args.algo](env=env, **hyperparams(args.algo, args.seed, tb))
        else:
            model = ALGOS[args.algo](env=env, **hyperparams(args.algo, args.seed, tb))

        print(f"[train] {name}: {args.timesteps} steps on '{args.workload}' ({args.algo})")
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(mpath)
        print(f"[train] saved {mpath}.zip")

    rows = evaluate(model, args.seed, action_mode=action_mode, scaling_mode=args.scaling_mode)
    os.makedirs(os.path.join(HERE, "eval"), exist_ok=True)  # committed (models/ is gitignored)
    out = os.path.join(HERE, "eval", f"eval_{name}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n[eval] {name}")
    print(f"{'workload':14s}{'SLA%':>8s}{'lat_ms':>9s}{'cost':>8s}{'churn':>7s}"
          f"{'rep_osc':>8s}{'cpu_osc':>8s}{'mem_osc':>8s}")
    for r in rows:
        print(f"{r['workload']:14s}{r['sla_pct']:>8}{r['mean_lat_ms']:>9}{r['mean_cost_vcpu']:>8}"
              f"{r['churn']:>7}{r['replica_reversals']:>8}{r['cpu_reversals']:>8}{r['mem_reversals']:>8}")
    print(f"[eval] wrote {out}")


# --- TODO (risk-averse headline): subclass QRDQN policy so action selection uses
# CVaR (mean of the lowest-alpha quantiles) instead of the mean over quantiles.
# That is the distributional/risk-averse novelty; standard QR-DQN here is the base.

if __name__ == "__main__":
    main()
