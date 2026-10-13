"""The multi-function ablation (#22): does the Phase 1 method hold at N functions?

Two axes, and both are necessary:

  N    2, 3, 4 functions, as nested prefixes of DEFAULT_CLASSES, so growing N only
       adds a function and never swaps the mix.
  phi  capacity relative to measured demand (calibrate_capacity.py). Without this
       axis the ablation is vacuous: at non-binding capacity N functions provably
       do not interact, and the run reproduces the committed single-function
       numbers by construction (smoke_test_multifn.py gate 2 proves it bit-exactly).

Controllers are the committed Phase 1 ones, unmodified, one instance per function,
with no knowledge of each other -- which is what a cluster running N functions with
N independently configured autoscalers actually looks like.

    python eval_multifn.py                      # full sweep
    python eval_multifn.py --only hpa-vpa,qrdqn --episodes 3
    python eval_multifn.py --trace-family bitbrains   # legacy profiles only, see below

On the Bitbrains traces only the legacy profiles exist: the azure profiles are the
measured duration/memory of the Azure function driving each trace, and a Bitbrains
VM has no such function. Bitbrains outputs carry a "_bitbrains" suffix.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_fn_env import DEFAULT_CLASSES, MultiFnEnv, jain          # noqa: E402
from calibrate_capacity import PHI_SWEEP, make_hpa_vpa, node_cpu_for, run  # noqa: E402
from envs.joint_hv_env import (                                      # noqa: E402
    ACTIONS, TRACE_FAMILIES, JointHVEnv, set_trace_family,
)
from train import _reversals, load_model                             # noqa: E402
from eval_baselines import (                                         # noqa: E402
    ClassicalController, PredictiveController, SloCostAwareController,
)

HERE = os.path.dirname(os.path.abspath(__file__))
P1_MODELS = os.path.join(HERE, "..", "phase1", "models")
N_SWEEP = [2, 3, 4]

# (label, algo, checkpoint) for the RL rows -- the azure headline checkpoints, i.e.
# exactly the models behind the committed comparison_azure.csv rows.
RL_ROWS = [
    ("qrdqn", "qrdqn", "qrdqn_all_azure"),
    ("cvar-bellman-qrdqn", "cvar-bellman-qrdqn", "cvar-bellman-qrdqn_all_azure"),
    ("ppo", "ppo", "ppo_all_azure"),
]
# On Bitbrains: the original Phase 1 checkpoints behind comparison.csv.
RL_ROWS_BITBRAINS = [
    ("qrdqn", "qrdqn", "qrdqn_all"),
    ("cvar-bellman-qrdqn", "cvar-bellman-qrdqn", "cvar-bellman-qrdqn_all"),
    ("ppo", "ppo", "ppo_all"),
]


def classical_rows(n: int):
    return [
        ("hpa-only", [ClassicalController(vpa=False, target_util=0.70) for _ in range(n)]),
        ("hpa-vpa", [ClassicalController(vpa=True, target_util=0.70) for _ in range(n)]),
        ("slo-cost-aware-vpa", [SloCostAwareController(vpa=True) for _ in range(n)]),
        ("predictive-vpa", [PredictiveController(vpa=True, target_util=0.70) for _ in range(n)]),
    ]


class RLController:
    """Adapts an SB3 policy to the per-function controller interface.

    The policy is the committed single-function checkpoint, untouched, and it reads
    the unmodified 12-feature Phase 1 observation. It has no channel through which
    it could learn about a neighbour -- that is the point of the ablation.
    """

    def __init__(self, model):
        self.model = model

    def reset(self) -> None:
        pass

    def act_obs(self, obs) -> int:
        return int(self.model.predict(obs, deterministic=True)[0])


def load_rl(algo: str, ckpt: str, n: int):
    path = os.path.join(P1_MODELS, ckpt)
    if not os.path.exists(path + ".zip"):
        return None
    probe = JointHVEnv(workload="all", episode_len=200, seed=0)
    model = load_model(algo, path, probe)
    return [RLController(model) for _ in range(n)]


def evaluate(controllers, classes, node_cpu_m, seed, episodes, episode_len=200,
             profiles="legacy"):
    """Roll out one controller per function and report per-function metrics."""
    n = len(classes)
    env = MultiFnEnv(classes=classes, node_cpu_m=node_cpu_m, seed=seed,
                     episode_len=episode_len, profiles=profiles)
    is_rl = hasattr(controllers[0], "act_obs")
    sla = [[] for _ in range(n)]
    lat = [[] for _ in range(n)]
    cost = [[] for _ in range(n)]
    churn = [0] * n
    rev = [dict(replica=0, cpu=0, mem=0) for _ in range(n)]
    denied = [0] * n
    deferred = [0] * n
    demand = []   # what this controller ASKS for, regardless of what it is granted
    steps = 0

    for ep in range(episodes):
        obs, infos = env.reset(seed=seed + ep)
        for c in controllers:
            c.reset()
        series = [dict(rep=[], cpu=[], mem=[]) for _ in range(n)]
        prev_a = [None] * n
        for _ in range(episode_len):
            if is_rl:
                acts = [c.act_obs(o) for c, o in zip(controllers, obs)]
            else:
                acts = [ACTIONS.index(c.act(inf["metrics"], s))
                        for c, inf, s in zip(controllers, infos, env.states)]
            obs, _, _, _, infos = env.step(acts)
            steps += 1
            demand.append(sum(s.target_replicas * s.cpu_req for s in env.states))
            for i, (inf, s) in enumerate(zip(infos, env.states)):
                m = inf["metrics"]
                sla[i].append(bool(m["sla_ok"]))
                lat[i].append(m["latency"])
                cost[i].append(m["cost_vcpu"])
                denied[i] += int(m["denied_replicas"] > 0)
                deferred[i] += int(m["resize_deferred"])
                if prev_a[i] is not None and acts[i] != prev_a[i]:
                    churn[i] += 1
                prev_a[i] = acts[i]
                series[i]["rep"].append(s.replicas)
                series[i]["cpu"].append(int(s.cpu_granted if s.cpu_granted is not None
                                            else s.cpu_req))
                series[i]["mem"].append(int(s.mem_granted if s.mem_granted is not None
                                            else s.mem_req))
        for i in range(n):
            rev[i]["replica"] += _reversals(series[i]["rep"])
            rev[i]["cpu"] += _reversals(series[i]["cpu"])
            rev[i]["mem"] += _reversals(series[i]["mem"])

    rows = []
    for i, wl in enumerate(classes):
        rows.append(dict(
            workload=wl,
            sla_pct=round(100 * float(np.mean(sla[i])), 1),
            mean_lat_ms=round(float(np.mean(lat[i])), 1),
            mean_cost_vcpu=round(float(np.mean(cost[i])), 2),
            churn=churn[i], replica_reversals=rev[i]["replica"],
            cpu_reversals=rev[i]["cpu"], mem_reversals=rev[i]["mem"],
            denied_pct=round(100.0 * denied[i] / max(steps, 1), 1),
            deferred_pct=round(100.0 * deferred[i] / max(steps, 1), 1)))
    return rows, float(np.percentile(demand, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated controller labels")
    ap.add_argument("--anchor", default="shared,self",
                    help="shared = one cluster sized from conventional HPA+VPA demand; "
                         "self = capacity scaled to each controller's own p95 demand")
    ap.add_argument("--profiles", default=None,
                    help="comma-separated: legacy (one shared service profile, the "
                         "Phase 1 model) and/or azure (measured per-function service "
                         "time, SLO and working set). Default: legacy,azure on the "
                         "azure traces, legacy on bitbrains")
    ap.add_argument("--trace-family", choices=TRACE_FAMILIES, default="azure")
    args = ap.parse_args()
    set_trace_family(args.trace_family)
    bitbrains = args.trace_family == "bitbrains"
    if args.profiles is None:
        args.profiles = "legacy" if bitbrains else "legacy,azure"
    if bitbrains and "azure" in args.profiles:
        raise SystemExit("azure profiles need the Azure function behind each trace; "
                         "use --profiles legacy on bitbrains")
    rl_rows = RL_ROWS_BITBRAINS if bitbrains else RL_ROWS
    sfx = "_bitbrains" if bitbrains else ""
    want = {x.strip() for x in args.only.split(",") if x.strip()} or None
    modes = [m.strip() for m in args.profiles.split(",") if m.strip()]
    anchors = [a.strip() for a in args.anchor.split(",") if a.strip()]

    os.makedirs(os.path.join(HERE, "eval"), exist_ok=True)
    fields = ["profiles", "anchor", "algo", "n_fns", "phi", "node_cpu_m", "workload", "sla_pct",
              "mean_lat_ms", "mean_cost_vcpu", "churn", "replica_reversals",
              "cpu_reversals", "mem_reversals", "denied_pct", "deferred_pct"]
    out_rows, summary = [], []

    for mode in modes:
        for n in N_SWEEP:
            classes = DEFAULT_CLASSES[:n]
            # capacity is anchored to what the controllers actually ask for at THIS n
            # and THIS profile set, so phi means the same thing across both axes
            ref = run(make_hpa_vpa(n), classes, None, args.seed,
                      max(2, args.episodes // 2), profiles=mode)
            print(f"\n=== {mode}  N={n}  {classes}  ref p95 {ref['p95']:.0f}m ===")

            rows_for_n = list(classical_rows(n))
            for label, algo, ckpt in rl_rows:
                ctrls = load_rl(algo, ckpt, n)
                if ctrls is None:
                    print(f"  [skip] {label}: {ckpt}.zip not found")
                    continue
                rows_for_n.append((label, ctrls))

            # Per-controller demand at infinite capacity. The RL policies ask for
            # ~8x what HPA+VPA does, so a single cluster size means very different
            # pressure for each -- hence the two anchorings below.
            own_p95 = {}
            for label, ctrls in rows_for_n:
                if want and label not in want:
                    continue
                _, own_p95[label] = evaluate(ctrls, classes, None, args.seed,
                                             max(2, args.episodes // 2), profiles=mode)

            for anchor in anchors:
                for phi in PHI_SWEEP:
                  for label, ctrls in rows_for_n:
                    if want and label not in want:
                        continue
                    # "shared": one cluster sized from conventional HPA+VPA demand --
                    #   what happens if you drop a new controller into today's cluster.
                    # "self": capacity scaled to this controller's own appetite --
                    #   isolates contention-handling from how much it wants.
                    base = ref["p95"] if anchor == "shared" else own_p95[label]
                    ncpu = node_cpu_for(phi, base)
                    rs, _ = evaluate(ctrls, classes, ncpu, args.seed, args.episodes,
                                     profiles=mode)
                    per_fn = [r["sla_pct"] for r in rs]
                    common = dict(profiles=mode, anchor=anchor, algo=label, n_fns=n,
                                  phi=("inf" if phi is None else phi),
                                  node_cpu_m=("inf" if ncpu is None else int(ncpu)))
                    for r in rs:
                        out_rows.append(dict(**common, **r))
                    summary.append(dict(
                        **common,
                        sla_pct=round(float(np.mean(per_fn)), 1),
                        sla_min=round(min(per_fn), 1),
                        sla_max=round(max(per_fn), 1),
                        jain=round(jain(per_fn), 4),
                        mean_cost_vcpu=round(
                            float(np.mean([r["mean_cost_vcpu"] for r in rs])), 2),
                        replica_reversals=sum(r["replica_reversals"] for r in rs)))
                    print(f"  [{anchor:<6}] phi={str(phi):>4} cap={str(ncpu):>6}  "
                          f"{label:<20} SLA {np.mean(per_fn):5.1f}%  "
                          f"(min {min(per_fn):5.1f}%)  Jain {jain(per_fn):.3f}")

    p1 = os.path.join(HERE, "eval", f"multifn_ablation{sfx}.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    p2 = os.path.join(HERE, "eval", f"multifn_summary{sfx}.csv")
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
