"""Correctness gates for MultiFnEnv (#22).

The ablation is only worth reporting if the env provably reduces to Phase 1 when
capacity does not bind. Gates 1-3 establish that; gate 4 establishes that the
scheduler cannot hand out capacity it does not have.

    python smoke_test_multifn.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_fn_env import MultiFnEnv                                 # noqa: E402
from calibrate_capacity import PHI_SWEEP, make_hpa_vpa, node_cpu_for, run  # noqa: E402
from envs.joint_hv_env import ACTIONS, JointHVEnv                    # noqa: E402

CLASSES = ["cpu-bound", "io-bound", "mixed", "memory-bound"]
STEPS = 200


def _rollout_single(wl: str, seed: int, acts: list[int]):
    env = JointHVEnv(workload=wl, episode_len=STEPS, seed=seed)
    obs, info = env.reset(seed=seed)
    tr = [(obs.copy(), 0.0, dict(info["metrics"]))]
    for a in acts:
        obs, r, _, _, info = env.step(a)
        tr.append((obs.copy(), float(r), dict(info["metrics"])))
    return tr


def _rollout_multi(classes: list[str], seed: int, acts_per_fn: list[list[int]],
                   node_cpu_m=None, strict=False):
    env = MultiFnEnv(classes=classes, node_cpu_m=node_cpu_m, seed=seed, episode_len=STEPS,
                     strict=strict)
    obs, infos = env.reset(seed=seed)
    trs = [[(o.copy(), 0.0, dict(inf["metrics"]))] for o, inf in zip(obs, infos)]
    for t in range(len(acts_per_fn[0])):
        obs, rews, _, _, infos = env.step([acts_per_fn[i][t] for i in range(len(classes))])
        for i, (o, r, inf) in enumerate(zip(obs, rews, infos)):
            trs[i].append((o.copy(), float(r), dict(inf["metrics"])))
    return trs


def _assert_same(a, b, label: str):
    for t, ((oa, ra, ma), (ob, rb, mb)) in enumerate(zip(a, b)):
        np.testing.assert_allclose(oa, ob, rtol=0, atol=0, err_msg=f"{label} obs @t={t}")
        assert ra == rb, f"{label} reward @t={t}: {ra} != {rb}"
        for k in ("latency", "success", "rho", "load_frac", "cpu_util", "mem_pressure"):
            assert ma[k] == mb[k], f"{label} metrics[{k}] @t={t}: {ma[k]} != {mb[k]}"


def gate1_n1_bit_exact() -> None:
    """MultiFnEnv(n=1, capacity=inf) must be bit-identical to JointHVEnv."""
    for wl in CLASSES:
        rng = np.random.default_rng(0)
        acts = [int(rng.integers(len(ACTIONS))) for _ in range(STEPS)]
        single = _rollout_single(wl, 777, acts)
        multi = _rollout_multi([wl], 777, [acts])[0]
        _assert_same(single, multi, f"gate1[{wl}]")
    print("  gate 1  N=1 bit-exact vs JointHVEnv ......... PASS")


def gate2_independence_at_infinite_capacity() -> None:
    """N=4 at phi=inf must equal four independent JointHVEnv rollouts.

    This is also the reason the ablation cannot be run at phi=inf alone: with
    capacity non-binding the functions provably do not interact, so such a result
    would reproduce the committed numbers by construction.
    """
    rng = np.random.default_rng(1)
    acts = [[int(rng.integers(len(ACTIONS))) for _ in range(STEPS)] for _ in CLASSES]
    multi = _rollout_multi(CLASSES, 777, acts)
    for i, wl in enumerate(CLASSES):
        single = _rollout_single(wl, 777 + 1000 * i, acts[i])
        _assert_same(single, multi[i], f"gate2[{wl}]")
    print("  gate 2  N=4 independent at phi=inf .......... PASS")


def gate3_contention_bites() -> None:
    """SLA must fall monotonically as capacity tightens.

    Capacity is set from *measured* demand (phi, see calibrate_capacity.py) rather
    than from a guessed millicore figure, and the controller is the real Phase 1
    HPA+VPA -- one instance per function -- so the only thing varying across the
    sweep is the scheduler. If SLA does not move, the pool is mis-sized and the
    ablation would be vacuous, so this gates the issue.
    """
    ref = run(make_hpa_vpa(len(CLASSES)), CLASSES, None, seed=0, episodes=2)
    slas = []
    for phi in PHI_SWEEP:
        ncpu = node_cpu_for(phi, ref["p95"])
        r = run(make_hpa_vpa(len(CLASSES)), CLASSES, ncpu, seed=0, episodes=2, strict=True)
        slas.append(r["sla_pct"])
    labels = ["inf" if p is None else str(p) for p in PHI_SWEEP]
    print("  gate 3  contention-onset (phi): "
          + "  ".join(f"{l}={s:.1f}%" for l, s in zip(labels, slas)))
    for i in range(len(slas) - 1):
        assert slas[i + 1] <= slas[i] + 1.0, (
            f"SLA rose as capacity tightened: phi={labels[i]} {slas[i]:.1f}% -> "
            f"phi={labels[i+1]} {slas[i+1]:.1f}%; capacity is mis-sized")
    assert slas[-1] < slas[0] - 5.0, (
        f"tightest capacity barely moved SLA ({slas[0]:.1f}% -> {slas[-1]:.1f}%); "
        "contention does not bite, so the ablation would be vacuous")
    print("  gate 3  monotone degradation ................ PASS")


def gate4_conservation() -> None:
    """The scheduler must never over-commit a node, under adversarial demand."""
    env = MultiFnEnv(classes=CLASSES, node_cpu_m=1200, seed=9, episode_len=STEPS,
                     strict=True)
    env.reset(seed=9)
    rng = np.random.default_rng(3)
    grow = [a for a, (r, c, m) in enumerate(ACTIONS) if r >= 0 and c >= 0]
    for _ in range(STEPS):
        env.step([int(rng.choice(grow)) for _ in CLASSES])  # everyone grows, always
    s = env.sched
    for n in range(s.n_nodes):
        c, m = s._alloc(n)
        assert c <= s.node_cpu + 1e-6, f"node {n} over-committed: {c} > {s.node_cpu}"
    for st in env.states:
        assert st.admitted_replicas <= st.target_replicas
    denied = sum(1 for st in env.states if st.admitted_replicas < st.target_replicas)
    print(f"  gate 4  conservation under max demand ....... PASS "
          f"({denied}/{len(CLASSES)} fns Pending at the end)")


def gate5_deferred_resize_occurs() -> None:
    """The vertical denial mode must actually be reachable, not just implemented."""
    env = MultiFnEnv(classes=CLASSES, node_cpu_m=1000, seed=11, episode_len=STEPS)
    env.reset(seed=11)
    up = ACTIONS.index((0, 100, 0))  # pure in-place CPU growth
    seen = 0
    for _ in range(STEPS):
        _, _, _, _, infos = env.step([up] * len(CLASSES))
        seen += sum(bool(i["metrics"]["resize_deferred"]) for i in infos)
    assert seen > 0, "no resize was ever deferred; the fragmentation path is dead code"
    print(f"  gate 5  deferred resize reachable ........... PASS ({seen} deferrals)")


def gate6_azure_profiles_are_distinct_and_usable() -> None:
    """The measured per-function profiles must differ materially AND still be
    achievable. A profile set where some function can never meet its own SLO would
    make every later number a property of the profile, not of contention."""
    env = MultiFnEnv(classes=CLASSES, profiles="azure", seed=0)
    env.reset(seed=0)
    l0 = [e._backend.l0 for e in env.envs]
    sla = [e.sla_ms for e in env.envs]
    spread = max(l0) / min(l0)
    assert spread > 5.0, f"azure profiles barely differ (L0 spread {spread:.1f}x)"
    for a, b in zip(l0, sla):
        assert b > a, f"SLO {b} below base service time {a}: unachievable by construction"
    r = run(make_hpa_vpa(len(CLASSES)), CLASSES, None, seed=0, episodes=2, profiles="azure")
    # With capacity non-binding a well-served function reaching 100% is the correct
    # outcome, not a degenerate one. What must not happen is a function that its own
    # SLO leaves permanently unreachable, which would bake a constant into every
    # later row. Nor should every function be pinned at 100%, which would leave the
    # capacity axis nothing to move.
    assert all(x > 20.0 for x in r["per_fn_sla"]), (
        f"a function is starved by its own profile: {r['per_fn_sla']}")
    assert any(x < 100.0 for x in r["per_fn_sla"]), (
        f"every function trivially saturated at 100%: {r['per_fn_sla']}")
    print(f"  gate 6  azure profiles distinct+usable ...... PASS "
          f"(L0 spread {spread:.0f}x, SLA {r['sla_pct']:.1f}%)")


if __name__ == "__main__":
    print("MultiFnEnv correctness gates (#22)")
    gate1_n1_bit_exact()
    gate2_independence_at_infinite_capacity()
    gate3_contention_bites()
    gate4_conservation()
    gate5_deferred_resize_occurs()
    gate6_azure_profiles_are_distinct_and_usable()
    print("all gates passed")
