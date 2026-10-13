"""Smoke test for JointHVEnv: API sanity + does the sim reproduce the differentiation?

We compare two hand policies on the MEMORY-BOUND workload:
  HPA-only    : only add/remove replicas (never touches memory) -> should FAIL
  memory-aware: raises memory under pressure                    -> should SUCCEED
If the sim is meaningful, memory-aware >> HPA-only on the memory-bound class, but
they are similar on the cpu-bound class.
"""
import numpy as np

from envs.joint_hv_env import ACTIONS, JointHVEnv, SLA_MEAN_LATENCY_MS

def aidx(dr, dc, dm):
    return ACTIONS.index((dr, dc, dm))

def hpa_only(m, s_pending):
    if m["latency"] > SLA_MEAN_LATENCY_MS and s_pending == 0:
        return aidx(1, 0, 0)
    if m["cpu_util"] < 0.3:
        return aidx(-1, 0, 0)
    return aidx(0, 0, 0)

def mem_aware(m, s_pending):
    if m["mem_pressure"] > 0.01:
        return aidx(0, 0, 64)
    if m["latency"] > SLA_MEAN_LATENCY_MS and s_pending == 0:
        return aidx(1, 0, 0)
    if m["cpu_util"] < 0.3 and m["mem_pressure"] == 0:
        return aidx(-1, 0, 0)
    return aidx(0, 0, 0)

def run(workload, policy, seed=0, steps=200):
    env = JointHVEnv(workload=workload, episode_len=steps, seed=seed)
    obs, info = env.reset(seed=seed)
    m = info["metrics"]; lat, ok = [], []
    done = False
    while not done:
        pend = env._backend.s.target_replicas - env._backend.s.replicas
        obs, r, term, trunc, info = env.step(policy(m, pend))
        m = info["metrics"]
        lat.append(m["latency"]); ok.append(m["latency"] < SLA_MEAN_LATENCY_MS and m["success"] >= 0.99)
        assert env.observation_space.contains(obs), "obs out of space!"
        assert np.isfinite(r), "non-finite reward!"
        done = term or trunc
    return np.mean(ok) * 100, np.mean(lat)

def main():
    print("== API sanity (random rollout per class) ==")
    for wl in ["cpu-bound", "memory-bound", "io-bound", "mixed"]:
        env = JointHVEnv(workload=wl, episode_len=100, seed=1)
        obs, _ = env.reset(); tot = 0.0
        for _ in range(100):
            obs, r, term, trunc, _ = env.step(env.action_space.sample()); tot += r
            if term or trunc: break
        print(f"  {wl:13s} ok: obs{obs.shape} rand-return={tot:7.1f}")

    print("\n== differentiation check (SLA%, mean latency) ==")
    print(f"{'workload':14s}{'HPA-only':>22s}{'memory-aware':>22s}")
    for wl in ["memory-bound", "cpu-bound"]:
        h_sla, h_lat = run(wl, hpa_only)
        m_sla, m_lat = run(wl, mem_aware)
        print(f"{wl:14s}{f'{h_sla:.0f}% / {h_lat:.0f}ms':>22s}{f'{m_sla:.0f}% / {m_lat:.0f}ms':>22s}")

if __name__ == "__main__":
    main()
