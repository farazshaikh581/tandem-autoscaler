# Simulator calibration against the real cluster (issue #1)

Date: 2026-07-03/04. Cluster: hv-phase0 namespace, web-service pinned to a fixed
configuration: HPA off, replicas=2, cpu requests=limits=200m per pod (total
capacity 400m) so the CFS throttle point equals the sim's notion of capacity.
Load: `calibrate_sweep.py` steps offered rps through 10 levels, 30 s warmup +
90 s measurement each, per-request latency from the rewritten `phase0/load.py`
(issue #5), per-pod CPU from metrics-server.

## What the sweep showed (sweep_v2.csv)

| rps | rho (demand) | mean ms | p99 ms | success |
|----:|----:|----:|----:|----:|
| 10–80 | 0.08–0.65 | 5.2 | 6.0–8.3 | 1.000 |
| 100 | 0.82 | 6.0 | 37 | 1.000 |
| 115 | 0.94 | 365 | 3212 | 0.955 |
| 130–160 | 1.06–1.31 | ~400 | ~3300 | ~0.95 |

Latency is FLAT until the CPU limit throttles, then jumps to a ~400 ms plateau
(the 5 s client timeout bounds the tail). It is not a smooth 1/(1-rho) curve.

`sweep_2026-07-03_2150.csv` (v1) is kept as a cautionary artifact: the load
generator itself was the bottleneck (too few workers), so every level above
40 rps looked saturated. v2 fixed the driver; only v2 is used for fitting.

## Fitted model (fit_constants.py, output in fit_v2.txt)

```
rho     = rps * M_PER_RPS / total_limit      # demand-based, can exceed 1
knee    = sigmoid((rho - RHO_KNEE) / KNEE_WIDTH)
latency = L0_MS + (LAT_SAT_MS - L0_MS) * knee
success = 1 - SAT_DROP * knee
```

| constant | value | meaning |
|---|---:|---|
| M_PER_RPS | 3.27 | mcores per rps (unsaturated levels) |
| L0_MS | 5.26 | base latency below the knee |
| LAT_SAT_MS | 403.3 | saturation plateau |
| RHO_KNEE | 0.910 | knee position |
| KNEE_WIDTH | 0.015 | knee sharpness |
| SAT_DROP | 0.052 | success loss on the plateau |

Fit quality: latency MAPE **0.8%** across all 10 levels (the old M/M/1 form
scored 46% on the unsaturated levels alone and underpredicted saturation by
5x); success MAE 0.0007.

## What changed in the sim (envs/joint_hv_env.py)

`SimBackend.step_physics()` now uses the knee model with the fitted constants.
The memory-pressure terms (BETA_MEM, OOM_DROP) and per-class INTENSITY remain
[CAL]: the phase0 app is CPU-only, so the memory axis has no ground truth yet.
Beyond the measured range (rho > 1.31) success decays linearly (OVERLOAD_DROP,
extrapolated). Smoke test after the change: memory-bound differentiation holds
(HPA-only 0% SLA / 69 ms vs memory-aware 98% / 7 ms; cpu-bound both 100%).

NOTE: the Phase 1 six-algo comparison (phase1/eval/comparison.csv) was trained
on the pre-calibration sim and needs a re-run under this model before any
result is quoted in the paper.

The IO axis was never fit either: `SimBackend` loads a trace's `io` field but
never reads it, so "io-bound" is purely the per-class INTENSITY coefficient
above, same [CAL] status as the memory terms. Issue #12 exercised the IO axis
for real on a second app (`web-service-io`, genuine write+fsync work) but
deliberately did NOT recalibrate the sim against it -- that thread is a
sim-to-real *transfer* test of the already-trained policy, not a fit, so this
line item stays [CAL].

## Reproduce

```
cd phase1/calibration
python3 calibrate_sweep.py --levels 10,20,40,60,80,100,115,130,145,160 --measure 90 --warmup 30
python3 fit_constants.py sweep_<ts>.csv
```

## Dense and deployed-configuration sweeps (2026-10-12)

`run_calibration_dense.sh`, three repetitions interleaved A,B,A,B,A,B, same
fixed conditions as above (k8s-worker3/4 cordoned, IO app scaled to 0).
`limit_total_m` in these CSVs counts every pod present when the sweep starts,
including one still terminating from the previous configuration, so it reads
900/1200 where the serving pods had 400/1000m; rho below is recomputed from the
offered rate.

**A: request = limit = 200m, dense across the knee** (`sweep_dense_A_r{1,2,3}.csv`):

| rps | rho | mean ms (3 runs) | success |
|----:|----:|---:|---:|
| 95 | 0.78 | 7.5 / 19.3 / 10.8 | 0.996 |
| 100 | 0.82 | 8.9 / 13.4 / 13.1 | 1.000 |
| 103 | 0.84 | 22.6 / 21.6 / 50.5 | 1.000 |
| 106 | 0.87 | 74 / 149 / 93 | 0.999 |
| 109 | 0.89 | 382 / 365 / 352 | 0.971 |
| 112-122 | 0.92-1.0 | ~390-410 | ~0.96 |

The transition sits earlier than the v2 fit: measured mean latency crosses
20 ms at rho ~0.83, the model at 0.86, and between 0.84 and 0.92 the model
underestimates latency by up to 4x (out-of-sample latency MAPE on these levels
41%; the 0.8% figure above is in-sample). Some of this may be drift: at 100 rps
v2 measured 6.0 ms in July. Refit on all 37 levels (v2 + dense, L0 fixed from
rho < 0.7): RHO_KNEE 0.88, KNEE_WIDTH 0.0125, LAT_SAT_MS 397.2, SAT_DROP 0.042,
20 ms crossing at rho 0.84, latency MAPE 15%.

**B: deployed configuration, request 200m / limit 500m** (`sweep_deployed_B_r{1,2,3}.csv`,
`--procs 2` load driver): latency stays at base level (<= 7 ms mean) up to
265 rps and saturates at 280-295 rps, i.e. at limit-based rho 0.87-0.92. The
knee follows the CPU limit, not the request: with node headroom a deployed pod
serves ~2.6x what the model (capacity = request) allows.

**Sensitivity of the reported results** (`knee_sensitivity.py`, output in
`knee_sensitivity/`): every trained controller re-evaluated with the standard
20-episode protocol under the nominal and the refitted constants, no
retraining, both trace families (QR-DQN / CVaR-Bellman seeds 0-6, CVaR-QR-DQN /
PPO / DQN seeds 0-2, all baselines; MPC keeps planning on the nominal model).
The nominal pass reproduces all 248 committed rows exactly. Under the refit:
QR-DQN and CVaR-Bellman QR-DQN change by <= 0.14 SLA points in every class on
both families; CVaR-QR-DQN/DQN <= 0.34; PPO -4.8 on Bitbrains mixed (<= 0.6
elsewhere); MPC <= 0.1; CoScal <= 0.5; the HPA/VPA, predictive and
SLO-cost-aware baselines lose up to 4.0 (Bitbrains mixed; <= 0.8 on Azure).
The constants in envs/joint_hv_env.py are left unchanged.
