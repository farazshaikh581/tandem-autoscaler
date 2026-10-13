# Phase 0 redo: matched setpoints, fixed emulator, real VPA (2026-10-05)

> **Single runs, superseded by `../2026-10-06_phase0_n3/NOTES.md`.** With three runs per
> cell, the matched 70/70 in-place cell reverses as often as the 60/70 cell, so the
> setpoint-gap reading below does not hold.

Supersedes the in-place numbers in `../2026-07-02_spiral_comparison/COMPARISON.md`
and the in-place column of `../2026-10-02_default_hpa_rps120/NOTES.md`.

## Why the redo

1. **Emulator bug (in-place mode).** `vpa_emulator.py` compared its target with the
   Deployment template's CPU request. In-place resize patches pods and never
   touches the template, so the template stayed at 200m and the 15m deadband never
   engaged: the emulator resized pods on almost every 15 s tick. The July in-place
   run (56 actuations, 253 pod resizes, 22 replica reversals) measured that bug, not
   in-place resize. Fixed: in in-place mode the deadband is applied per pod against
   its live request. Disruptive mode was not affected.
2. **Setpoint mismatch.** Every July/October run paired an HPA at 60% with an
   emulator targeting 70% of mean usage. The two loops aim at different values of
   the same ratio, which is a separate cause of oscillation. This redo runs both at
   70% (`hpa70.yaml`, `hpa-tuned70.yaml`) and keeps 60/70 as the mismatched case.
3. **Emulator vs real VPA.** The upstream VPA 1.8.0 is now installed, scoped to
   `hv-phase0` (`../../vpa/install_vpa.sh`), so the same runs can use the real
   controller (`Recreate` and `InPlaceOrRecreate`).

## Setup

Constant 120 rps, 900 s, k3s v1.35.5, k8s-worker3/4 cordoned (2-worker topology of
all earlier Phase 0 runs), IO app scaled to 0. Emulator: 15 s loop, EMA 0.3, 15m
deadband, 100m to 800m, target 70% of mean usage. HPA "tuned" = stabilization windows shortened
to 15 s / 0 s (`hpa-tuned*.yaml`), "stock" = upstream default (300 s scale-down). One run per cell.
Scripts: `run_phase0_7070.sh`, `run_phase0_6070.sh`, `run_matched_2w.sh`;
`summarize_runs.py` reproduces every number below from the CSVs in this directory.

## Results (n = 1 per cell)

| HPA / vertical | Actuation | Replica chg / rev | Request chg / rev | Vertical actions (pod resizes) | Valid HPA samples | Under-prov. | Total request | p99 |
|---|---|---|---|---|---|---|---|---|
| tuned 70 / emu 70 | disruptive | 2 / 0 | 50 / 16 | 15 | 126/180 | 7.1% | 463m | 34 ms |
| tuned 70 / emu 70 | in-place | 1 / 0 | 6 / 4 | 5 (13) | 177/180 | 0.0% | 490m | 116 ms |
| stock 70 / emu 70 | disruptive | 2 / 0 | 30 / 8 | 9 | 165/180 | 10.9% | 497m | 45 ms |
| stock 70 / emu 70 | in-place | 2 / 0 | 8 / 2 | 6 (26) | 177/180 | 3.4% | 546m | 149 ms |
| tuned 60 / emu 70 | disruptive | 6 / 2 | 35 / 10 | 9 | 144/180 | 6.2% | 578m | 145 ms |
| tuned 60 / emu 70 | in-place | 5 / 2 | 17 / 8 | 13 (39) | 177/180 | 5.1% | 614m | 104 ms |
| stock 60 / emu 70 | disruptive | 2 / 0 | 12 / 2 | 3 | 174/180 | 1.7% | 579m | 53 ms |
| tuned 70 / real VPA | Recreate | 1 / 0 | 0 / 0 | 0 | 178/180 | 0.0% | 591m | 9.6 ms |
| tuned 70 / real VPA | InPlaceOrRecreate | 1 / 0 | 0 / 0 | 0 | 177/180 | 0.0% | 590m | 9.5 ms |
| stock 70 / real VPA | Recreate | 1 / 0 | 0 / 0 | 0 | 177/180 | 0.0% | 590m | 9.6 ms |
| stock 70 / real VPA | InPlaceOrRecreate | 1 / 0 | 0 / 0 | 0 | 177/180 | 0.0% | 590m | 9.6 ms |

Success rate was at least 99.94% in every run. The stock 60 in-place run of this
day used the emulator before the fix (`*_default2w_buggyemu`, kept for the record,
not reported).

## Reading

- **The July in-place spiral does not reproduce.** With the fixed emulator, the
  same 60/70 tuned configuration gives 2 replica reversals in both modes (July: 10
  disruptive, 22 in-place). The "in-place amplifies the oscillation" claim is
  withdrawn. The July disruptive run was not affected by the bug, yet the same
  configuration now gives 6 changes / 2 reversals instead of 17 / 10, so
  run-to-run variation at n = 1 is large; the repeats below are needed before any
  count is quoted.
- **The setpoint gap is what drives replica oscillation here.** At 70/70 no
  configuration reverses the replica count; at tuned 60/70 both actuation modes do
  (2 each). With n = 1 this is a direction, not a measured effect size.
- **In-place removes the metric blackout.** The HPA had a utilization reading in
  177/180 samples with in-place resize in every configuration, against 126 to 174
  with disruptive resize (rollouts leave new pods without metrics).
- **In-place does not improve tail latency in these runs.** p99 is higher with
  in-place than with disruptive resize at 70/70 (116 vs 34 ms tuned, 149 vs 45 ms
  stock) and lower at 60/70 (104 vs 145 ms). Unexplained at n = 1.
- **The real VPA never acted.** Its recommender produced a target of 143 to 203m
  within a minute, but the 1.8.0 updater only touches a pod whose request lies
  outside [lowerBound, upperBound], or that has lived 12 h and differs from the
  target by at least 10%. The upper bound is P95 x 1.15 x (1 + 1/c) with c the
  history length in days, so it sits at maxAllowed (800m) for hours and a 200m
  request is always in range. All four real-VPA runs are therefore HPA-only at
  200m, and they have the best latency (p99 about 9.6 ms) at about 590m total
  request, more than the emulator's 70/70 runs (463 to 546m).

## Follow-up

`../../run_phase0_repeats.sh` brings every emulator cell to three runs (adding stock
60 in-place with the fixed emulator) and runs the real VPA with minute-scale flags
(`../../vpa/vpa_profile.sh fast`) at tuned 60 and 70, three runs per cell.
