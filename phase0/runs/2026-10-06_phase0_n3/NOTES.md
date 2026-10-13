# Phase 0 with three runs per cell, and the real VPA on a short time scale (2026-10-06)

Supersedes the single-run readings in `../2026-10-05_matched_setpoints/NOTES.md`. Same app,
load (constant 120 rps, 900 s), 2-worker topology and metric definitions. First runs are in
`../2026-10-05_matched_setpoints/` (the stock 60% disruptive first run is
`monitor_disruptive_default2w.csv`); runs 2 and 3, the new stock 60% in-place cell and all
short-time-scale real-VPA runs are here. Driver: `../../run_phase0_repeats.sh`.

Reproduce `grouped_node0.csv`:

    python ../../summarize_runs.py --group \
      ../2026-10-05_matched_setpoints/monitor_*_{tuned70,stock70,tuned60}.csv \
      ../2026-10-05_matched_setpoints/monitor_disruptive_default2w.csv \
      monitor_*_r[123].csv

"Real VPA, short time scale" is VPA 1.8.0 with `../../vpa/vpa_profile.sh fast`: recommender and
updater loops 15 s, CPU histogram half-life 1 min, no 12 h minimum pod lifetime; target (P90 +
15%) and 10% minimum change unchanged.

## node0, mean +- sd over 3 runs

| Vertical controller | HPA | Actuation | Replica chg / rev | Mean replicas | Mean request (m) | Total request (m) | Metric avail. (%) | Under-prov. (%) | p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| emulator 70% | tuned 70% | disruptive | 3.0+-1.7 / 0.7+-1.2 | 4.5 | 111 | 491+-24 | 85.6+-13.6 | 7.7+-4.4 | 54+-26 |
| emulator 70% | tuned 70% | in-place | 4.7+-3.2 / 3.3+-3.1 | 3.6 | 148 | 518+-24 | 98.3 | 2.8+-2.6 | 137+-28 |
| emulator 70% | stock 70% | disruptive | 2.0 / 0.0 | 4.4 | 120 | 504+-33 | 81.7+-13.2 | 7.9+-3.0 | 73+-33 |
| emulator 70% | stock 70% | in-place | 2.0 / 0.0 | 5.5 | 108 | 592+-41 | 98.3 | 2.3+-1.0 | 121+-25 |
| emulator 70% | tuned 60% | disruptive | 3.3+-2.3 / 0.7+-1.2 | 5.5 | 109 | 583+-5 | 86.7+-7.6 | 4.5+-1.5 | 130+-13 |
| emulator 70% | tuned 60% | in-place | 5.3+-1.5 / 2.7+-1.2 | 5.1 | 118 | 595+-34 | 98.3 | 3.4+-1.7 | 108+-9 |
| emulator 70% | stock 60% | disruptive | 2.7+-0.6 / 0.3+-0.6 | 5.6 | 107 | 582+-61 | 91.7+-5.0 | 5.0+-5.6 | 85+-28 |
| emulator 70% | stock 60% | in-place | 2.0 / 0.0 | 5.8 | 111 | 642+-35 | 98.3 | 2.3+-1.0 | 120+-11 |
| real VPA, short | tuned 70% | Recreate | 0.0 / 0.0 | 2.0 | 250 | 500+-7 | 88.9+-12.1 | 2.5+-0.8 | 10.3 |
| real VPA, short | tuned 70% | InPlaceOrRecreate | 5.3+-2.5 / 2.7+-3.1 | 3.6 | 159 | 544+-3 | 98.3 | 2.3+-2.0 | 9.8 |
| real VPA, short | tuned 60% | Recreate | 6.3+-1.2 / 4.0+-2.0 | 3.1 | 211 | 596+-3 | 84.1+-12.7 | 2.2+-2.3 | 10.5 |
| real VPA, short | tuned 60% | InPlaceOrRecreate | 4.7+-1.2 / 0.7+-1.2 | 4.6 | 138 | 596+-16 | 98.3 | 1.1+-2.0 | 9.7 |
| real VPA, stock flags (4 runs, n=1 each) | tuned/stock 70% | both | 1 / 0 | 3.0 | 200 | 590 | 98.3-98.9 | 0.0 | 9.6 |

Success rate was 100% (to two decimals) in every run.

## saul (second testbed), real VPA short time scale, mean +- sd over 3 runs

Same cells at 205 rps, calibrated so the total CPU usage matches node0's 120 rps (348m at 3
replicas). saul's worker vCPUs are pinned 1:1 to physical cores, node0's are not.

| HPA | Actuation | Replica chg / rev | Mean replicas | Mean request (m) | Total request (m) | Metric avail. (%) | Under-prov. (%) | p99 (ms) |
|---|---|---|---|---|---|---|---|---|
| tuned 70% | Recreate | 0.0 / 0.0 | 2.0 | 284 | 569+-19 | 71.7+-8.7 | 7.6+-2.9 | 5.1 |
| tuned 70% | InPlaceOrRecreate | 5.0+-3.0 / 2.0+-2.0 | 3.6 | 158 | 547+-18 | 98.3 | 1.1+-1.0 | 5.1 |
| tuned 60% | Recreate | 2.0 / 0.0 | 3.2 | 206 | 623+-9 | 81.7+-6.0 | 2.0+-1.9 | 5.0 |
| tuned 60% | InPlaceOrRecreate | 6.7+-0.6 / 2.0+-0.0 | 4.5 | 149 | 616+-14 | 98.5+-0.3 | 2.3+-1.0 | 4.9 |

The saul per-run files (`monitor_`, `load_` and `vpaevents_` with the `_saul_r*` suffix) are in this
directory.

## What holds

- **In-place resize keeps the HPA's signal.** Every in-place configuration had a utilization
  reading in 98.3% of the samples, on both testbeds and with both vertical controllers.
  Configurations that recreate pods had 72-92%.
- **The actuation mode decides the split between replicas and pod size, not the total.** With the
  real VPA on a short time scale, `Recreate` settled on two to three large pods (206-284m) and
  `InPlaceOrRecreate` on four to five small ones (138-159m) on both testbeds, while the total
  request stayed within 500-623m. With the HPA at 70% and `Recreate`, the HPA never scaled out
  on either testbed: the VPA absorbed the load vertically.
- **The stock VPA does not act within 15 minutes** (updater bounds and 12 h pod lifetime, see the
  2026-10-05 notes). In a 26 h run it first acted at 12 h, resizing the pods in place, after
  which the HPA added one replica and both settled (`../2026-10-06_vpa_long/NOTES.md`).
- **The stock HPA damps the replica count.** With its 300 s scale-down window, emulator runs
  reversed at most once in three runs; with the tuned HPA, 0.7-3.3 reversals per run.

## What does not hold

- **Setpoint gap as the cause of oscillation.** With three runs, the matched 70/70 tuned in-place
  cell reverses as often as the mismatched 60/70 cell (3.3+-3.1 vs 2.7+-1.2). The single-run
  reading of 2026-10-05 is withdrawn.
- **A consistent effect of the actuation mode on oscillation.** Emulator: in-place 2.7-3.3
  reversals vs disruptive 0.7. Real VPA, `Recreate` at 60%: 4.0+-2.0 on node0 but 0.0 on saul.
  Reversal counts are small (0-4 per 15 min) with standard deviations of the same size, so
  this experiment does not support a claim about which mode oscillates more.
- **In-place resize improving latency.** It does not in any configuration.

## Emulator tail latency (resolved 2026-10-08)

Emulator runs have p99 of 54-137 ms against about 10 ms with the real VPA, at similar total
requests. This is CPU throttling: the emulator sets the limit to 2.0x its request, the real VPA
keeps the template's 2.5x. At 2.5x or with no effective limit the emulator's p99 is 9.8-10.0 ms,
the same as the real VPA's (`../2026-10-08_throttle_check/NOTES.md`). The emulator p99 column above
measures the limit, not the controllers, and is not compared across vertical controllers.
