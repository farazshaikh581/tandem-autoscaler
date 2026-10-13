# Emulator tail latency is CPU throttling from its limit ratio (2026-10-08)

In `../2026-10-06_phase0_n3/NOTES.md` the emulator runs had p99 of 54-137 ms against about 10 ms
with the real VPA, at similar total requests. The emulator sets the CPU limit to 2.0x its request
(about 110m -> 220m); the real VPA scales the template's limit with the request and keeps its 2.5x
ratio. This check changes only the emulator's limit ratio and records CFS throttling in every run.

Setup: node0, 2 workers, 120 rps, 900 s, tuned HPA at 70%, emulator at 70% (as in the n3 runs).
Emulator in place at limit ratio 2.0 (as before), 2.5 (the real VPA's) and 10 (in effect no limit),
emulator disruptive at 2.0 and 10, and the minute-scale real VPA (`InPlaceOrRecreate`) as the
reference. Three runs each, interleaved. Driver: `../../run_throttle_check.sh` (sets
`VPA_LIMIT_RATIO` in `../../run_spiral.sh`). Throttling comes from each worker's cAdvisor
(`../../throttle_sampler.py`, 15 s): the share of CFS periods in which the `web` container was
throttled, and the total time it was throttled, summed over the run's pods.

Reproduce: `python ../../summarize_runs.py --group monitor_*.csv` (`grouped.csv`) and
`python ../../throttle_sampler.py --summarize throttle_*.csv` (`throttle_summary.csv`).

## Result, mean +- sd over 3 runs

| Vertical controller | Actuation | Limit ratio | Throttled periods (%) | Throttled time (s) | p99 (ms) | Mean (ms) | Replica chg / rev | Total request (m) | Metric avail. (%) | Under-prov. (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| emulator | in-place | 2.0 | 1.50+-0.21 | 57.7+-8.4 | 95.7+-12.0 | 11.3+-1.6 | 4.7+-1.5 / 3.3+-1.2 | 507+-16 | 98.3 | 2.8+-1.0 |
| emulator | in-place | 2.5 | 0.29+-0.12 | 2.0+-1.5 | 9.8+-0.2 | 4.9 | 3.3+-2.5 / 2.0+-2.0 | 501+-16 | 98.3 | 1.1+-1.0 |
| emulator | in-place | 10 | 0.04+-0.01 | 0.7+-0.2 | 9.8+-0.1 | 4.9+-0.1 | 3.7+-1.2 / 2.7+-1.2 | 501+-15 | 98.3 | 3.4 |
| emulator | disruptive | 2.0 | 3.57+-1.36 | 98.3+-32.6 | 96.5+-43.3 | 11.9+-2.5 | 3.0+-1.7 / 0.7+-1.2 | 452+-36 | 70.5+-21.7 | 14.1+-9.5 |
| emulator | disruptive | 10 | 0.00 | 0.0 | 10.0+-0.2 | 4.9+-0.1 | 2.0 / 0.0 | 509+-12 | 91.1+-5.4 | 4.3+-0.9 |
| real VPA, short | InPlaceOrRecreate | 2.5 (template) | 0.17+-0.06 | 2.8+-0.8 | 9.8+-0.1 | 4.9+-0.1 | 5.0+-2.0 / 1.3+-1.2 | 542+-2 | 98.3 | 2.3+-2.0 |

Success rate was at least 99.98% in every run.

## Reading

- **The emulator's tail latency was CPU throttling.** At a limit of 2.0x the request, pods were
  throttled in 1.5-3.6% of CFS periods and p99 was 96 ms. At 2.5x (the real VPA's ratio) or with
  no effective limit, throttling fell to 0.3% or less and p99 to 9.8-10.0 ms, the same as the real
  VPA. The median did not move (5.0-5.1 ms everywhere): throttling hits the tail.
- **With the limit ratio matched, tail latency does not depend on the vertical controller or the
  actuation mode.** Every configuration at 2.5x or more had p99 of 9.8-10.0 ms. The emulator p99
  values in the n3 table (54-137 ms, all at 2.0x) measure the limit, not the controllers, and
  should not be compared with the real VPA's.
- **What survives without throttling.** In-place resize still kept the HPA's utilization signal
  available more often than recreating pods (98.3% vs 91.1+-5.4% at 10x), and the in-place cell
  still reversed more than the disruptive one (2.7+-1.2 vs 0.0 at 10x), as in the n3 runs.
- **What does not.** At 2.0x, disruptive runs lost more signal and were under-provisioned more
  (70.5% and 14.1%, with large spread) than at 10x (91.1% and 4.3%). Without throttling the
  under-provisioning gap between disruptive and in-place resize is about one point (4.3% vs
  3.4%), so the emulator's disruptive-vs-in-place difference in under-provisioning is not
  attributed to the actuation mode.
