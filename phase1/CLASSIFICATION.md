# Bitbrains workload classification (Phase 1 data foundation)

Confirms a real **memory-bound** cohort exists in Bitbrains GWA-T-12 fastStorage,
so joint H+V (with memory scaling) is genuinely motivated.

## Source
GWA-T-12 Bitbrains fastStorage (1250 VMs, Aug 2013, 5-min samples).
Download: https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip
Data (c) Bitbrains IT Services Inc.; cite GWA-T-12 / CCGrid 2015 in any publication.
Raw data is gitignored (`phase1/data/`); regenerate with `classify_bitbrains.py`.

## Method
Per VM, p95 intensity on three axes:
- `cpu_int`  = p95 CPU usage %
- `mem_int`  = p95 (mem_usage / mem_provisioned) %
- `io_int`   = 100 * p95(disk+net throughput) / population-p95 (relative)

Class = dominant axis if meaningful (>=20) and beats runner-up by >=1.2x;
else `idle` (all < 20) or `mixed` (no clear winner).

## Result (1247/1250 parsed)

| Class | VMs |
|---|---|
| idle | 696 |
| **memory-bound** | **238** |
| cpu-bound | 191 |
| io-bound | 114 |
| mixed | 8 |

**Memory-bound: 238 VMs**, mem_util_p95 median 39% (range 20-894%), median
provisioned 143 MB. Of ~551 active VMs, ~43% are memory-bound: a large, clean,
separable cohort.

## Caveats
- ~56% of VMs are idle (typical for Bitbrains), excluded from experiments.
- Some `mem_util_p95` > 100%: Bitbrains reports active memory that can exceed the
  (small) provisioned value. Known quirk; reinforces "memory-saturated". Cap at a
  sensible ceiling in the simulator.
- IO uses a population-relative threshold (no natural % ceiling for throughput).

## Outputs
- `workload_classes.csv`: per-VM stats + class (committed)
- `classification_summary.png`: class counts + CPU-vs-memory scatter

## Next
- Pick representative VMs per class (e.g., median-intensity VM) to drive the load
  generator and calibrate the simulator.
- Build `JointHVEnv` around CPU-bound / memory-bound / io-bound / mixed classes.
