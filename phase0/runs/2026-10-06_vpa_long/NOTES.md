# Stock VPA next to the stock HPA for 26 h (2026-10-06 16:27 to 2026-10-07 18:27 UTC)

When does the upstream VPA 1.8.0 with its stock flags first act on running pods, and what does
the HPA do afterwards? In the 15-minute runs (`../2026-10-05_matched_setpoints/NOTES.md`) it never
acted.

Setup: same app, 2-worker topology and constant 120 rps as the 15-minute runs. Stock VPA profile
(`../../vpa/vpa_profile.sh stock`), `updateMode: InPlaceOrRecreate`, CPU only, 100m-800m bounds;
stock HPA at 70% (`../../hpa70.yaml`). Driver: `../../run_vpa_long.sh` with `HOURS=26`. The monitor
samples every 10 s, the VPA recommendation is traced every 60 s, and load runs in 1-hour chunks.

Files:
- `monitor_vpa-InPlaceOrRecreate_long_stock70.csv`: 9318 samples, 10 s apart.
- `vparec_InPlaceOrRecreate_long_stock70.csv`: the VPA's target, lower and upper bound.
- `load_hourly_vpa-InPlaceOrRecreate_long_stock70.txt`: client latency per hour. Hours 1-25 are
  complete; the 26th chunk was still running when the run ended and has no summary.
- `vpaevents_stream_*.txt`, `vpaevents_stream2_*.txt`: Kubernetes events. The first watch was closed
  by the API server about 30 minutes in and the second was started at 15.8 h, so events from
  0.5 h to 15.8 h were not recorded (`run_vpa_long.sh` now restarts the watch). The resize at 12 h
  is reconstructed from the monitor, which records each pod's request and the ready count.

## Result

| Replica chg / rev | Request chg / rev | Mean replicas | Total request (m) | Metric avail. (%) | Under-prov. (%) | p99 per hour (ms) | Success |
|---|---|---|---|---|---|---|---|
| 2 / 0 | 3 / 0 (one per pod) | 3.5 | 628 (600 before 12 h, 652 after) | 100.0 | 0.0 | 9.1-9.5 | 100% every hour |

(`python ../../summarize_runs.py monitor_vpa-InPlaceOrRecreate_long_stock70.csv`; p99 from the load file.)

Timeline:

| Time | Event |
|---|---|
| 1 min | HPA 2 -> 3 replicas (utilization 90% of the 200m request). |
| 1 h | VPA target 163m, lower bound 136m, upper bound 800m. The 200m request is inside the bounds, so the updater leaves the pods alone. |
| 1-12 h | Nothing changes. The target stays at 163m; the upper bound starts falling at 7 h (723m, then 313m by 26 h). Per-pod usage about 124m, HPA utilization about 62%. |
| 11.99-12.02 h | The updater resizes each pod in place from 200m to 163m, one pod a minute, as each reaches 12 h of age (the pod the HPA added at 1 min last). Ready replicas stay at 3 throughout; the pod added at 1 min was still running at 40 h with the resized limit. |
| 12.06 h | HPA utilization rises to 78% of the smaller request, above the 70% target plus 10% tolerance. HPA 3 -> 4 replicas. |
| 12.06-26 h | Nothing changes. Per-pod usage about 94m, utilization about 57%. The target stays at 163m, so the updater has nothing left to do. |

## Reading

- **The stock VPA's first action comes at 12 h, from the minimum pod lifetime.** With a request
  inside the recommendation bounds, the updater waits until a pod is 12 h old and the request is
  at least 10% off target (200m vs 163m, 18%). This is the behavior expected from the 1.8.0
  source (see the 2026-10-05 notes) and it means that, at stock flags, a 15-minute experiment
  measures the HPA alone.
- **One coupling step, and it settled.** Shrinking the pods pushed the HPA's utilization over its
  band, and the HPA added a replica: the vertical decision caused a horizontal one, and the total
  request went up 9% (600m to 652m) while total usage stayed about 375m. There were no reversals
  in 26 h, on either axis.
- **No latency cost.** The in-place resize and the scale-out did not show in the hourly client
  latency (p99 9.1-9.5 ms every hour, 100% success).

One run of one cell; it shows when the stock updater acts and that this cell settled, not how
often coupling settles in general.
