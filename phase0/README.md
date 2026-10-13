# Section 3: the HPA and a vertical controller on a real cluster

Measurements behind Section 3 of the paper: the Kubernetes HPA operating next to a
vertical controller on a k3s 1.35 cluster, with vertical changes applied either by
recreating pods (*disruptive*) or with the in-place `resize` subresource (*in place*).
The vertical controller is either a VPA-style emulator (`vpa_emulator.py`) or the
upstream VPA 1.8.0.

All objects live in the namespace `hv-phase0`; `common.guard_namespace()` makes every
mutating script refuse other namespaces, and teardown is a single namespace delete.
The upstream VPA, if installed with `vpa/install_vpa.sh`, runs in `kube-system` but only
acts on `hv-phase0`.

## Files

| File | Role |
|---|---|
| `namespace.yaml` | the isolated `hv-phase0` namespace |
| `web-service.yaml` | benchmark CPU-bound HTTP service + Service, with in-place `resizePolicy` |
| `hpa.yaml` | CPU-utilization HPA (the horizontal half of the loop) |
| `vpa_emulator.py` | VPA-style vertical actuator; `--mode disruptive\|inplace` |
| `monitor.py` | samples replicas / HPA view / CPU requests / usage → CSV |
| `load.py` | constant-rate HTTP load (stdlib only) |
| `plot_spiral.py` | replicas vs CPU-request-over-time figure |
| `run_phase0.sh` | orchestrates one run (one mode) |
| `run_spiral.sh` | one emulator run with a chosen HPA manifest and tag |
| `hpa70.yaml`, `hpa-tuned70.yaml` | HPA at 70%, matching the emulator's setpoint (stock / tuned behavior) |
| `vpa/install_vpa.sh` | installs upstream VPA 1.8.0 scoped to `hv-phase0` (`--down` removes it) |
| `vpa/vpa_profile.sh` | switches VPA flags between `stock` and `fast` (minute-scale) |
| `vpa/vpa-web-service.yaml` | VerticalPodAutoscaler object (CPU only, 100m to 800m) |
| `run_vpa_real.sh` | one run with the real VPA instead of the emulator |
| `run_phase0_7070.sh`, `run_phase0_6070.sh` | the 2026-10-05 matrix (matched and mismatched setpoints) |
| `run_phase0_repeats.sh` | repeats of that matrix plus the real VPA with `fast` flags |
| `run_vpa_long.sh` | one long run (default 26 h) of the stock VPA next to the HPA |
| `run_throttle_check.sh` | emulator at CPU limit ratios 2.0 / 2.5 / 10 and the real VPA, with CFS throttling recorded |
| `throttle_sampler.py` | per-pod CFS throttling from the workers' cAdvisor (`--summarize` for run totals) |
| `summarize_runs.py` | per-run metrics from the CSVs (`--group` for mean +- std per cell) |
| `teardown.sh` | `kubectl delete ns hv-phase0` |
| `common.py` | kubectl helpers + namespace safety guard |

## Prerequisites

- A Kubernetes 1.35 or later cluster with `metrics-server`, and `kubectl` pointing at it.
- Python 3; the run, monitor and emulator scripts use only the standard library,
  `matplotlib` is needed for plotting.

## Reproduce

```bash
./run_phase0_7070.sh          # emulator, matched 70/70 setpoints, disruptive vs in place
./run_phase0_6070.sh          # HPA 60% vs emulator 70%
./run_phase0_repeats.sh       # three runs per cell, plus the real VPA with minute-scale flags
./run_vpa_long.sh             # 26 h run of the stock VPA next to the HPA
./run_throttle_check.sh       # CPU limit ratios 2.0 / 2.5 / 10 with CFS throttling recorded
# mean +- std per cell: see runs/2026-10-06_phase0_n3/NOTES.md for the exact command
```

## Results

- `runs/2026-10-06_phase0_n3/`: the table of Section 3 (three runs per cell, both testbeds).
- `runs/2026-10-05_matched_setpoints/`: first single runs of the same matrix.
- `runs/2026-10-06_vpa_long/`: the 26 h run of the stock VPA.
- `runs/2026-10-08_throttle_check/`: tail latency against the CPU limit ratio.
