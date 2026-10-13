# KEDA production-autoscaler baseline (real cluster)

Measures the RL policy against the actual [KEDA](https://keda.sh) autoscaler
operator running live on the k3s testbed, not just the in-sim rule-based baselines
(`hpa-only`, `slo-cost-aware`, `predictive`). Same H-only / +VPA pairing as those
rows:

| variant    | horizontal            | vertical (memory request)                          | manifest                |
|------------|-----------------------|----------------------------------------------------|-------------------------|
| `keda`     | KEDA, cpu util        | none (driver passive, request stays at baseline)   | `scaledobject-cpu.yaml` |
| `keda-vpa` | KEDA, cpu util        | `eval_real_keda.py --vpa`: working-set x1.15 rule   | `scaledobject-cpu.yaml` |

Both variants use the **same cpu-only ScaledObject**. The sim baselines' HPA
(`ClassicalController` in `eval_baselines.py`) scales replicas on CPU utilization
only -- never on memory -- so KEDA is held to the same axis for a faithful
comparison. The two variants differ only in the memory axis, exactly as
`hpa-only` and `hpa-vpa` do: `keda-vpa` splits it the recommended way (HPA on
CPU, VPA on memory), with the VPA half being the same rule as
`ClassicalController(vpa=True)`. The rule is actuated the way a production k8s VPA
that is not doing in-place resize does -- `kubectl set resources` on the
deployment + wait for the roll-out (`vpa_set_memory`), i.e. evict-and-recreate at
the new request -- rather than the in-place `resize` subresource, which silently
no-ops for memory on this k3s when the container resizePolicy needs a restart and
would score `keda-vpa` on that actuation gap instead of the policy. The real k8s
VPA control plane (recommender/updater/admission) is still **not** deployed,
matching how `hpa-vpa` works everywhere else in this repo. CPU trigger target is
`70` = sim `HPA_TARGET_UTIL = 0.70`.

Scope caveats (same honesty as every other real-cluster result): KEDA is
horizontal-only, and CPU-only here, so on the memory-bound window `keda` has no
lever for the memory axis at all (the honest reproduction of what `hpa-only`
does); `keda-vpa`'s vertical half is a rule, not the k8s VPA; single-rep runs on
short trace windows -- treat the numbers as directional.

## One-time: install KEDA

No Helm on node0, so use the release manifest (needs cluster-admin; egress to
github.com):

```
kubectl apply --server-side -f https://github.com/kedacore/keda/releases/download/v2.16.1/keda-2.16.1.yaml
kubectl -n keda rollout status deploy/keda-operator --timeout=180s
```

The operator lives in its own `keda` namespace. Everything this project actuates
stays in `hv-phase0` (guarded in `envs/real_backend.py`). `run_keda_real.sh`
refuses to start if the CRDs are missing.

Uninstall:
`kubectl delete -f https://github.com/kedacore/keda/releases/download/v2.16.1/keda-2.16.1.yaml`

## Run

```
source venv/bin/activate
cd phase1
./run_keda_real.sh          # detached; see the script header
```

Drives both variants over the same cpu-bound (`--start 1095`) and memory-bound
(`--start 7670`) busy windows in the current traces (the older `--start 3800`
cpu-bound window went idle after the mid-2025 trace regen), writes
`eval/real_{keda,keda-vpa}_{cpu-bound,memory-bound}.csv` (gitignored), re-runs the
RL headliners (`qrdqn`, `cvar-bellman-qrdqn`) matched-conditions on the same
windows tagged `_kedacmp`, then `compare_keda.py` folds everything into
`eval/keda_vs_rl_summary.csv` (committed).

Single smoke step:
```
kubectl apply -f keda/scaledobject-cpu.yaml
python eval_real_keda.py --workload cpu-bound --start 1095 --steps 5
kubectl delete -f keda/scaledobject-cpu.yaml
```
