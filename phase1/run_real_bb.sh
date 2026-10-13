#!/usr/bin/env bash
# Bitbrains repeats of the node0 real-cluster experiments: the IO app (#12), the
# 2-worker vs 4-worker scale-out (#13) and the KEDA baseline (#19). Same drivers,
# step counts and max rps as the Azure runs; the original Phase 1 `all` models
# (trained on Bitbrains); every output tagged `bb` so nothing Azure is touched.
#
# Windows are the busy Bitbrains ones the original Phase 1 real runs used:
# cpu-bound 3800 (near load 1.0 throughout; this trace peaks at 1.0, so there is
# no overload ramp to match Azure's 1097), io-bound 4051, mixed 76 (15 steps,
# like Azure's mixed run). The coupling repeat runs from r0, see README.
#
# Topology follows the Azure runs: #12 ran before k8s-worker3/4 existed and #13's
# baseline cordoned them, so the IO-app and 2-node runs happen with both cordoned;
# the 4-node and KEDA runs with all four workers schedulable.
#
# Sequential by design: every phase drives the same deployments.
#   source venv/bin/activate
#   setsid nohup ./run_real_bb.sh > logs/real_bb.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs eval
export OMP_NUM_THREADS=1
FAM=(--trace-family bitbrains)
STEPS=20
CPU_WIN=3800; IO_WIN=4051; MIX_WIN=76; MIX_STEPS=15
EXTRA_WORKERS=(k8s-worker3 k8s-worker4)
ALGOS=(qrdqn cvar-bellman-qrdqn)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
real() {   # <log name> <eval_real.py args...>
  local name="$1"; shift
  log "--- $name"
  python eval_real.py --model-tag all "${FAM[@]}" "$@" > "logs/real_bb_${name}.log" 2>&1
  log "    rc=$? $(tail -n 3 "logs/real_bb_${name}.log" | grep -o 'SLA [0-9.]*%.*ms' || true)"
}
restart_apps() {
  kubectl -n hv-phase0 rollout restart deploy/web-service deploy/web-service-io
  kubectl -n hv-phase0 rollout status deploy/web-service --timeout=180s
  kubectl -n hv-phase0 rollout status deploy/web-service-io --timeout=180s
  kubectl -n hv-phase0 get pods -o wide --no-headers | awk '{print "    " $1, $7}'
}

# never leave the extra workers cordoned, whatever happens below
trap 'kubectl uncordon "${EXTRA_WORKERS[@]}" >/dev/null 2>&1' EXIT

log "=== 2-worker phase: cordon ${EXTRA_WORKERS[*]} ==="
kubectl cordon "${EXTRA_WORKERS[@]}"
restart_apps
for algo in "${ALGOS[@]}"; do
  real "${algo}_io-bound_io-burn" --algo "$algo" --app io-burn --workload io-bound \
    --start "$IO_WIN" --steps "$STEPS" --cluster-tag bb
  real "${algo}_mixed_io-burn" --algo "$algo" --app io-burn --workload mixed \
    --start "$MIX_WIN" --steps "$MIX_STEPS" --cluster-tag bb
  real "${algo}_cpu-bound_2node" --algo "$algo" --workload cpu-bound \
    --start "$CPU_WIN" --steps "$STEPS" --cluster-tag 2node_bb
done

log "=== 4-worker phase: uncordon ${EXTRA_WORKERS[*]} ==="
kubectl uncordon "${EXTRA_WORKERS[@]}"
for algo in "${ALGOS[@]}"; do
  real "${algo}_cpu-bound_4node" --algo "$algo" --workload cpu-bound \
    --start "$CPU_WIN" --steps "$STEPS" --cluster-tag 4node_bb
done

log "=== KEDA (4 workers) ==="
KEDA_FAMILY=bitbrains ./run_keda_real.sh > logs/keda_real_bb.log 2>&1
log "    rc=$?"

log "=== sim side of every sim-vs-real pair ==="
pair() {   # <algo> <workload> <start> <steps> <cluster-tag> [--app io-burn]
  local algo="$1" wl="$2" start="$3" steps="$4" tag="$5"; shift 5
  python compare_sim_real.py --algo "$algo" --workload "$wl" --start "$start" --steps "$steps" \
    --cluster-tag "$tag" --model-tag all "${FAM[@]}" "$@" 2>&1 | tail -n 1
  python plot_sim_real.py --algo "$algo" --workload "$wl" --cluster-tag "$tag" --model-tag all \
    "$@" > /dev/null 2>&1
}
for algo in "${ALGOS[@]}"; do
  pair "$algo" io-bound "$IO_WIN" "$STEPS" bb --app io-burn
  pair "$algo" mixed "$MIX_WIN" "$MIX_STEPS" bb --app io-burn
  pair "$algo" cpu-bound "$CPU_WIN" "$STEPS" 2node_bb
  pair "$algo" cpu-bound "$CPU_WIN" "$STEPS" 4node_bb
done
log "ALL BITBRAINS REAL RUNS DONE"
touch eval/.real_bb_done
