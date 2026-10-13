#!/usr/bin/env bash
# Bitbrains repeat of the two-tenant coupling experiment (#14) on the spiritquest
# bare-metal cluster. Runs ON r0 (that cluster's control plane; node0 is not on its
# pod network). Same shape as the Azure run: cpu-burn driven by QR-DQN on cpu-bound
# and io-burn driven by Option B on io-bound, each alone, then both at once, 25
# steps each. Original Phase 1 `all` models, the Bitbrains busy windows (cpu-bound
# 3800, io-bound 4051), outputs tagged spiritquest_bb_{solo,coupled}.
#
# Keep the r-nodes otherwise idle while this runs: the pods share their cores.
#   setsid nohup ./run_coupling_bb.sh > logs/coupling_bb.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs eval
export OMP_NUM_THREADS=1
PY="${PY:-python3}"
STEPS=25
CPU=(--algo qrdqn --app cpu-burn --workload cpu-bound --start 3800)
IO=(--algo cvar-bellman-qrdqn --app io-burn --workload io-bound --start 4051)
COMMON=(--model-tag all --steps "$STEPS" --trace-family bitbrains)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
tenant() {   # <log name> <cluster-tag> <tenant args...>
  local name="$1" tag="$2"; shift 2
  "$PY" eval_real.py "$@" "${COMMON[@]}" --cluster-tag "$tag" > "logs/coupling_bb_${name}.log" 2>&1
  log "    $name rc=$? $(grep -o 'SLA [0-9.]*%.*ms' "logs/coupling_bb_${name}.log" | tail -n 1)"
}

log "=== solo: cpu-burn / QR-DQN ==="
tenant cpu_solo spiritquest_bb_solo "${CPU[@]}"
log "=== solo: io-burn / Option B ==="
tenant io_solo spiritquest_bb_solo "${IO[@]}"
log "=== coupled: both tenants at once ==="
tenant cpu_coupled spiritquest_bb_coupled "${CPU[@]}" &
tenant io_coupled spiritquest_bb_coupled "${IO[@]}" &
wait
"$PY" compare_coupling.py --family bitbrains
log "ALL COUPLING RUNS DONE"
touch eval/.coupling_bb_done
