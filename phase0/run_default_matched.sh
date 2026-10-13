#!/usr/bin/env bash
# Matched control for the Phase 0 spiral runs: identical 120 rps load and VPA
# emulator gain, but the stock HPA (hpa.yaml: default 300 s scale-down
# stabilization) instead of the tuned one. Runs disruptive then in-place.
# Does not call teardown.sh (it would delete the IO app in hv-phase0); between
# runs it removes only the HPA and resets web-service to the manifest.
#
# TWO_WORKERS=1 cordons k8s-worker3/4 for the whole run (outputs tagged
# default2w), matching the 2-worker topology of the 2026-07-02 tuned runs; the
# extra workers were added after those runs.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"
export HPA_FILE=hpa.yaml TAG=default
if [ "${TWO_WORKERS:-0}" = 1 ]; then
  export TAG=default2w
  trap 'kubectl uncordon k8s-worker3 k8s-worker4 >/dev/null 2>&1' EXIT
  kubectl cordon k8s-worker3 k8s-worker4
  # evict hv-phase0 web-service pods off the extra workers before the first run
  kubectl -n hv-phase0 rollout restart deploy/web-service
  kubectl -n hv-phase0 rollout status deploy/web-service --timeout=180s
fi
for MODE in disruptive inplace; do
  kubectl -n hv-phase0 delete hpa web-service --ignore-not-found
  kubectl apply -f web-service.yaml
  kubectl -n hv-phase0 scale deploy/web-service --replicas=2
  kubectl -n hv-phase0 rollout status deploy/web-service --timeout=180s
  sleep 30
  echo "##### $MODE start $(date -u +%T)"
  taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 ./run_spiral.sh "$MODE" 900 120
  echo "##### $MODE end $(date -u +%T)"
done
kubectl -n hv-phase0 delete hpa web-service --ignore-not-found
kubectl apply -f web-service.yaml
kubectl -n hv-phase0 scale deploy/web-service --replicas=2
touch ".${TAG}_matched_done"
