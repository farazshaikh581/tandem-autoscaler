#!/usr/bin/env bash
# Same-day, same-topology rerun of the Phase 0 comparison after the in-place
# deadband fix in vpa_emulator.py (in-place mode used to compare the
# recommendation against the Deployment template, which resize never changes,
# so its 15m deadband never engaged and it re-patched pods almost every
# interval). k8s-worker3/4 cordoned throughout to match the 2026-07-02 runs.
# Runs: tuned HPA x {disruptive, inplace}, stock HPA x inplace (stock x
# disruptive is unaffected by the fix: run_default_matched.sh, tag default2w).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"
trap 'kubectl uncordon k8s-worker3 k8s-worker4 >/dev/null 2>&1' EXIT
kubectl cordon k8s-worker3 k8s-worker4
run() {   # <hpa file> <tag> <mode>
  kubectl -n hv-phase0 delete hpa web-service --ignore-not-found
  kubectl apply -f web-service.yaml
  kubectl -n hv-phase0 rollout restart deploy/web-service
  kubectl -n hv-phase0 scale deploy/web-service --replicas=2
  kubectl -n hv-phase0 rollout status deploy/web-service --timeout=180s
  sleep 30
  echo "##### $3 $2 start $(date -u +%T)"
  HPA_FILE="$1" TAG="$2" taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 ./run_spiral.sh "$3" 900 120
  echo "##### $3 $2 end $(date -u +%T)"
}
run hpa-tuned.yaml spiral2w disruptive
run hpa-tuned.yaml spiral2w inplace
run hpa.yaml default2w inplace
kubectl -n hv-phase0 delete hpa web-service --ignore-not-found
kubectl apply -f web-service.yaml
kubectl -n hv-phase0 scale deploy/web-service --replicas=2
touch .matched_2w_done
