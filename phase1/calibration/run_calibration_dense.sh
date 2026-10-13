#!/usr/bin/env bash
# Two extra calibration sweeps for the service model (paper Sec. 4.2).
#   A: the original fixed configuration (2 pods, request = limit = 200m) at dense offered loads
#      across the knee (demand-based rho 0.78-0.99), to pin down where latency leaves its base level.
#   B: the DEPLOYED configuration (2 pods, request 200m, limit 500m, web-service.yaml), offered load
#      raised until it saturates, to see whether the knee sits at the request or at the limit.
# Three repetitions interleaved A,B,A,B,A,B; 30 s warm-up + 90 s measurement per level, as in the
# original sweep. Same fixed conditions as the Phase 0 runs (k8s-worker3/4 cordoned, IO app at 0
# replicas, restored on exit). A repetition whose CSV exists is skipped, so a relaunch resumes.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"
NS=hv-phase0
YAML=../../phase0/web-service.yaml
IO_REPLICAS="$(kubectl -n $NS get deploy web-service-io -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
restore() {
  kubectl apply -f $YAML >/dev/null 2>&1 || true
  kubectl -n $NS rollout restart deploy/web-service >/dev/null 2>&1 || true
  kubectl -n $NS scale deploy/web-service --replicas=2 >/dev/null 2>&1 || true
  [ "$IO_REPLICAS" -gt 0 ] && kubectl -n $NS scale deploy/web-service-io --replicas="$IO_REPLICAS" >/dev/null 2>&1 || true
  kubectl uncordon k8s-worker3 k8s-worker4 >/dev/null 2>&1 || true
}
trap restore EXIT
kubectl cordon k8s-worker3 k8s-worker4
kubectl -n $NS scale deploy/web-service-io --replicas=0

LEVELS_A=95,100,103,106,109,112,115,118,122
LEVELS_B=40,80,120,160,200,230,250,265,280,295,310,330,360

settle() {
  kubectl -n $NS rollout status deploy/web-service --timeout=180s
  kubectl -n $NS get pods -l app=web-service -o wide --no-headers | awk '{print "  pod",$1,"on",$7}'
  sleep 30
}
for r in 1 2 3; do
  if [ ! -s sweep_dense_A_r$r.csv ]; then
    echo "[$(date -u +%T)] A r$r: request = limit = 200m"
    kubectl apply -f $YAML
    kubectl -n $NS set resources deploy/web-service --limits=cpu=200m --requests=cpu=200m
    kubectl -n $NS rollout restart deploy/web-service
    kubectl -n $NS scale deploy/web-service --replicas=2
    settle
    taskset -c 4,5 nice -n 10 $PYTHON calibrate_sweep.py --levels $LEVELS_A --warmup 30 --measure 90 --out sweep_dense_A_r$r.csv
  fi
  if [ ! -s sweep_deployed_B_r$r.csv ]; then
    echo "[$(date -u +%T)] B r$r: request 200m, limit 500m"
    kubectl apply -f $YAML
    kubectl -n $NS rollout restart deploy/web-service
    kubectl -n $NS scale deploy/web-service --replicas=2
    settle
    taskset -c 4,5 nice -n 10 $PYTHON calibrate_sweep.py --levels $LEVELS_B --warmup 30 --measure 90 --procs 2 --out sweep_deployed_B_r$r.csv
  fi
done
touch .calibration_dense_done
echo "[$(date -u +%T)] done"
