#!/usr/bin/env bash
# Phase 0 matrix with matched setpoints: the HPA and the vertical controller both
# target 70% CPU utilization (hpa70.yaml / hpa-tuned70.yaml), so any oscillation
# cannot come from the two loops aiming at different values of the same ratio.
#
#   vertical controller   x   HPA behavior        x   actuation
#   vpa_emulator.py           tuned (15 s / 0 s)      disruptive | in-place
#   upstream VPA 1.8.0        stock (300 s)           Recreate   | InPlaceOrRecreate
#
# 8 runs of 900 s at 120 rps, one after another, same day. Conditions held fixed:
# k8s-worker3/4 cordoned (the 2-worker topology of every earlier Phase 0 run) and
# the IO app scaled to 0 so web-service is the only workload in hv-phase0. Both are
# restored on exit. Needs vpa/install_vpa.sh to have been run.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"
NS=hv-phase0
IO_REPLICAS="$(kubectl -n $NS get deploy web-service-io -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
restore() {
  kubectl -n $NS delete vpa web-service --ignore-not-found >/dev/null 2>&1 || true
  kubectl -n $NS delete hpa web-service --ignore-not-found >/dev/null 2>&1 || true
  kubectl apply -f web-service.yaml >/dev/null 2>&1 || true
  kubectl -n $NS scale deploy/web-service --replicas=2 >/dev/null 2>&1 || true
  [ "$IO_REPLICAS" -gt 0 ] && kubectl -n $NS scale deploy/web-service-io --replicas="$IO_REPLICAS" >/dev/null 2>&1 || true
  kubectl uncordon k8s-worker3 k8s-worker4 >/dev/null 2>&1 || true
}
trap restore EXIT
kubectl cordon k8s-worker3 k8s-worker4
kubectl -n $NS scale deploy/web-service-io --replicas=0

reset_app() {
  kubectl -n $NS delete vpa web-service --ignore-not-found
  kubectl -n $NS delete hpa web-service --ignore-not-found
  kubectl apply -f web-service.yaml
  kubectl -n $NS rollout restart deploy/web-service
  kubectl -n $NS scale deploy/web-service --replicas=2
  kubectl -n $NS rollout status deploy/web-service --timeout=180s
  sleep 30
}
emu() {   # <hpa file> <tag> <disruptive|inplace>
  reset_app
  echo "##### emulator $3 $2 start $(date -u +%T)"
  HPA_FILE="$1" TAG="$2" VPA_TARGET_UTIL=0.70 taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 \
    ./run_spiral.sh "$3" 900 120
  echo "##### emulator $3 $2 end $(date -u +%T)"
}
vpa() {   # <hpa file> <tag> <InPlaceOrRecreate|Recreate>
  reset_app
  echo "##### vpa $3 $2 start $(date -u +%T)"
  taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 ./run_vpa_real.sh "$3" "$1" "$2" 900 120
  echo "##### vpa $3 $2 end $(date -u +%T)"
}
emu hpa-tuned70.yaml tuned70 disruptive
emu hpa-tuned70.yaml tuned70 inplace
emu hpa70.yaml stock70 disruptive
emu hpa70.yaml stock70 inplace
vpa hpa-tuned70.yaml tuned70 Recreate
vpa hpa-tuned70.yaml tuned70 InPlaceOrRecreate
vpa hpa70.yaml stock70 Recreate
vpa hpa70.yaml stock70 InPlaceOrRecreate
touch .phase0_7070_done
echo "##### ALL DONE $(date -u +%T)"
