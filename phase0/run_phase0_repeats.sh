#!/usr/bin/env bash
# Repeats for the Phase 0 matrix, plus the real VPA with minute-scale flags.
#
# Emulator cells (vpa_emulator.py at 70%, fixed per-pod deadband):
#   HPA setpoint 60% | 70%   x   HPA behavior tuned | stock   x   disruptive | in-place
# brought to three runs each. Existing first runs: *_tuned70, *_stock70, *_tuned60
# (run_phase0_7070.sh / run_phase0_6070.sh) and monitor_disruptive_default2w
# (stock 60%, disruptive). The stock 60% in-place run from 2026-10-05 used the
# emulator before the deadband fix (*_default2w_buggyemu), so that cell gets three
# new runs.
#
# Real VPA 1.8.0 with vpa/vpa_profile.sh fast (stock flags never act within 15
# minutes, see that script): tuned HPA at 60% and 70% x Recreate | InPlaceOrRecreate,
# three runs each. The stock profile is restored on exit.
#
# 29 runs of 900 s at 120 rps (17 emulator, 12 real VPA), about 8 h. Runs are interleaved by repetition so a
# partial batch is balanced across cells. Same fixed conditions as
# run_phase0_7070.sh: k8s-worker3/4 cordoned, IO app scaled to 0, both restored on
# exit. Outputs are tagged _r2/_r3 (and _r1 for cells without a first run).
# A run whose monitor CSV already exists is skipped, so a relaunch resumes.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"
NS=hv-phase0
IO_REPLICAS="$(kubectl -n $NS get deploy web-service-io -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
restore() {
  kubectl -n $NS delete vpa web-service --ignore-not-found >/dev/null 2>&1 || true
  kubectl -n $NS delete hpa web-service --ignore-not-found >/dev/null 2>&1 || true
  vpa/vpa_profile.sh stock >/dev/null 2>&1 || true
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
  [ -s "results/monitor_$3_$2.csv" ] && { echo "##### skip emulator $3 $2 (done)"; return; }
  reset_app
  echo "##### emulator $3 $2 start $(date -u +%T)"
  HPA_FILE="$1" TAG="$2" VPA_TARGET_UTIL=0.70 taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 \
    ./run_spiral.sh "$3" 900 120
  echo "##### emulator $3 $2 end $(date -u +%T)"
}
vpa() {   # <hpa file> <tag> <InPlaceOrRecreate|Recreate>
  [ -s "results/monitor_vpa-$3_$2.csv" ] && { echo "##### skip vpa $3 $2 (done)"; return; }
  reset_app
  echo "##### vpa $3 $2 start $(date -u +%T)"
  taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 ./run_vpa_real.sh "$3" "$1" "$2" 900 120
  echo "##### vpa $3 $2 end $(date -u +%T)"
}
emu_rep() {   # <rep>
  emu hpa-tuned70.yaml "tuned70_r$1" disruptive
  emu hpa-tuned70.yaml "tuned70_r$1" inplace
  emu hpa70.yaml       "stock70_r$1" disruptive
  emu hpa70.yaml       "stock70_r$1" inplace
  emu hpa-tuned.yaml   "tuned60_r$1" disruptive
  emu hpa-tuned.yaml   "tuned60_r$1" inplace
  emu hpa.yaml         "stock60_r$1" disruptive
  emu hpa.yaml         "stock60_r$1" inplace
}
vpa_rep() {   # <rep>
  vpa hpa-tuned70.yaml "fast_tuned70_r$1" Recreate
  vpa hpa-tuned70.yaml "fast_tuned70_r$1" InPlaceOrRecreate
  vpa hpa-tuned.yaml   "fast_tuned60_r$1" Recreate
  vpa hpa-tuned.yaml   "fast_tuned60_r$1" InPlaceOrRecreate
}

emu hpa.yaml stock60_r1 inplace
vpa/vpa_profile.sh fast
vpa_rep 1
emu_rep 2
vpa/vpa_profile.sh fast
vpa_rep 2
emu_rep 3
vpa/vpa_profile.sh fast
vpa_rep 3
vpa/vpa_profile.sh stock
touch .phase0_repeats_done
echo "##### ALL DONE $(date -u +%T)"
