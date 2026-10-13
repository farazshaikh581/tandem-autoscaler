#!/usr/bin/env bash
# Does CFS throttling explain the emulator's p99 (54-137 ms vs about 10 ms with the real VPA)?
#
# The emulator sets the CPU limit to 2.0x its request (about 110m -> 220m); the real VPA keeps
# the template's 2.5x on larger requests. This repeats the tuned 70% emulator cells with the
# limit ratio as the only change (2.0 as before, 2.5 as the real VPA, 10 = effectively no
# limit), plus the minute-scale real VPA (InPlaceOrRecreate) as the reference, and records
# per-pod CFS throttling from cAdvisor (throttle_sampler.py) in every run.
#
# 18 runs of 900 s at 120 rps (6 cells x 3), about 5 h, interleaved by repetition. Same fixed
# conditions as run_phase0_repeats.sh (k8s-worker3/4 cordoned, IO app scaled to 0, restored on
# exit). A run whose monitor CSV already exists is skipped, so a relaunch resumes.
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
  kubectl -n $NS rollout restart deploy/web-service >/dev/null 2>&1 || true
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
emu() {   # <disruptive|inplace> <limit ratio> <rep>
  local tag="tuned70_lr$2_r$3"
  [ -s "results/monitor_$1_${tag}.csv" ] && { echo "##### skip emulator $1 $tag (done)"; return; }
  reset_app
  echo "##### emulator $1 $tag start $(date -u +%T)"
  taskset -c 5 nice -n 19 "$PYTHON" throttle_sampler.py --duration 900 --out "results/throttle_$1_${tag}.csv" &
  local sp=$!
  HPA_FILE=hpa-tuned70.yaml TAG="$tag" VPA_TARGET_UTIL=0.70 VPA_LIMIT_RATIO="$2" \
    taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 ./run_spiral.sh "$1" 900 120
  wait "$sp" || true
  echo "##### emulator $1 $tag end $(date -u +%T)"
}
vpa() {   # <rep>
  local tag="fast_tuned70_thr_r$1"
  [ -s "results/monitor_vpa-InPlaceOrRecreate_${tag}.csv" ] && { echo "##### skip vpa $tag (done)"; return; }
  reset_app
  echo "##### vpa InPlaceOrRecreate $tag start $(date -u +%T)"
  taskset -c 5 nice -n 19 "$PYTHON" throttle_sampler.py --duration 900 --out "results/throttle_vpa-InPlaceOrRecreate_${tag}.csv" &
  local sp=$!
  taskset -c 5 nice -n 19 env OMP_NUM_THREADS=1 ./run_vpa_real.sh InPlaceOrRecreate hpa-tuned70.yaml "$tag" 900 120
  wait "$sp" || true
  echo "##### vpa InPlaceOrRecreate $tag end $(date -u +%T)"
}

for r in 1 2 3; do
  emu inplace 2.0 "$r"
  emu inplace 2.5 "$r"
  emu inplace 10 "$r"
  emu disruptive 2.0 "$r"
  emu disruptive 10 "$r"
  vpa/vpa_profile.sh fast
  vpa "$r"
  vpa/vpa_profile.sh stock
done
touch .throttle_check_done
echo "##### ALL DONE $(date -u +%T)"
