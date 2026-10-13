#!/usr/bin/env bash
# Real-VPA Phase 0 matrix with the testbed-specific settings as parameters, so the
# same cells can be run on a second testbed (e.g. a host whose worker vCPUs are
# pinned, or whose CPU needs a different request rate for the same per-pod load).
#
#   PROFILE=fast REPS=3 RPS=120 DRIVER_CPU=5 ./run_vpa_matrix.sh
#
#   PROFILE      vpa/vpa_profile.sh profile: fast | stock          (default fast)
#   CELLS        space-separated <hpa file>:<tag>:<VPA mode>       (default: tuned 70/60
#                x Recreate/InPlaceOrRecreate, the run_phase0_repeats.sh cells)
#   REPS         repetitions per cell, tagged _r1.._rN             (default 3)
#   RPS          constant request rate                             (default 120)
#   DURATION     seconds per run                                   (default 900)
#   DRIVER_CPU   core for the load generator and monitor           (default 5)
#   TESTBED      suffix appended to every tag, e.g. "saul"         (default none)
#
# Fixed conditions as in run_phase0_7070.sh: k8s-worker3/4 cordoned, IO app scaled to
# 0, both restored on exit, stock VPA profile restored on exit. A run whose monitor CSV
# already exists is skipped, so a relaunch resumes.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON="${PYTHON:-python3}"
PROFILE="${PROFILE:-fast}"; REPS="${REPS:-3}"; RPS="${RPS:-120}"; DURATION="${DURATION:-900}"
DRIVER_CPU="${DRIVER_CPU:-5}"; SUFFIX="${TESTBED:+_$TESTBED}"
CELLS="${CELLS:-hpa-tuned70.yaml:tuned70:Recreate hpa-tuned70.yaml:tuned70:InPlaceOrRecreate hpa-tuned.yaml:tuned60:Recreate hpa-tuned.yaml:tuned60:InPlaceOrRecreate}"
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
[ "$IO_REPLICAS" -gt 0 ] && kubectl -n $NS scale deploy/web-service-io --replicas=0
vpa/vpa_profile.sh "$PROFILE"

reset_app() {
  kubectl -n $NS delete vpa web-service --ignore-not-found
  kubectl -n $NS delete hpa web-service --ignore-not-found
  kubectl apply -f web-service.yaml
  kubectl -n $NS rollout restart deploy/web-service
  kubectl -n $NS scale deploy/web-service --replicas=2
  kubectl -n $NS rollout status deploy/web-service --timeout=180s
  sleep 30
}
echo "##### profile=$PROFILE reps=$REPS rps=$RPS duration=$DURATION driver_cpu=$DRIVER_CPU suffix=$SUFFIX"
for r in $(seq 1 "$REPS"); do
  for cell in $CELLS; do
    IFS=: read -r hpa tag mode <<< "$cell"
    full="${PROFILE}_${tag}${SUFFIX}_r$r"
    [ "$PROFILE" = stock ] && full="${tag}${SUFFIX}_r$r"
    if [ -s "results/monitor_vpa-${mode}_${full}.csv" ]; then echo "##### skip vpa $mode $full (done)"; continue; fi
    reset_app
    echo "##### vpa $mode $full start $(date -u +%T)"
    taskset -c "$DRIVER_CPU" nice -n 19 env OMP_NUM_THREADS=1 ./run_vpa_real.sh "$mode" "$hpa" "$full" "$DURATION" "$RPS"
    echo "##### vpa $mode $full end $(date -u +%T)"
  done
done
touch ".vpa_matrix_${PROFILE}${SUFFIX}_done"
echo "##### ALL DONE $(date -u +%T)"
