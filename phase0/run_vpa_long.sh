#!/usr/bin/env bash
# Long run of the upstream VPA with its STOCK flags next to the HPA, to measure when the
# updater first acts on running pods (12 h minimum pod lifetime before an in-range update)
# and what the HPA does afterwards. Same app, load and topology as the 15-minute runs.
#
#   MODE=InPlaceOrRecreate HPA_FILE=hpa70.yaml TAG=long_stock70 HOURS=26 RPS=120 ./run_vpa_long.sh
#
# Differences from run_vpa_real.sh, needed at this length:
#   - load.py runs in 1-hour chunks (it keeps every sample in memory), which also gives an
#     hourly latency series: results/load_hourly_vpa-<mode>_<tag>.txt
#   - VPA events are streamed as they happen (the API server keeps events for about an hour):
#     results/vpaevents_stream_<mode>_<tag>.txt
#   - the recommendation is traced every 60 s and the monitor samples every 10 s.
# Conditions: stock VPA profile, k8s-worker3/4 cordoned, IO app scaled to 0; all restored on exit.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
MODE="${MODE:-InPlaceOrRecreate}"; HPA_FILE="${HPA_FILE:-hpa70.yaml}"; TAG="${TAG:-long_stock70}"
HOURS="${HOURS:-26}"; RPS="${RPS:-120}"; DRIVER_CPU="${DRIVER_CPU:-5}"
NS=hv-phase0
DURATION=$(( HOURS * 3600 ))
OUT="vpa-${MODE}_${TAG}"
mkdir -p results
IO_REPLICAS="$(kubectl -n $NS get deploy web-service-io -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)"
BG=()
restore() {
  for p in "${BG[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  kubectl -n $NS delete vpa web-service --ignore-not-found >/dev/null 2>&1 || true
  kubectl -n $NS delete hpa web-service --ignore-not-found >/dev/null 2>&1 || true
  kubectl apply -f web-service.yaml >/dev/null 2>&1 || true
  kubectl -n $NS scale deploy/web-service --replicas=2 >/dev/null 2>&1 || true
  [ "$IO_REPLICAS" -gt 0 ] && kubectl -n $NS scale deploy/web-service-io --replicas="$IO_REPLICAS" >/dev/null 2>&1 || true
  kubectl uncordon k8s-worker3 k8s-worker4 >/dev/null 2>&1 || true
}
trap restore EXIT
kubectl cordon k8s-worker3 k8s-worker4
[ "$IO_REPLICAS" -gt 0 ] && kubectl -n $NS scale deploy/web-service-io --replicas=0
vpa/vpa_profile.sh stock

echo "##### long run $OUT hours=$HOURS rps=$RPS start $(date -u +%FT%T)"
kubectl -n $NS delete vpa web-service --ignore-not-found
kubectl -n $NS delete hpa web-service --ignore-not-found
kubectl apply -f web-service.yaml
kubectl -n $NS rollout restart deploy/web-service
kubectl -n $NS scale deploy/web-service --replicas=2
kubectl -n $NS rollout status deploy/web-service --timeout=180s
sleep 30
kubectl apply -f "$HPA_FILE"
kubectl -n $NS delete verticalpodautoscalercheckpoints --all --ignore-not-found
kubectl -n kube-system rollout restart deploy/vpa-recommender
kubectl -n kube-system rollout status deploy/vpa-recommender --timeout=180s
sed "s/__MODE__/$MODE/" vpa/vpa-web-service.yaml | kubectl apply -f -
SVC_IP="$(kubectl -n $NS get svc web-service -o jsonpath='{.spec.clusterIP}')"

# The API server closes a watch after 30-60 min, so the watch is restarted for the whole run
# (bounded by timeout so that no watch outlives the run by more than 30 min).
(
  start=$(date +%s)
  while [ $(( $(date +%s) - start )) -lt "$DURATION" ]; do
    timeout 1800 kubectl -n $NS get events --watch-only --no-headers \
      -o custom-columns=TIME:.lastTimestamp,REASON:.reason,KIND:.involvedObject.kind,OBJ:.involvedObject.name,COUNT:.count,MSG:.message || true
    sleep 2
  done
) > "results/vpaevents_stream_${MODE}_${TAG}.txt" 2>&1 &
BG+=($!)
(
  echo "elapsed_s,target_m,lower_m,upper_m"
  start=$(date +%s)
  while [ $(( $(date +%s) - start )) -lt "$DURATION" ]; do
    rec=$(kubectl -n $NS get vpa web-service -o jsonpath='{.status.recommendation.containerRecommendations[0].target.cpu},{.status.recommendation.containerRecommendations[0].lowerBound.cpu},{.status.recommendation.containerRecommendations[0].upperBound.cpu}' 2>/dev/null || true)
    echo "$(( $(date +%s) - start )),${rec}"
    sleep 60
  done
) > "results/vparec_${MODE}_${TAG}.csv" &
BG+=($!)
(
  for h in $(seq 1 "$HOURS"); do
    printf "hour %d " "$h"
    taskset -c "$DRIVER_CPU" nice -n 19 "$PY" load.py --url "http://${SVC_IP}:80/" --rps "$RPS" --duration 3600 2>&1 | grep "done:" || echo "load chunk failed"
  done
) > "results/load_hourly_${OUT}.txt" &
BG+=($!)

taskset -c "$DRIVER_CPU" nice -n 19 "$PY" monitor.py --interval 10 --duration "$DURATION" --out "results/monitor_${OUT}.csv"
echo "##### long run $OUT end $(date -u +%FT%T)"
