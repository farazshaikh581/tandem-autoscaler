#!/usr/bin/env bash
# Phase 0 run with the real upstream VPA (vpa/install_vpa.sh) as the vertical
# controller instead of vpa_emulator.py. Same app, load, HPA and monitor as
# run_spiral.sh.
#
#   ./run_vpa_real.sh <InPlaceOrRecreate|Recreate> <hpa file> <tag> [duration_s] [rps]
#
# Each run starts the recommender from scratch (VPA object and checkpoints
# deleted, recommender restarted), so no usage history carries over between
# runs. Writes results/monitor_vpa-<mode>_<tag>.csv, a 15 s trace of the VPA
# recommendation (results/vparec_<mode>_<tag>.csv) and the updater's actions
# (results/vpaevents_<mode>_<tag>.csv: in-place resizes and evictions).
set -euo pipefail
cd "$(dirname "$0")"
MODE="$1"; HPA_FILE="$2"; TAG="$3"; DURATION="${4:-900}"; RPS="${5:-120}"
NS=hv-phase0
PY="${PYTHON:-python3}"
case "$MODE" in InPlaceOrRecreate|Recreate) ;; *) echo "bad mode $MODE"; exit 1;; esac
mkdir -p results

echo "=== [1/6] apply app + hpa ($HPA_FILE) ==="
kubectl apply -f namespace.yaml
kubectl apply -f web-service.yaml
kubectl apply -f "$HPA_FILE"
kubectl -n "$NS" rollout status deploy/web-service --timeout=120s

echo "=== [2/6] fresh VPA recommender state ==="
kubectl -n "$NS" delete vpa web-service --ignore-not-found
kubectl -n "$NS" delete verticalpodautoscalercheckpoints --all --ignore-not-found
kubectl -n kube-system rollout restart deploy/vpa-recommender
kubectl -n kube-system rollout status deploy/vpa-recommender --timeout=180s
sed "s/__MODE__/$MODE/" vpa/vpa-web-service.yaml | kubectl apply -f -
T0="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

SVC_IP="$(kubectl -n "$NS" get svc web-service -o jsonpath='{.spec.clusterIP}')"
echo "=== [3/6] start load @ ${RPS} rps (VPA mode=$MODE) ==="
"$PY" load.py --url "http://${SVC_IP}:80/" --rps "$RPS" --duration "$DURATION" \
  > "results/load_vpa-${MODE}_${TAG}.txt" 2>&1 &
LOAD_PID=$!

echo "=== [4/6] trace VPA recommendation every 15 s ==="
(
  echo "elapsed_s,target_m,lower_m,upper_m"
  start=$(date +%s)
  while [ $(( $(date +%s) - start )) -lt "$DURATION" ]; do
    rec=$(kubectl -n "$NS" get vpa web-service -o jsonpath='{.status.recommendation.containerRecommendations[0].target.cpu},{.status.recommendation.containerRecommendations[0].lowerBound.cpu},{.status.recommendation.containerRecommendations[0].upperBound.cpu}' 2>/dev/null || true)
    echo "$(( $(date +%s) - start )),${rec}"
    sleep 15
  done
) > "results/vparec_${MODE}_${TAG}.csv" &
REC_PID=$!

echo "=== [5/6] monitor for ${DURATION}s ==="
"$PY" monitor.py --duration "$DURATION" --out "results/monitor_vpa-${MODE}_${TAG}.csv"

echo "=== [6/6] stop, collect updater events, remove VPA ==="
kill "$LOAD_PID" "$REC_PID" 2>/dev/null || true
wait "$LOAD_PID" "$REC_PID" 2>/dev/null || true
cat "results/load_vpa-${MODE}_${TAG}.txt"
kubectl -n "$NS" get events -o json | "$PY" -c "
import json, sys
t0 = '$T0'
rows = {}
for e in json.load(sys.stdin)['items']:
    r = e.get('reason')
    if r not in ('InPlaceResizedByVPA', 'EvictedByVPA', 'InPlaceResizeSkipped'):
        continue
    ts = e.get('lastTimestamp') or e.get('eventTime') or ''
    if ts < t0:
        continue
    rows[r] = rows.get(r, 0) + int(e.get('count') or 1)
print('reason,count')
for r in ('InPlaceResizedByVPA', 'EvictedByVPA', 'InPlaceResizeSkipped'):
    print(f'{r},{rows.get(r, 0)}')
" > "results/vpaevents_${MODE}_${TAG}.csv"
cat "results/vpaevents_${MODE}_${TAG}.csv"
kubectl -n "$NS" delete vpa web-service --ignore-not-found
echo "done -> results/monitor_vpa-${MODE}_${TAG}.csv"
