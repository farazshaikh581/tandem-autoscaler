#!/usr/bin/env bash
# Tuned "spiral" run: tight HPA (hpa-tuned.yaml) + high-gain VPA emulator, at a
# load that parks per-pod CPU near the HPA target so the two controllers fight.
# Outputs are tagged "_spiral" so they never overwrite the baseline run.
#
#   ./run_spiral.sh <mode:disruptive|inplace> [duration_s] [rps]
#
# Applies ONLY into namespace hv-phase0.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-disruptive}"
DURATION="${2:-900}"
RPS="${3:-120}"
NS="hv-phase0"
PY="${PYTHON:-python3}"
# HPA manifest and output tag; defaults reproduce the tuned "spiral" run.
HPA_FILE="${HPA_FILE:-hpa-tuned.yaml}"
TAG="${TAG:-spiral}"

# High-gain emulator knobs (override via env if desired)
VPA_INTERVAL="${VPA_INTERVAL:-15}"
VPA_MIN_CHANGE="${VPA_MIN_CHANGE:-15}"
VPA_EMA="${VPA_EMA:-0.3}"
VPA_TARGET_UTIL="${VPA_TARGET_UTIL:-0.70}"
VPA_LIMIT_RATIO="${VPA_LIMIT_RATIO:-2.0}"   # CPU limit = ratio x request

if [[ "$MODE" != "disruptive" && "$MODE" != "inplace" ]]; then
  echo "usage: $0 <disruptive|inplace> [duration_s] [rps]"; exit 1
fi
mkdir -p results

echo "=== [1/5] apply app + hpa ($HPA_FILE) (namespace $NS only) ==="
kubectl apply -f namespace.yaml
kubectl apply -f web-service.yaml
kubectl apply -f "$HPA_FILE"
kubectl -n "$NS" rollout status deploy/web-service --timeout=120s

SVC_IP="$(kubectl -n "$NS" get svc web-service -o jsonpath='{.spec.clusterIP}')"
URL="http://${SVC_IP}:80/"
echo "service ClusterIP = $SVC_IP ; mode=$MODE rps=$RPS gain(interval=$VPA_INTERVAL min_change=$VPA_MIN_CHANGE ema=$VPA_EMA)"

echo "=== [2/5] start load @ ${RPS} rps ==="
"$PY" load.py --url "$URL" --rps "$RPS" --duration "$DURATION" > "results/load_${MODE}_${TAG}.txt" 2>&1 &
LOAD_PID=$!

echo "=== [3/5] start high-gain VPA emulator (mode=$MODE) ==="
"$PY" vpa_emulator.py --mode "$MODE" --duration "$DURATION" \
    --interval "$VPA_INTERVAL" --min-change "$VPA_MIN_CHANGE" --ema "$VPA_EMA" \
    --target-util "$VPA_TARGET_UTIL" --limit-ratio "$VPA_LIMIT_RATIO" --out "results/vpa_${MODE}_${TAG}.csv" &
VPA_PID=$!

echo "=== [4/5] monitor for ${DURATION}s ==="
"$PY" monitor.py --duration "$DURATION" --out "results/monitor_${MODE}_${TAG}.csv"

echo "=== [5/5] stop background jobs ==="
kill "$LOAD_PID" "$VPA_PID" 2>/dev/null || true
wait "$LOAD_PID" "$VPA_PID" 2>/dev/null || true
cat "results/load_${MODE}_${TAG}.txt"
echo "done -> results/monitor_${MODE}_${TAG}.csv , results/vpa_${MODE}_${TAG}.csv"
