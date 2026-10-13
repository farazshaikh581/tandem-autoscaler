#!/usr/bin/env bash
# Phase 0: provoke and measure the HPA/VPA death spiral in an isolated namespace.
# One run = one VPA actuation mode. Run it twice (disruptive, inplace) to compare.
#
#   ./run_phase0.sh <mode:disruptive|inplace> [duration_s] [rps]
#
# Applies ONLY into namespace hv-phase0. Nothing else on the cluster is touched.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-inplace}"
DURATION="${2:-900}"
RPS="${3:-40}"
NS="hv-phase0"

if [[ "$MODE" != "disruptive" && "$MODE" != "inplace" ]]; then
  echo "usage: $0 <disruptive|inplace> [duration_s] [rps]"; exit 1
fi

PY="${PYTHON:-python3}"
mkdir -p results

echo "=== [1/5] apply manifests (namespace $NS only) ==="
kubectl apply -f namespace.yaml
kubectl apply -f web-service.yaml
kubectl apply -f hpa.yaml
kubectl -n "$NS" rollout status deploy/web-service --timeout=120s

SVC_IP="$(kubectl -n "$NS" get svc web-service -o jsonpath='{.spec.clusterIP}')"
URL="http://${SVC_IP}:80/"
echo "service ClusterIP = $SVC_IP"

echo "=== [2/5] start constant load @ ${RPS} rps ==="
"$PY" load.py --url "$URL" --rps "$RPS" --duration "$DURATION" &
LOAD_PID=$!

echo "=== [3/5] start VPA emulator (mode=$MODE) ==="
"$PY" vpa_emulator.py --mode "$MODE" --duration "$DURATION" \
    --out "results/vpa_${MODE}.csv" &
VPA_PID=$!

echo "=== [4/5] monitor for ${DURATION}s ==="
"$PY" monitor.py --duration "$DURATION" --out "results/monitor_${MODE}.csv"

echo "=== [5/5] stop background jobs ==="
kill "$LOAD_PID" "$VPA_PID" 2>/dev/null || true
wait "$LOAD_PID" "$VPA_PID" 2>/dev/null || true

echo "done. CSVs in results/ (monitor_${MODE}.csv, vpa_${MODE}.csv)"
echo "when both modes are captured:"
echo "  $PY plot_spiral.py --run results/monitor_disruptive.csv:disruptive \\"
echo "                     --run results/monitor_inplace.csv:in-place"
echo "teardown when finished:  ./teardown.sh"
