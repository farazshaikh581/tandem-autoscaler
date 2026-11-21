#!/usr/bin/env bash
# Five repetitions of every real-cluster cell in the paper (sim-to-real panel and
# KEDA panel), on both trace families, with a fresh deployment before every run.
#
# The earlier real-cluster numbers were one run per cell. A run that follows an
# OOM window inherits crash-looping pods, and one 20-interval run moves 5 points
# per interval, so single runs cannot separate the controllers. Here:
#   - RL repetitions use model seeds 0-4 (TANDEM and QR-DQN),
#   - KEDA / KEDA+VPA repeat the same controller five times,
#   - every run starts from a rollout-restarted deployment,
#   - repetitions are the outer loop, so partial results stay balanced.
# Windows, step counts, rps and topology are those of the original runs.
# Output: eval/real_reps/<cell>__<controller>__<family>__rep<k>.csv (idempotent).
#
#   setsid nohup ./run_real_reps.sh > logs/real_reps.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs eval/real_reps logs/real_reps
export OMP_NUM_THREADS=1
PY="${PYTHON:-python3}"
DRV=(taskset -c 5)
REPS=${REPS:-5}
EXTRA_WORKERS=(k8s-worker3 k8s-worker4)
NS=hv-phase0
log() { echo "[$(date -u +%F' '%T)] $*"; }

declare -A WIN=( [azure:cpu]=1097 [azure:kedacpu]=1095 [azure:io]=957 [azure:mixed]=9 [azure:mem]=7670
                 [bitbrains:cpu]=3800 [bitbrains:kedacpu]=3800 [bitbrains:io]=4051 [bitbrains:mixed]=76
                 [bitbrains:mem]=7670 )

fam_args() { [ "$1" = bitbrains ] && echo "--trace-family bitbrains --cluster-tag bb" || echo ""; }
model_tag() {   # <family> <seed>
  local f=$1 s=$2
  if [ "$f" = azure ]; then [ "$s" = 0 ] && echo all_azure || echo "all_azure_seed$s"
  else case $s in 0) echo all;; 1|2) echo "all_seed$s";; *) echo "all_bb_seed$s";; esac; fi
}
restart_apps() {
  kubectl -n $NS rollout restart deploy/web-service deploy/web-service-io >/dev/null
  kubectl -n $NS rollout status deploy/web-service --timeout=240s >/dev/null
  kubectl -n $NS rollout status deploy/web-service-io --timeout=240s >/dev/null
  sleep 10
}
rl() {   # <cell> <family> <rep> <algo> <workload> <window key> <steps> [--app io-burn]
  local cell=$1 fam=$2 rep=$3 algo=$4 wl=$5 key=$6 steps=$7; shift 7
  local out="eval/real_reps/${cell}__${algo}__${fam}__rep${rep}.csv"
  [ -f "$out" ] && return 0
  restart_apps
  "${DRV[@]}" $PY -u eval_real.py --algo "$algo" --model-tag "$(model_tag $fam $rep)" --seed "$rep" \
      --workload "$wl" --start "${WIN[$fam:$key]}" --steps "$steps" $(fam_args $fam) --out "$out.tmp" "$@" \
      > "logs/real_reps/${cell}__${algo}__${fam}__rep${rep}.log" 2>&1 && mv "$out.tmp" "$out"
  log "$cell $algo $fam rep$rep rc=$? $(grep -o 'SLA [0-9.]*%' "logs/real_reps/${cell}__${algo}__${fam}__rep${rep}.log" | tail -1)"
}
keda() {   # <family> <rep> <keda|keda-vpa> <workload> <window key> <cell>
  local fam=$1 rep=$2 name=$3 wl=$4 key=$5 cell=$6
  local out="eval/real_reps/${cell}__${name}__${fam}__rep${rep}.csv"
  [ -f "$out" ] && return 0
  restart_apps
  kubectl apply -f keda/scaledobject-cpu.yaml >/dev/null
  for _ in $(seq 1 30); do
    kubectl get scaledobject web-service-keda -n $NS \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | grep -q True && break
    sleep 2
  done
  local vpa=(); [ "$name" = keda-vpa ] && vpa=(--vpa)
  "${DRV[@]}" $PY -u eval_real_keda.py --workload "$wl" --start "${WIN[$fam:$key]}" --steps 20 \
      --seed "$rep" "${vpa[@]}" $(fam_args $fam) --out "$out.tmp" \
      > "logs/real_reps/${cell}__${name}__${fam}__rep${rep}.log" 2>&1 && mv "$out.tmp" "$out"
  log "$cell $name $fam rep$rep rc=$? $(grep -o 'SLA [0-9.]*%' "logs/real_reps/${cell}__${name}__${fam}__rep${rep}.log" | tail -1)"
  kubectl delete -f keda/scaledobject-cpu.yaml --ignore-not-found >/dev/null
  sleep 5
}

kubectl get crd scaledobjects.keda.sh >/dev/null || { log "FATAL: KEDA CRDs missing"; exit 1; }
kubectl delete scaledobject web-service-keda -n $NS --ignore-not-found >/dev/null
trap 'kubectl uncordon "${EXTRA_WORKERS[@]}" >/dev/null 2>&1; kubectl delete scaledobject web-service-keda -n '$NS' --ignore-not-found >/dev/null 2>&1' EXIT

log "=== phase 1: two workers (IO app, mixed, CPU 2-node) ==="
kubectl cordon "${EXTRA_WORKERS[@]}" >/dev/null
for rep in $(seq 0 $((REPS-1))); do
  for fam in azure bitbrains; do
    for algo in cvar-bellman-qrdqn qrdqn; do
      rl io-bound_io-burn  $fam $rep $algo io-bound  io    20 --app io-burn
      rl mixed_io-burn     $fam $rep $algo mixed     mixed 15 --app io-burn
      rl cpu-bound_2node   $fam $rep $algo cpu-bound cpu   20
    done
  done
done

log "=== phase 2: four workers (CPU 4-node, KEDA panel) ==="
kubectl uncordon "${EXTRA_WORKERS[@]}" >/dev/null
for rep in $(seq 0 $((REPS-1))); do
  for fam in azure bitbrains; do
    for algo in cvar-bellman-qrdqn qrdqn; do
      rl cpu-bound_4node   $fam $rep $algo cpu-bound    cpu     20
      rl kedacmp_cpu-bound $fam $rep $algo cpu-bound    kedacpu 20
      rl kedacmp_memory-bound $fam $rep $algo memory-bound mem  20
    done
    for name in keda keda-vpa; do
      keda $fam $rep $name cpu-bound    kedacpu kedacmp_cpu-bound
      keda $fam $rep $name memory-bound mem     kedacmp_memory-bound
    done
  done
done
log "ALL REAL REPS DONE"
touch eval/.real_reps_done
