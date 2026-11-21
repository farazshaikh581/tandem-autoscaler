#!/usr/bin/env bash
# Memory-bound KEDA-panel cell for the learned controllers with the VPA's actuation:
# a memory change is rolled out on the Deployment (evict and recreate) instead of
# resized in place. KEDA+VPA already actuates this way (eval_real_keda.py), so the
# panel then compares memory policies, not actuation paths. Same windows, seeds,
# rps, topology (4 workers) and fresh deployment per run as run_real_reps.sh.
# Output: eval/real_reps/kedacmp_memory-bound-recreate__<algo>__<family>__rep<k>.csv
#   setsid nohup ./run_real_mem_recreate.sh > logs/real_mem_recreate.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p eval/real_reps logs/real_reps
export OMP_NUM_THREADS=1
PY="${PYTHON:-python3}"
REPS=${REPS:-5}
NS=hv-phase0
CELL=kedacmp_memory-bound-recreate
log() { echo "[$(date -u +%F' '%T)] $*"; }
model_tag() {
  local f=$1 s=$2
  if [ "$f" = azure ]; then [ "$s" = 0 ] && echo all_azure || echo "all_azure_seed$s"
  else case $s in 0) echo all;; 1|2) echo "all_seed$s";; *) echo "all_bb_seed$s";; esac; fi
}
restore() {
  kubectl -n $NS set resources deploy/web-service -c web --requests=cpu=200m,memory=128Mi \
      --limits=cpu=500m,memory=256Mi >/dev/null 2>&1
}
trap restore EXIT
kubectl uncordon k8s-worker3 k8s-worker4 >/dev/null
for rep in $(seq 0 $((REPS-1))); do
  for fam in azure bitbrains; do
    fa=""; [ $fam = bitbrains ] && fa="--trace-family bitbrains --cluster-tag bb"
    for algo in cvar-bellman-qrdqn qrdqn; do
      out="eval/real_reps/${CELL}__${algo}__${fam}__rep${rep}.csv"
      lg="logs/real_reps/${CELL}__${algo}__${fam}__rep${rep}.log"
      [ -f "$out" ] && continue
      restore
      kubectl -n $NS rollout restart deploy/web-service >/dev/null
      kubectl -n $NS rollout status deploy/web-service --timeout=240s >/dev/null; sleep 10
      taskset -c 5 $PY -u eval_real.py --algo $algo --model-tag "$(model_tag $fam $rep)" --seed $rep \
          --workload memory-bound --start 7670 --steps 20 --mem-actuation recreate $fa \
          --out "$out.tmp" > "$lg" 2>&1 && mv "$out.tmp" "$out"
      log "$CELL $algo $fam rep$rep rc=$? $(grep -o 'SLA [0-9.]*%' "$lg" | tail -1)"
    done
  done
done
log "ALL DONE"
touch eval/.real_mem_recreate_done
