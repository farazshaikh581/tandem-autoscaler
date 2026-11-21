#!/usr/bin/env bash
# IO-app cells of the sim-to-real panel with the host CPU governor pinned to performance.
#
# Each IO-app request writes and fsyncs 1 MiB. With the default intel_pstate powersave
# governor an idle host clocks down, and this request takes ~37 ms instead of ~17 ms,
# above the 20 ms SLO, so at low load the IO app misses the SLO whatever the controller
# does. In July the host also ran training, which kept the clocks up. Pinning the
# governor for the duration of the batch restores that condition; it is restored on exit.
# Same windows, seeds, rps, two-worker topology and fresh deployment per run as
# run_real_reps.sh.
# Output: eval/real_reps/<cell>-perf__<algo>__<family>__rep<k>.csv
#   setsid nohup ./run_real_io_perf.sh > logs/real_io_perf.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p eval/real_reps logs/real_reps
export OMP_NUM_THREADS=1
PY="${PYTHON:-python3}"
REPS=${REPS:-5}
NS=hv-phase0
GOV=/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
ORIG_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
log() { echo "[$(date -u +%F' '%T)] $*"; }
declare -A WIN=( [azure:io]=957 [azure:mixed]=9 [bitbrains:io]=4051 [bitbrains:mixed]=76 )
model_tag() {
  local f=$1 s=$2
  if [ "$f" = azure ]; then [ "$s" = 0 ] && echo all_azure || echo "all_azure_seed$s"
  else case $s in 0) echo all;; 1|2) echo "all_seed$s";; *) echo "all_bb_seed$s";; esac; fi
}
cleanup() {
  echo "$ORIG_GOV" | sudo -n tee $GOV >/dev/null
  kubectl uncordon k8s-worker3 k8s-worker4 >/dev/null 2>&1
  log "governor restored to $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
}
trap cleanup EXIT
echo performance | sudo -n tee $GOV >/dev/null || { log "FATAL: cannot set governor"; exit 1; }
log "governor $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u | tr '\n' ' ')"
kubectl cordon k8s-worker3 k8s-worker4 >/dev/null
for rep in $(seq 0 $((REPS-1))); do
  for fam in azure bitbrains; do
    fa=""; [ $fam = bitbrains ] && fa="--trace-family bitbrains --cluster-tag bb"
    for algo in cvar-bellman-qrdqn qrdqn; do
      for spec in "io-bound_io-burn io-bound io 20" "mixed_io-burn mixed mixed 15"; do
        read -r cell wl key steps <<< "$spec"
        out="eval/real_reps/${cell}-perf__${algo}__${fam}__rep${rep}.csv"
        lg="logs/real_reps/${cell}-perf__${algo}__${fam}__rep${rep}.log"
        [ -f "$out" ] && continue
        kubectl -n $NS rollout restart deploy/web-service-io >/dev/null
        kubectl -n $NS rollout status deploy/web-service-io --timeout=240s >/dev/null; sleep 10
        taskset -c 5 $PY -u eval_real.py --algo $algo --model-tag "$(model_tag $fam $rep)" --seed $rep \
            --workload $wl --start "${WIN[$fam:$key]}" --steps $steps --app io-burn $fa \
            --out "$out.tmp" > "$lg" 2>&1 && mv "$out.tmp" "$out"
        log "${cell}-perf $algo $fam rep$rep rc=$? $(grep -o 'SLA [0-9.]*%' "$lg" | tail -1)"
      done
    done
  done
done
log "ALL DONE"
touch eval/.real_io_perf_done
