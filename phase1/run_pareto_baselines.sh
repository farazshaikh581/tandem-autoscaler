#!/usr/bin/env bash
# MPC + classical arms of the cost/SLA Pareto sweep (#20). Runs ON node0, on a
# pinned core, nice -19. Idempotent: skips a cell whose eval CSV already exists.
#
#   run_pareto_baselines.sh <lane>      lane 0 or 1
#     lane 0 : even-indexed MPC (w_cost,w_viol) cells  +  the classical hpa-target sweep
#     lane 1 : odd-indexed MPC cells
#
# PARETO_FAMILY=bitbrains runs the same cells on the Bitbrains traces (tags 'bb_..'),
# and lane 1 additionally writes the default-weight fixed points for the controllers
# that have no committed Bitbrains row yet (mpc, predictive, predictive-vpa, coscal).
#
# Launch both, detached:
#   for L in 0 1; do setsid nohup ./run_pareto_baselines.sh $L \
#       > logs/pareto_baselines_$L.log 2>&1 < /dev/null & done
set -u
cd "$(dirname "$0")"
mkdir -p logs eval
export OMP_NUM_THREADS=1
# glibc otherwise trims and regrows the heap on every replay-buffer batch, and the
# brk() page zeroing costs 14-30% of each job's CPU time in the kernel. Raising the
# trim/mmap thresholds removes it (1.16x on node0, 1.40x on the r-nodes); trained
# weights are bit-identical either way, so results are unaffected.
export MALLOC_TRIM_THRESHOLD_=1073741824 MALLOC_TOP_PAD_=268435456 MALLOC_MMAP_THRESHOLD_=1073741824

lane="${1:?usage: run_pareto_baselines.sh <0|1>}"
core=$(( lane == 0 ? 0 : 1 ))
PY="${PARETO_PY:-python3}"
EPISODES="${EPISODES:-20}"
FAMILY="${PARETO_FAMILY:-azure}"
case "$FAMILY" in
  azure)     PFX="azure"; FAM_ARGS=();;
  bitbrains) PFX="bb";    FAM_ARGS=(--trace-family bitbrains);;
  *) echo "PARETO_FAMILY must be azure or bitbrains" >&2; exit 2;;
esac

W_COST=(0.0375 0.075 0.15 0.3 0.6 1.2 2.4 4.8)
W_VIOL=(0.5 1.0 2.0 4.0)
HPA_TARGETS=(0.30 0.45 0.60 0.70 0.80 0.90)

run() {  # <core> <tag> <extra args...>
  local c="$1" tag="$2"; shift 2
  taskset -c "$c" nice -n 19 "$PY" eval_baselines.py --episodes "$EPISODES" --tag "$tag" "${FAM_ARGS[@]}" "$@" \
    > "logs/baseline_${tag}.log" 2>&1
}

echo "[$(date -u +%H:%M:%S)] [pareto-baselines lane $lane] core $core, py=$PY, family=$FAMILY"

# ---- default-weight fixed points with no committed Bitbrains row: lane 1 only ----
if [ "$FAMILY" = "bitbrains" ] && [ "$lane" -eq 1 ]; then
  if [ -f "eval/eval_coscal_all_bb.csv" ]; then
    echo "[$(date -u +%H:%M:%S)] SKIP fixed points bb (done)"
  else
    echo "[$(date -u +%H:%M:%S)] START fixed points bb"
    run "$core" "bb" --only mpc,predictive,predictive-vpa,coscal
    echo "[$(date -u +%H:%M:%S)] DONE  fixed points bb rc=$?"
  fi
fi

# ---- MPC: one cell per (w_cost, w_viol), round-robin even/odd by lane ----
i=0
for wv in "${W_VIOL[@]}"; do
  for wc in "${W_COST[@]}"; do
    if [ $(( i % 2 )) -eq "$lane" ]; then
      tag="${PFX}_wc${wc}_wv${wv}"
      if [ -f "eval/eval_mpc_all_${tag}.csv" ]; then
        echo "[$(date -u +%H:%M:%S)] SKIP mpc $tag (done)"
      else
        echo "[$(date -u +%H:%M:%S)] START mpc $tag"
        run "$core" "$tag" --only mpc --w-cost "$wc" --w-viol "$wv"
        echo "[$(date -u +%H:%M:%S)] DONE  mpc $tag rc=$?"
      fi
    fi
    i=$(( i + 1 ))
  done
done

# ---- classical utilisation-target sweep (w_viol-independent): lane 0 only ----
if [ "$lane" -eq 0 ]; then
  for t in "${HPA_TARGETS[@]}"; do
    tag="${PFX}_hpa${t}"
    if [ -f "eval/eval_hpa-only_all_${tag}.csv" ]; then
      echo "[$(date -u +%H:%M:%S)] SKIP classical $tag (done)"
    else
      echo "[$(date -u +%H:%M:%S)] START classical $tag"
      run "$core" "$tag" --only hpa-only,hpa-vpa,predictive,predictive-vpa --hpa-target "$t"
      echo "[$(date -u +%H:%M:%S)] DONE  classical $tag rc=$?"
    fi
  done
fi

echo "[$(date -u +%H:%M:%S)] [pareto-baselines lane $lane] ALL DONE"
touch "eval/.pareto_baselines_${PFX}_${lane}_done"
