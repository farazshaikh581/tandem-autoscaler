#!/usr/bin/env bash
# Action-space ablation: does the joint action space actually matter, or would a
# horizontal-only (replicas) or vertical-only (in-place resize) controller do just
# as well? Retrains the two distributional-value headliners (qrdqn, Option B) under
# a restricted action space on the Azure organic traces, so each becomes its own row
# in comparison_azure.csv next to the joint version. Everything but the action space
# (obs, reward, hyperparams, seed, timesteps) is identical to the headline runs, so
# this isolates the action axis -- the quantitative form of the Phase 0 coupling story.
#
# Node is free -> one pinned core per job, all four run concurrently (nice -19).
# Detach so it survives logout:
#   setsid nohup ./run_ablation.sh > logs/ablation.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs models eval
export OMP_NUM_THREADS=1

# run_job <core> <algo> <horizontal|vertical> [extra train.py args...]
run_job() {
  local core="$1" algo="$2" mode="$3"; shift 3
  local suffix; case "$mode" in horizontal) suffix="-h-only";; vertical) suffix="-v-only";; esac
  local name="${algo}${suffix}_all_azure"
  local model="models/${name}.zip" csv="eval/eval_${name}.csv" log="logs/train_${name}.log"
  if [ -f "$model" ] && [ -f "$csv" ]; then
    echo "[$(date -u +%H:%M:%S)] SKIP $name (already done)"; return
  fi
  echo "[$(date -u +%H:%M:%S)] START $name on core $core -> $log"
  taskset -c "$core" nice -n 19 python3 train.py --algo "$algo" --workload all \
    --scaling-mode "$mode" --tag azure --timesteps 300000 "$@" > "$log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE $name rc=$?"
}

run_job 2 qrdqn              horizontal                    &
run_job 3 qrdqn              vertical                      &
run_job 4 cvar-bellman-qrdqn horizontal --cvar-alpha 0.25  &
run_job 5 cvar-bellman-qrdqn vertical   --cvar-alpha 0.25  &
wait
echo "[$(date -u +%H:%M:%S)] ALL ABLATION JOBS DONE"
touch eval/.ablation_done
