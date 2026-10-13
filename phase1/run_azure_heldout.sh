#!/usr/bin/env bash
# Held-out + domain-rand generalization eval rerun on the azure-retrained models (#11).
# Companion to run_azure_lane.sh: that script retrained the models, this replays
# them on traces_heldout/ (already azure-rebuilt) via eval_heldout.py.
#
# 2 single-rep models (qrdqn, cvar-bellman-qrdqn azure) + 6 domain-rand models
# (same 2 algos x seeds 0/1/2, azure_dr_seed{n}). Pinned to ONE core, serial, nice-19.
#
# Launch detached:
#   setsid nohup ./run_azure_heldout.sh 5 > logs/azure_heldout.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs eval
export OMP_NUM_THREADS=1
core="$1"
echo "[$(date -u +%H:%M:%S)] [azure-heldout] pinned core $core"

run() {
  local algo="$1" tag="$2" seed="$3"
  local name="${algo}_all${tag:+_$tag}"
  local out="eval/heldout_${name}_summary.csv"
  if [ -f "$out" ]; then
    echo "[$(date -u +%H:%M:%S)] [azure-heldout] SKIP $name (already done)"; return
  fi
  local log="logs/heldout_${name}.log"
  echo "[$(date -u +%H:%M:%S)] [azure-heldout] START $name -> $log"
  taskset -c "$core" nice -n 19 python3 eval_heldout.py --algo "$algo" --tag "$tag" --seed "$seed" --cvar-alpha 0.25 > "$log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] [azure-heldout] DONE $name rc=$rc"
  [ $rc -ne 0 ] && echo "[WARN] [azure-heldout] $name exited rc=$rc (see $log)"
}

run qrdqn azure 0
run cvar-bellman-qrdqn azure 0
run qrdqn azure_dr_seed0 0
run qrdqn azure_dr_seed1 1
run qrdqn azure_dr_seed2 2
run cvar-bellman-qrdqn azure_dr_seed0 0
run cvar-bellman-qrdqn azure_dr_seed1 1
run cvar-bellman-qrdqn azure_dr_seed2 2

echo "[$(date -u +%H:%M:%S)] [azure-heldout] ALL DONE"
touch eval/.azure_heldout_done
