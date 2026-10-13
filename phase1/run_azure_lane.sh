#!/usr/bin/env bash
# One pinned, serial, IDEMPOTENT training lane for the organic-traffic rebuild (#11).
# Generalizes run_seed_lane.sh (which is specific to the seed-tag convention) to
# arbitrary train.py argument strings, since this batch mixes plain retrains,
# --domain-rand runs, and the continuous-action Pandey algos.
#
# Usage:  run_azure_lane.sh <core> <label> "<train.py args>" ["<train.py args>" ...]
# Each job string must include --algo, --tag, and (if applicable) --workload/--domain-rand;
# workload defaults to "all" like train.py itself. Skips a job whose model+eval CSV
# already exist, so a lane can be safely restarted.
#
# Core policy: same as run_seed_lane.sh -- pins to ONE core in {4,5} only, nice -19,
# never cores 0-3 (reserved for the other project on this node).
#
# Launch detached:
#   setsid nohup ./run_azure_lane.sh 5 A "--algo qrdqn --tag azure" "--algo ppo --tag azure" \
#       > logs/azure_lane_A.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs models eval
export OMP_NUM_THREADS=1
core="$1"; label="$2"; shift 2
echo "[$(date -u +%H:%M:%S)] [azure-lane $label] pinned core $core, jobs: $#"
job_num=0
for job in "$@"; do
  job_num=$((job_num + 1))
  algo=$(echo "$job" | grep -oP '(?<=--algo )\S+')
  tag=$(echo "$job" | grep -oP '(?<=--tag )\S+')
  workload=$(echo "$job" | grep -oP '(?<=--workload )\S+' || echo "all")
  name="${algo}_${workload}_${tag}"
  model="models/${name}.zip"; csv="eval/eval_${name}.csv"
  if [ -f "$model" ] && [ -f "$csv" ]; then
    echo "[$(date -u +%H:%M:%S)] [azure-lane $label] SKIP job $job_num ($name, already done)"; continue
  fi
  log="logs/train_${name}.log"
  echo "[$(date -u +%H:%M:%S)] [azure-lane $label core $core] START job $job_num: $job -> $log"
  # shellcheck disable=SC2086
  taskset -c "$core" nice -n 19 python3 train.py $job --timesteps 300000 > "$log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] [azure-lane $label core $core] DONE job $job_num rc=$rc"
  [ $rc -ne 0 ] && echo "[WARN] [azure-lane $label] job $job_num ($job) exited rc=$rc (see $log)"
done
echo "[$(date -u +%H:%M:%S)] [azure-lane $label] ALL DONE"
touch "eval/.azure_lane_${label}_done"
