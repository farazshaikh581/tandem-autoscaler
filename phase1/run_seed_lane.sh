#!/usr/bin/env bash
# One pinned, serial, IDEMPOTENT training lane for the multi-seed headline (#2).
# Usage:  run_seed_lane.sh <core> <label> <algo:seed> [<algo:seed> ...]
# Skips any job whose model AND eval CSV already exist, so a lane can be safely
# restarted and two lanes never redo each other's finished work.
#
# Core policy (2026-07-10): the LLM project (llm_autoscaler, unpinned/floating)
# is reserved 4 logical cores. This lane pins to ONE core in {4,5} only, leaving
# cores 0-3 for the LLM job + QEMU workers + k3s, and runs nice -19 so anything
# that floats onto core 4/5 preempts it. Never point a lane at cores 0-3.
#
# Launch detached:
#   setsid nohup ./run_seed_lane.sh 5 A qrdqn:1 cvar-qrdqn:1 cvar-qrdqn:2 ppo:1 \
#       > logs/lane_A.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs models eval
export OMP_NUM_THREADS=1
core="$1"; label="$2"; shift 2
echo "[$(date -u +%H:%M:%S)] [lane $label] pinned core $core, jobs: $*"
for job in "$@"; do
  a="${job%%:*}"; s="${job##*:}"
  model="models/${a}_all_seed${s}.zip"; csv="eval/eval_${a}_all_seed${s}.csv"
  if [ -f "$model" ] && [ -f "$csv" ]; then
    echo "[$(date -u +%H:%M:%S)] [lane $label] SKIP $a seed $s (already done)"; continue
  fi
  log="logs/train_${a}_seed${s}.log"
  echo "[$(date -u +%H:%M:%S)] [lane $label core $core] START $a seed $s -> $log"
  taskset -c "$core" nice -n 19 python3 train.py --algo "$a" --seed "$s" --tag "seed${s}" \
      --timesteps 300000 > "$log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] [lane $label core $core] DONE  $a seed $s rc=$rc"
  [ $rc -ne 0 ] && echo "[WARN] [lane $label] $a seed $s exited rc=$rc (see $log)"
done
echo "[$(date -u +%H:%M:%S)] [lane $label] ALL DONE"
touch "eval/.lane_${label}_done"
