#!/usr/bin/env bash
# Multi-seed headline (#2), contenders tier: train seeds 1 & 2 for the four
# policies worth error bars (Option B already has seeds 0/1/2). Serial, one core
# (taskset -c 5 + OMP_NUM_THREADS=1 + nice -19) per the project infra cap.
# Launch detached:  setsid nohup ./run_seeds_contenders.sh > logs/seeds_contenders.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs models eval
export OMP_NUM_THREADS=1

ALGOS="qrdqn cvar-qrdqn ppo dqn"   # headline first
SEEDS="1 2"
i=0; total=8
for a in $ALGOS; do
  for s in $SEEDS; do
    i=$((i+1))
    log="logs/train_${a}_seed${s}.log"
    echo "[$(date -u +%H:%M:%S)] ($i/$total) START $a seed $s -> $log"
    taskset -c 5 nice -n 19 python3 train.py --algo "$a" --seed "$s" --tag "seed${s}" \
        --timesteps 300000 > "$log" 2>&1
    rc=$?
    echo "[$(date -u +%H:%M:%S)] ($i/$total) DONE  $a seed $s rc=$rc"
    [ $rc -ne 0 ] && echo "[WARN] $a seed $s exited rc=$rc (see $log)"
  done
done
echo "[$(date -u +%H:%M:%S)] ALL DONE"
touch eval/.seeds_contenders_done
