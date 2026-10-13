#!/usr/bin/env bash
# Full training: all 5 algorithms on the generalist ('all') workload, concurrently.
#   ./run_full.sh [timesteps] [seed]
# CPU-only node -> run concurrently (1 torch thread each) to use the cores.
set -uo pipefail
cd "$(dirname "$0")"

STEPS="${1:-300000}"
SEED="${2:-0}"
ALGOS=("cvar-qrdqn" "qrdqn" "dqn" "ppo" "trpo" "sac-discrete")
export OMP_NUM_THREADS=1
mkdir -p logs eval models

echo "launching ${#ALGOS[@]} algos x ${STEPS} steps (seed $SEED)"
pids=()
for a in "${ALGOS[@]}"; do
  # nice 19: training soaks idle cycles only, never starves the k8s VMs / k3s /
  # experiment drivers sharing this host
  nice -n 19 python3 train.py --algo "$a" --workload all --timesteps "$STEPS" --seed "$SEED" \
      > "logs/train_${a}.log" 2>&1 &
  pids+=($!)
  echo "  $a -> pid $!  (log: logs/train_${a}.log)"
done

fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "pid $p failed"; fail=1; }; done
echo "=== all training done (fail=$fail) ==="
python3 aggregate_eval.py 2>&1 | grep -vE "Gym has been|Please upgrade|migration guide|contact the authors|drop-in|Users of this version"
echo "done."
