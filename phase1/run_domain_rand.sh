#!/usr/bin/env bash
# Domain-randomization training: headline QR-DQN + Option B (cvar-bellman) over the
# traces_train/ VM pool, seeds 0/1/2 -- the follow-up to the held-out finding that
# the clean-on-all-knobs stability result was a per-representative-VM artifact.
#   ./run_domain_rand.sh [timesteps]
# Gate at 2 concurrent (not 4) so other projects keep half the node -> the 6 runs
# go in 3 serial waves of 2.
set -uo pipefail
cd "$(dirname "$0")"

STEPS="${1:-300000}"
ALGOS=("qrdqn" "cvar-bellman-qrdqn")
SEEDS=(0 1 2)
MAXJOBS=2
export OMP_NUM_THREADS=1
mkdir -p logs eval models

echo "launching ${#ALGOS[@]}x${#SEEDS[@]}=6 domain-rand runs x ${STEPS} steps (<=${MAXJOBS} concurrent)"
for a in "${ALGOS[@]}"; do
  for s in "${SEEDS[@]}"; do
    # throttle to MAXJOBS concurrent nice-19 trainers so we never starve the k8s VMs
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
    tag="dr_seed${s}"
    log="logs/train_${a}_${tag}.log"
    nice -n 19 python3 train.py --algo "$a" --workload all --timesteps "$STEPS" \
        --seed "$s" --domain-rand --tag "$tag" > "$log" 2>&1 &
    echo "  ${a} seed ${s} -> pid $!  (log: $log)"
  done
done

fail=0
for p in $(jobs -rp); do wait "$p" || fail=1; done
echo "=== all domain-rand training done (fail=$fail) ==="
