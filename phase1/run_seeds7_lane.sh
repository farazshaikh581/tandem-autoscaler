#!/usr/bin/env bash
# One pinned, serial, idempotent lane that brings every learned controller of the
# main comparison to seven seeds on both trace families.
# Usage: run_seeds7_lane.sh <core> <label> <algo:seed:family> [...]
# family = azure (tag azure_seed<s>) or bitbrains (tag bb_seed<s>).
set -u
cd "$(dirname "$0")"
mkdir -p logs models eval
export OMP_NUM_THREADS=1
# Off-policy SB3 training spends ~30% of its time in brk() heap trim without these.
export MALLOC_TRIM_THRESHOLD_=1073741824 MALLOC_TOP_PAD_=268435456 MALLOC_MMAP_THRESHOLD_=1073741824
PY="${PYTHON:-python3}"
core="$1"; label="$2"; shift 2
echo "[$(date -u +%F' '%H:%M:%S)] [lane $label] core $core, ${#} jobs"
for job in "$@"; do
  IFS=: read -r a s fam <<< "$job"
  if [ "$fam" = azure ]; then tag="azure_seed${s}"; else tag="bb_seed${s}"; fi
  name="${a}_all_${tag}"
  if [ -f "models/${name}.zip" ] && [ -f "eval/eval_${name}.csv" ]; then
    echo "[$(date -u +%F' '%H:%M:%S)] [lane $label] SKIP $name"; continue
  fi
  log="logs/train_${name}.log"
  echo "[$(date -u +%F' '%H:%M:%S)] [lane $label] START $name"
  taskset -c "$core" nice -n 19 "$PY" train.py --algo "$a" --seed "$s" --tag "$tag" \
      --trace-family "$fam" --timesteps 300000 --no-tb > "$log" 2>&1
  echo "[$(date -u +%F' '%H:%M:%S)] [lane $label] DONE  $name rc=$?"
done
echo "[$(date -u +%F' '%H:%M:%S)] [lane $label] ALL DONE"
touch "eval/.seeds7_${label}_done"
