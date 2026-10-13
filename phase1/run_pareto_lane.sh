#!/usr/bin/env bash
# One pinned, serial, IDEMPOTENT training lane for the cost/SLA Pareto sweep (#20).
# Reads a jobs file (one train.py argument string per line, from gen_pareto_jobs.py)
# and runs each job on a single pinned core, nice -19. Skips a job whose model +
# eval CSV already exist, so a lane is safe to kill and restart at any point.
#
# Usage:  run_pareto_lane.sh <core> <lane_id> <jobs_file>
#
# Launch detached (survives logout):
#   setsid nohup ./run_pareto_lane.sh 5 node0-c5 jobs/lane_node0_5.txt \
#       > logs/pareto_lane_node0-c5.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs models eval
export OMP_NUM_THREADS=1
# glibc otherwise trims and regrows the heap on every replay-buffer batch, and the
# brk() page zeroing costs 14-30% of each job's CPU time in the kernel. Raising the
# trim/mmap thresholds removes it (1.16x on node0, 1.40x on the r-nodes); trained
# weights are bit-identical either way, so results are unaffected.
export MALLOC_TRIM_THRESHOLD_=1073741824 MALLOC_TOP_PAD_=268435456 MALLOC_MMAP_THRESHOLD_=1073741824

core="$1"; lane_id="$2"; jobs_file="$3"
PY="${PARETO_PY:-python3}"   # dispatch sets this to the node's venv python

[ -r "$jobs_file" ] || { echo "[pareto-lane $lane_id] no jobs file: $jobs_file" >&2; exit 1; }
total=$(grep -cve '^[[:space:]]*$' "$jobs_file")
echo "[$(date -u +%H:%M:%S)] [pareto-lane $lane_id] core $core, $total jobs, py=$PY"

n=0
while IFS= read -r job || [ -n "$job" ]; do
  case "$job" in ""|\#*) continue;; esac
  n=$((n + 1))
  algo=$(printf '%s\n' "$job" | grep -oP '(?<=--algo )\S+')
  tag=$(printf '%s\n' "$job" | grep -oP '(?<=--tag )\S+')
  case "$job" in
    *"--scaling-mode horizontal"*) algo="${algo}-h-only";;
    *"--scaling-mode vertical"*)   algo="${algo}-v-only";;
  esac
  name="${algo}_all_${tag}"
  csv="eval/eval_${name}.csv"
  # Skip on the eval CSV alone: train.py writes it last (after train + save), so
  # its presence means the job is complete. The model file is not a reliable
  # marker here -- sweep tags end in "_wv1.0" etc., so SB3's save() treats the
  # trailing ".0" as an extension and never appends ".zip".
  if [ -f "$csv" ]; then
    echo "[$(date -u +%H:%M:%S)] [pareto-lane $lane_id] SKIP $n/$total $name (eval CSV present)"
    continue
  fi
  log="logs/train_${name}.log"
  echo "[$(date -u +%H:%M:%S)] [pareto-lane $lane_id] START $n/$total: $job -> $log"
  # shellcheck disable=SC2086
  taskset -c "$core" nice -n 19 "$PY" train.py $job --timesteps 300000 --no-tb > "$log" 2>&1
  rc=$?
  echo "[$(date -u +%H:%M:%S)] [pareto-lane $lane_id] DONE $n/$total $name rc=$rc"
  [ $rc -ne 0 ] && echo "[WARN] [pareto-lane $lane_id] $name exited rc=$rc (see $log)"
done < "$jobs_file"

echo "[$(date -u +%H:%M:%S)] [pareto-lane $lane_id] ALL DONE ($n jobs)"
touch "eval/.pareto_lane_${lane_id}_done"
