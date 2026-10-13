#!/usr/bin/env bash
# KEDA production-autoscaler baseline on the real k3s cluster: the `keda` /
# `keda-vpa` pair, measured live over a cpu-bound and a memory-bound trace window.
# Analogue of the sim hpa-only / hpa-vpa rows, but with the actual KEDA operator
# driving horizontal scaling.
#
# Also re-runs the RL headliners (qrdqn, Option B; `all_azure` models) with no
# ScaledObject present, on the SAME cluster state and windows, tagged `kedacmp`,
# so the RL-vs-KEDA comparison is matched-conditions rather than against the
# months-old real_<algo>_<workload>.csv from an earlier trace + testbed topology.
#
# Prereq: KEDA installed once (see keda/README.md) and the venv active.
# Sequential by design -- every phase scales the SAME web-service deployment.
# Real-cluster I/O, not training: no core pinning.
# KEDA_FAMILY=bitbrains repeats it on the Bitbrains traces: original Phase 1 `all`
# models, Bitbrains busy windows, every output suffixed `_bb`.
# Detach so it survives logout:
#   source venv/bin/activate
#   setsid nohup ./run_keda_real.sh > logs/keda_real.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
mkdir -p logs eval
KEDA_DIR=keda
STEPS=20
# Busy windows in the CURRENT traces (cpu-bound.npz / memory-bound.npz, regen'd
# Jul 2025). cpu-bound: index 1095 starts moderate and ramps into sustained
# overload (mean load ~1.2, 15/20 steps over 1.0) -- the old --start 3800 went
# idle after the trace regen. memory-bound: index 7670 holds mem_frac ~0.91
# (working set ~435 MiB vs 128 MiB base -> severe OOM pressure), unchanged.
WINDOWS=("cpu-bound 1095" "memory-bound 7670")
FAMILY="${KEDA_FAMILY:-azure}"
case "$FAMILY" in
  azure) FAM_ARGS=(); KEDA_TAG=(); RL_MODEL=all_azure; RL_TAG=kedacmp; LOG_SFX="";;
  # Bitbrains windows: the busy ones the original Phase 1 real-cluster runs used.
  # cpu-bound 3800 sits near load 1.0 for the whole window (this trace peaks at
  # 1.0, so there is no overload ramp to match Azure's); memory-bound shares its
  # memory trace with Azure, so 7670 is the same memory window.
  bitbrains) WINDOWS=("cpu-bound 3800" "memory-bound 7670")
             FAM_ARGS=(--trace-family bitbrains); KEDA_TAG=(--cluster-tag bb)
             RL_MODEL=all; RL_TAG=kedacmp_bb; LOG_SFX=_bb;;
  *) echo "KEDA_FAMILY must be azure or bitbrains" >&2; exit 2;;
esac

preflight() {   # KEDA must already be installed cluster-wide (see keda/README.md).
                # A missing CRD makes `kubectl apply` fail while the eval still
                # runs -- against a static, un-autoscaled deployment -- so the
                # `keda` rows would look fine and mean nothing. Fail loud instead.
  kubectl get crd scaledobjects.keda.sh >/dev/null 2>&1 || {
    echo "FATAL: KEDA CRDs not found. Install once (see phase1/keda/README.md):"
    echo "  kubectl apply --server-side -f https://github.com/kedacore/keda/releases/download/v2.16.1/keda-2.16.1.yaml"
    exit 1
  }
  kubectl -n keda rollout status deploy/keda-operator --timeout=60s || {
    echo "FATAL: keda-operator not Ready"; exit 1
  }
}

wait_ready() {   # wait for the ScaledObject's managed HPA to come Ready
  for _ in $(seq 1 30); do
    kubectl get scaledobject web-service-keda -n hv-phase0 \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | grep -q True && return 0
    sleep 2
  done
  echo "[$(date -u +%H:%M:%S)] WARN scaledobject not Ready after 60s (continuing)"
}

run_keda_variant() {   # <manifest> <label> [extra eval_real_keda.py args...]
  local manifest="$1" label="$2"; shift 2
  local extra=("$@")   # e.g. --vpa; saved before `set -- $ws` below clobbers $@
  echo "[$(date -u +%H:%M:%S)] === $label: apply $manifest ==="
  kubectl apply -f "$KEDA_DIR/$manifest"
  wait_ready
  for ws in "${WINDOWS[@]}"; do
    set -- $ws; local wl="$1" start="$2"
    echo "[$(date -u +%H:%M:%S)] --- $label $wl (start $start) ---"
    python eval_real_keda.py --workload "$wl" --start "$start" --steps "$STEPS" "${extra[@]}" \
      "${FAM_ARGS[@]}" "${KEDA_TAG[@]}" > "logs/keda_real_${label}_${wl}${LOG_SFX}.log" 2>&1
    echo "[$(date -u +%H:%M:%S)]     rc=$? -> logs/keda_real_${label}_${wl}${LOG_SFX}.log"
  done
  echo "[$(date -u +%H:%M:%S)] === $label: delete $manifest ==="
  kubectl delete -f "$KEDA_DIR/$manifest"
  sleep 5
}

run_rl_ref() {   # <algo> -- matched RL reference run, no ScaledObject. Uses the
                 # canonical single azure model (`all_azure`), the same one the
                 # earlier real-cluster tables replayed -- not the `_seedN`
                 # multi-seed set (#15-18), which exists for sim significance
                 # testing, not real-cluster replay.
  local algo="$1"
  for ws in "${WINDOWS[@]}"; do
    set -- $ws; local wl="$1" start="$2"
    echo "[$(date -u +%H:%M:%S)] --- rl-ref $algo $wl (start $start) ---"
    python eval_real.py --algo "$algo" --model-tag "$RL_MODEL" --workload "$wl" \
      --start "$start" --steps "$STEPS" --cluster-tag "$RL_TAG" "${FAM_ARGS[@]}" \
      > "logs/keda_real_rlref_${algo}_${wl}${LOG_SFX}.log" 2>&1
    echo "[$(date -u +%H:%M:%S)]     rc=$? -> logs/keda_real_rlref_${algo}_${wl}${LOG_SFX}.log"
  done
}

preflight

# make sure nothing is left over from a prior run
kubectl delete scaledobject web-service-keda -n hv-phase0 --ignore-not-found

# Both variants use the CPU-only ScaledObject (KEDA never scales on memory here --
# it mirrors the sim HPA, which is CPU-only). `keda` leaves the memory request at
# baseline; `keda-vpa` adds the driver's --vpa memory rule.
run_keda_variant scaledobject-cpu.yaml keda
run_keda_variant scaledobject-cpu.yaml keda-vpa --vpa
run_rl_ref qrdqn
run_rl_ref cvar-bellman-qrdqn

echo "[$(date -u +%H:%M:%S)] ALL KEDA-REAL RUNS DONE"
python compare_keda.py --family "$FAMILY"
touch "eval/.keda_real${LOG_SFX}_done"
