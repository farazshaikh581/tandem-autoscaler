#!/usr/bin/env bash
# Switch the installed VPA 1.8.0 (install_vpa.sh) between two flag profiles.
#
#   ./vpa_profile.sh stock   upstream defaults
#   ./vpa_profile.sh fast    same code, minute-scale flags
#
# Why "fast" exists: with the stock flags the updater only touches a pod whose
# request lies outside [lowerBound, upperBound], or that has lived 12 h and differs
# from the target by at least 10%. The upper bound is P95 x 1.15 x (1 + 1/c), with c
# the history length in days, so for the first hours it sits at maxAllowed and a
# 200m request is always in range: in a 15-minute run the stock VPA never acts.
# "fast" removes the 12 h pod-lifetime gate and shortens the loops and the
# histogram half-life toward the time scale of vpa_emulator.py (15 s loop, newest
# sample weighted 0.7). The 10% minimum change (--pod-update-threshold) and the
# P90 + 15% target are left at their defaults, so "fast" still targets roughly
# 1/1.15 of P90 usage rather than the emulator's 70% of mean usage.
#
# Only the recommender and updater are touched; the admission controller keeps its
# args (restarting it would re-register the webhook with the wrong selector).
set -euo pipefail
PROFILE="${1:?usage: $0 stock|fast}"
NS_ARG="--vpa-object-namespace=hv-phase0"
case "$PROFILE" in
  stock)
    REC="[\"$NS_ARG\"]"
    UPD="[\"$NS_ARG\"]" ;;
  fast)
    REC="[\"$NS_ARG\",\"--recommender-interval=15s\",\"--cpu-histogram-decay-half-life=1m\"]"
    UPD="[\"$NS_ARG\",\"--updater-interval=15s\",\"--in-recommendation-bounds-eviction-lifetime-threshold=0s\"]" ;;
  *) echo "unknown profile $PROFILE"; exit 1 ;;
esac
kubectl -n kube-system patch deploy vpa-recommender --type json \
  -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$REC}]"
kubectl -n kube-system patch deploy vpa-updater --type json \
  -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$UPD}]"
kubectl -n kube-system rollout status deploy/vpa-recommender --timeout=180s
kubectl -n kube-system rollout status deploy/vpa-updater --timeout=180s
