#!/usr/bin/env bash
# Removes ALL Phase 0 resources by deleting its dedicated namespace.
# Scoped to hv-phase0 only; your default/kube-system workloads are untouched.
# NOTE: hv-phase0 now holds two apps (web-service + web-service-io, issue #12) --
# this removes both.
set -euo pipefail
NS="hv-phase0"
echo "Deleting namespace $NS (this removes the Phase 0 apps, HPA, service)..."
kubectl delete ns "$NS" --ignore-not-found
echo "done."
