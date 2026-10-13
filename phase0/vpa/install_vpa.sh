#!/usr/bin/env bash
# Install the upstream Vertical Pod Autoscaler (pinned 1.8.0) for the Phase 0
# real-VPA runs, scoped to the hv-phase0 namespace.
#
# Components go to kube-system (upstream layout). All three are started with
# --vpa-object-namespace=hv-phase0, so they only look at VPA objects there and
# the admission webhook only selects that namespace. The webhook keeps the
# upstream failurePolicy (Ignore): if it is down, pod creation proceeds.
# Remove everything with ./install_vpa.sh --down.
set -euo pipefail
TAG=vertical-pod-autoscaler-1.8.0
SRC="${VPA_SRC:-${TMPDIR:-/tmp}/vpa-src-1.8.0}"
[ -d "$SRC" ] || git clone -q --depth 1 --branch "$TAG" https://github.com/kubernetes/autoscaler.git "$SRC"
cd "$SRC/vertical-pod-autoscaler"
if [ "${1:-}" = "--down" ]; then
  ./hack/vpa-down.sh
  exit 0
fi
./hack/vpa-up.sh
for d in vpa-recommender vpa-updater vpa-admission-controller; do
  if kubectl -n kube-system get deploy "$d" -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q '"'; then
    op='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--vpa-object-namespace=hv-phase0"}]'
  else
    op='[{"op":"add","path":"/spec/template/spec/containers/0/args","value":["--vpa-object-namespace=hv-phase0"]}]'
  fi
  kubectl -n kube-system get deploy "$d" -o jsonpath='{.spec.template.spec.containers[0].args}' \
    | grep -q 'vpa-object-namespace' || kubectl -n kube-system patch deploy "$d" --type json -p "$op"
  kubectl -n kube-system rollout status deploy "$d" --timeout=300s
done
# The admission controller registers its webhook itself. With
# --ignored-vpa-object-namespaces empty it still takes its "ignored namespaces"
# branch (strings.Split("") is [""]), so the selector comes out as NotIn [""],
# i.e. every namespace. Its informers only watch hv-phase0, so other pods pass
# through unmodified, but pin the selector so they never reach the webhook.
# (It re-registers on restart; rerun this script after restarting it.)
# The controller registers about 20 s after it starts, so wait for a registration newer
# than the running pod's start before patching; patching an older one gets overwritten.
start="$(kubectl -n kube-system get pod -l app=vpa-admission-controller \
  -o jsonpath='{.items[0].status.startTime}')"
for _ in $(seq 1 60); do
  created="$(kubectl get mutatingwebhookconfiguration vpa-webhook-config \
    -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || true)"
  [ -n "$created" ] && [[ "$created" > "$start" ]] && break; sleep 3
done
pin='[{"op":"replace","path":"/webhooks/0/namespaceSelector","value":{"matchExpressions":[{"key":"kubernetes.io/metadata.name","operator":"In","values":["hv-phase0"]}]}}]'
kubectl patch mutatingwebhookconfiguration vpa-webhook-config --type json -p "$pin"
sleep 30
kubectl get mutatingwebhookconfiguration vpa-webhook-config -o jsonpath='{.webhooks[0].namespaceSelector}' \
  | grep -q '"In"' || kubectl patch mutatingwebhookconfiguration vpa-webhook-config --type json -p "$pin"
kubectl get mutatingwebhookconfiguration vpa-webhook-config \
  -o jsonpath='{.webhooks[0].failurePolicy} {.webhooks[0].namespaceSelector}{"\n"}'
