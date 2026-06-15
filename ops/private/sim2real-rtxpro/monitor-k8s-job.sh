#!/usr/bin/env bash
set -euo pipefail
JOB="${1:?usage: monitor-k8s-job.sh <job-name>}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.npa/clusters/npa-rtxpro-mk8s/kubeconfig}"
CTX="${KUBECONTEXT:-npa-rtxpro-mk8s}"
LOG="/tmp/sim2real-cluster/${JOB}-monitor.log"
echo "Monitoring ${JOB} on ${CTX}..." | tee "${LOG}"
kubectl --context "${CTX}" wait --for=condition=complete "job/${JOB}" -n default --timeout=7200s 2>&1 | tee -a "${LOG}" || true
POD="$(kubectl --context "${CTX}" get pods -n default -l job-name="${JOB}" -o jsonpath='{.items[0].metadata.name}')"
echo "Pod: ${POD}" | tee -a "${LOG}"
kubectl --context "${CTX}" logs -n default "${POD}" --all-containers=true 2>&1 | tee -a "${LOG}" | tail -80
kubectl --context "${CTX}" get "job/${JOB}" -n default -o wide | tee -a "${LOG}"
