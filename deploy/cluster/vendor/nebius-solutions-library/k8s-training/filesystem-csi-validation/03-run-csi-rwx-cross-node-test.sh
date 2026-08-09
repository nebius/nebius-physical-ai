#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# File: 03-run-csi-rwx-cross-node-test.sh
# Purpose:
#   Validate ReadWriteMany behavior across nodes by mounting the same PVC into
#   two pods scheduled onto different hosts.
#
# Why We Run This:
#   A single-pod test proves basic functionality, but shared filesystems are
#   most valuable when data written from one node can be read from another. This
#   script confirms that cross-node sharing works in practice.
#
# Reference Docs:
#   https://docs.nebius.com/kubernetes/storage/filesystem-over-csi
#
# What This Script Does:
#   - Applies a RWX PVC plus reader/writer pod manifest
#   - Discovers one Ready CPU node and one Ready >=8-GPU node
#   - Pins one pod to each exact node and records their actual node types
#   - Waits for the PVC and both pods to become ready
#   - Verifies that the PVC inherited the expected default StorageClass
#   - Writes unique checksummed markers in both directions
#
# Usage:
#   ./03-run-csi-rwx-cross-node-test.sh
#
# Optional Environment Variables:
#   TEST_NAMESPACE  Namespace where the validation resources should be created.
#                   Defaults to the current kubectl namespace or default.
#
# Manifest Used:
#   manifests/02-csi-rwx-cross-node.yaml
#
# Created By: Aaron Fagan
# Created On: 2026-03-17
# Version: 0.1.0
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

log_step "Starting cross-node RWX validation"
log_info "Namespace: ${TEST_NAMESPACE}"
log_info "Manifest: ${FILESYSTEM_RWX_MANIFEST_PATH}"
log_info "PVC name: ${FILESYSTEM_RWX_PVC_NAME}"
log_info "Writer pod: ${FILESYSTEM_RWX_WRITER_POD_NAME}"
log_info "Reader pod: ${FILESYSTEM_RWX_READER_POD_NAME}"
log_info "Expected default StorageClass: ${FILESYSTEM_DEFAULT_STORAGE_CLASS_NAME}"

log_step "Checking required local dependencies"
require_command kubectl
require_command jq
require_command sed
log_pass "Required local commands for the RWX validation are available"

log_step "Selecting one Ready CPU node and one Ready eight-GPU node"
NODES_JSON="$(kubectl get nodes -o json)"
CPU_NODE="$(printf '%s' "${NODES_JSON}" | jq -r '
  [.items[]
   | select(any(.status.conditions[]?; .type == "Ready" and .status == "True"))
   | select(((.status.allocatable["nvidia.com/gpu"] // "0") | tonumber) == 0)
   | .metadata.name][0] // empty')"
GPU_NODE="$(printf '%s' "${NODES_JSON}" | jq -r '
  [.items[]
   | select(any(.status.conditions[]?; .type == "Ready" and .status == "True"))
   | select(((.status.allocatable["nvidia.com/gpu"] // "0") | tonumber) >= 8)
   | .metadata.name][0] // empty')"
if [[ -z "${CPU_NODE}" || -z "${GPU_NODE}" || "${CPU_NODE}" == "${GPU_NODE}" ]]; then
  log_fail "Could not resolve distinct Ready CPU and >=8-GPU nodes"
  exit 1
fi
log_info "Selected CPU node: ${CPU_NODE} (allocatable GPUs: 0)"
log_info "Selected GPU node: ${GPU_NODE} (allocatable GPUs: >=8)"
log_pass "Resolved distinct CPU and eight-GPU nodes for deliberate placement"

log_step "Applying the RWX validation manifest"
sed \
  -e "s/__CPU_NODE__/${CPU_NODE}/g" \
  -e "s/__GPU_NODE__/${GPU_NODE}/g" \
  "${FILESYSTEM_RWX_MANIFEST_PATH}" | kubectl apply -n "${TEST_NAMESPACE}" -f -
log_pass "RWX validation manifest applied in namespace '${TEST_NAMESPACE}'"

log_step "Waiting for the RWX PVC to bind"
kubectl wait -n "${TEST_NAMESPACE}" \
  --for=jsonpath='{.status.phase}'=Bound \
  "pvc/${FILESYSTEM_RWX_PVC_NAME}" \
  --timeout=120s
log_info "PVC '${FILESYSTEM_RWX_PVC_NAME}' is bound"
log_pass "RWX PVC '${FILESYSTEM_RWX_PVC_NAME}' bound successfully"

log_step "Verifying that the RWX PVC inherited the default StorageClass"
RWX_STORAGE_CLASS_NAME="$(kubectl get pvc -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_PVC_NAME}" -o jsonpath='{.spec.storageClassName}')"
if [[ -z "${RWX_STORAGE_CLASS_NAME}" ]]; then
  log_fail "RWX PVC '${FILESYSTEM_RWX_PVC_NAME}' did not receive a StorageClass from the cluster default"
  exit 1
fi
if [[ "${RWX_STORAGE_CLASS_NAME}" != "${FILESYSTEM_DEFAULT_STORAGE_CLASS_NAME}" ]]; then
  log_fail "RWX PVC '${FILESYSTEM_RWX_PVC_NAME}' used StorageClass '${RWX_STORAGE_CLASS_NAME}', expected '${FILESYSTEM_DEFAULT_STORAGE_CLASS_NAME}'"
  exit 1
fi
log_info "PVC '${FILESYSTEM_RWX_PVC_NAME}' was assigned StorageClass '${RWX_STORAGE_CLASS_NAME}'"
log_pass "RWX PVC '${FILESYSTEM_RWX_PVC_NAME}' inherited the expected default StorageClass"

log_step "Waiting for both RWX test pods to become ready"
kubectl wait -n "${TEST_NAMESPACE}" \
  --for=condition=Ready \
  "pod/${FILESYSTEM_RWX_WRITER_POD_NAME}" \
  --timeout=180s
kubectl wait -n "${TEST_NAMESPACE}" \
  --for=condition=Ready \
  "pod/${FILESYSTEM_RWX_READER_POD_NAME}" \
  --timeout=180s
log_info "Both RWX test pods are ready"
log_pass "RWX writer and reader pods both reached Ready state"

log_step "Checking the node placement for the reader and writer pods"
WRITER_NODE="$(kubectl get pod -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_WRITER_POD_NAME}" -o jsonpath='{.spec.nodeName}')"
READER_NODE="$(kubectl get pod -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_READER_POD_NAME}" -o jsonpath='{.spec.nodeName}')"

echo "writer node: ${WRITER_NODE}"
echo "reader node: ${READER_NODE}"

WRITER_GPUS="$(kubectl get node "${WRITER_NODE}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
READER_GPUS="$(kubectl get node "${READER_NODE}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
WRITER_GPUS="${WRITER_GPUS:-0}"
READER_GPUS="${READER_GPUS:-0}"
echo "writer node type: CPU (allocatable GPUs: ${WRITER_GPUS})"
echo "reader node type: GPU (allocatable GPUs: ${READER_GPUS})"

if [[ "${WRITER_NODE}" != "${CPU_NODE}" || "${READER_NODE}" != "${GPU_NODE}" ]]; then
  log_fail "Pods did not land on the exact selected CPU and GPU nodes"
  exit 1
fi
if [[ "${WRITER_GPUS}" != "0" || ! "${READER_GPUS}" =~ ^[0-9]+$ || "${READER_GPUS}" -lt 8 ]]; then
  log_fail "Actual node allocatable GPU counts do not prove CPU versus eight-GPU placement"
  exit 1
fi

kubectl get pods -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_WRITER_POD_NAME}" "${FILESYSTEM_RWX_READER_POD_NAME}" -o wide
log_pass "RWX pod placement details collected for both nodes"

log_step "Writing a unique marker on the CPU node and checking it on the GPU node"
CPU_MARKER="cpu-to-gpu:${CPU_NODE}:$(date +%s):$$"
CPU_WRITTEN_SHA="$(kubectl exec -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_WRITER_POD_NAME}" -- sh -lc \
  "printf '%s\\n' '${CPU_MARKER}' > /data/cpu-to-gpu.txt; sha256sum /data/cpu-to-gpu.txt | awk '{print \$1}'")"
CPU_READ_SHA="$(kubectl exec -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_READER_POD_NAME}" -- sh -lc \
  "cat /data/cpu-to-gpu.txt >/dev/null; sha256sum /data/cpu-to-gpu.txt | awk '{print \$1}'")"
if [[ -z "${CPU_WRITTEN_SHA}" || "${CPU_WRITTEN_SHA}" != "${CPU_READ_SHA}" ]]; then
  log_fail "GPU pod did not read the exact checksum written by the CPU pod"
  exit 1
fi
log_pass "GPU pod read the CPU pod's unique marker with checksum ${CPU_READ_SHA}"

log_step "Writing a unique marker on the GPU node and checking it on the CPU node"
GPU_MARKER="gpu-to-cpu:${GPU_NODE}:$(date +%s):$$"
GPU_WRITTEN_SHA="$(kubectl exec -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_READER_POD_NAME}" -- sh -lc \
  "printf '%s\\n' '${GPU_MARKER}' > /data/gpu-to-cpu.txt; sha256sum /data/gpu-to-cpu.txt | awk '{print \$1}'")"
GPU_READ_SHA="$(kubectl exec -n "${TEST_NAMESPACE}" "${FILESYSTEM_RWX_WRITER_POD_NAME}" -- sh -lc \
  "cat /data/gpu-to-cpu.txt >/dev/null; sha256sum /data/gpu-to-cpu.txt | awk '{print \$1}'")"
if [[ -z "${GPU_WRITTEN_SHA}" || "${GPU_WRITTEN_SHA}" != "${GPU_READ_SHA}" ]]; then
  log_fail "CPU pod did not read the exact checksum written by the GPU pod"
  exit 1
fi
log_pass "CPU pod read the GPU pod's unique marker with checksum ${GPU_READ_SHA}"

log_step "Cross-node RWX validation completed successfully"
log_info "The PVC used the expected StorageClass and unique checksummed files were visible in both directions between exact CPU and GPU nodes"
log_pass "Bidirectional CPU/GPU ReadWriteMany behavior and exact node placement confirmed"
