#!/usr/bin/env bash
set -euo pipefail

TERRAFORM_DIR="${TERRAFORM_DIR:-deploy/cluster}"
CLUSTER_CONTEXT="${CLUSTER_CONTEXT:-npa-cluster}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-$HOME/.npa/clusters/${CLUSTER_CONTEXT}/kubeconfig}"

require() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "ERROR: ${name} is required" >&2
    exit 2
  fi
}

bucket_name() {
  local value="${NPA_S3_BUCKET:-}"
  value="${value#s3://}"
  printf '%s\n' "${value%%/*}"
}

ensure_s3_bucket() {
  require NPA_PROJECT_ID
  local bucket
  bucket="$(bucket_name)"
  if [ -z "$bucket" ]; then
    echo "S3: skipped, NPA_S3_BUCKET is empty"
    return
  fi
  if nebius storage bucket list --parent-id "$NPA_PROJECT_ID" --format json \
      | grep -F "\"name\":\"${bucket}\"" >/dev/null 2>&1; then
    echo "S3: reused bucket ${bucket}"
    return
  fi
  nebius storage bucket create \
    --name "$bucket" \
    --parent-id "$NPA_PROJECT_ID" \
    --versioning-policy enabled
  echo "S3: created bucket ${bucket}"
}

ensure_k8s() {
  require NPA_PROJECT_ID
  require NPA_TENANT_ID
  if [ -f "$KUBECONFIG_PATH" ]; then
    echo "Kubernetes: reused kubeconfig ${KUBECONFIG_PATH}"
    return
  fi
  export TF_VAR_parent_id="${TF_VAR_parent_id:-$NPA_PROJECT_ID}"
  export TF_VAR_tenant_id="${TF_VAR_tenant_id:-$NPA_TENANT_ID}"
  export TF_VAR_region="${TF_VAR_region:-${NPA_REGION:-eu-north1}}"
  terraform -chdir="$TERRAFORM_DIR" init
  terraform -chdir="$TERRAFORM_DIR" apply -auto-approve
  mkdir -p "$(dirname "$KUBECONFIG_PATH")"
  cluster_id="$(terraform -chdir="$TERRAFORM_DIR" output -json kube_cluster \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
  nebius mk8s cluster get-credentials \
    --id "$cluster_id" \
    --force \
    --kubeconfig "$KUBECONFIG_PATH" \
    --external \
    --context-name "$CLUSTER_CONTEXT"
  echo "Kubernetes: wrote kubeconfig ${KUBECONFIG_PATH}"
}

main() {
  case "${1:-all}" in
    all)
      ensure_s3_bucket
      ensure_k8s
      ;;
    s3)
      ensure_s3_bucket
      ;;
    k8s)
      ensure_k8s
      ;;
    *)
      echo "Usage: $0 [all|s3|k8s]" >&2
      exit 2
      ;;
  esac
}

main "$@"
