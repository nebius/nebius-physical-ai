#!/usr/bin/env bash
# Verify operator paste scripts (pull + install; no cluster submit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

export SIM2REAL_PASTE_SKIP_DEMO=1
export HOME="${WORK}/home"
export NPA_SIM2REAL_DEMO="${WORK}/npa-sim2real-demo"
mkdir -p "${HOME}/.npa/clusters/test-k8s-context" "${NPA_SIM2REAL_DEMO}"

if [[ ! -d "${ROOT}/.git" ]]; then
  echo "SKIP: not a git checkout"
  exit 0
fi

echo "=== test paste-customer-demo (existing repo) ==="
cp -a "${ROOT}" "${NPA_SIM2REAL_DEMO}/nebius-physical-ai"
bash "${ROOT}/ops/private/sim2real-rtxpro/paste-customer-demo.sh"
test -x "${NPA_SIM2REAL_DEMO}/run.sh"

echo "=== test PASTE-NEW-TERMINAL bootstrap ==="
rm -rf "${NPA_SIM2REAL_DEMO}/nebius-physical-ai"
cp -a "${ROOT}" "${NPA_SIM2REAL_DEMO}/nebius-physical-ai"
mkdir -p "${NPA_SIM2REAL_DEMO}/private/clusters/test-k8s-context"
cat > "${NPA_SIM2REAL_DEMO}/private/config.yaml" <<'YAML'
storage:
  bucket: s3://test-bucket
  endpoint_url: https://storage.eu-north1.nebius.cloud
  registry: test.cr.eu-north1.nebius.cloud
  k8s_context: test-k8s-context
YAML
cat > "${NPA_SIM2REAL_DEMO}/private/operator.env" <<'ENV'
TRIGGER_DATASET_URI=s3://test-bucket/sim2real-triggers/test-batch/lerobot-pusht/
TRIGGER_DATASET_ID=lerobot/pusht
ENV
echo "apiVersion: v1" > "${NPA_SIM2REAL_DEMO}/private/clusters/test-k8s-context/kubeconfig"
cat > "${NPA_SIM2REAL_DEMO}/private/credentials.yaml" <<'YAML'
tokens:
  HF_TOKEN: test-token-for-verify
storage:
  aws_access_key_id: test-ak
  aws_secret_access_key: test-sk
YAML
bash "${ROOT}/ops/private/sim2real-rtxpro/PASTE-NEW-TERMINAL.sh" 2>&1 | tail -3
test -x "${NPA_SIM2REAL_DEMO}/run.sh"
test -f "${HOME}/.npa/config.yaml"

echo "=== test first-time-setup.sh (local checkout) ==="
FT_DEMO="${WORK}/first-time-demo"
export NPA_SIM2REAL_DEMO="${FT_DEMO}"
mkdir -p "${FT_DEMO}"
cp -a "${ROOT}" "${FT_DEMO}/nebius-physical-ai"
bash "${ROOT}/ops/private/sim2real-rtxpro/first-time-setup.sh" "${FT_DEMO}/nebius-physical-ai"
test -x "${FT_DEMO}/run.sh"
test -x "${FT_DEMO}/nebius-physical-ai/npa/.venv/bin/npa"
test -f "${ROOT}/ops/private/sim2real-rtxpro/first-time-setup.sh"
test -f "${ROOT}/ops/private/sim2real-rtxpro/bootstrap-npa-venv.sh"
test -f "${ROOT}/ops/private/sim2real-rtxpro/install-prereqs.sh"
test -f "${ROOT}/ops/private/sim2real-rtxpro/setup-customer-demo.sh"

echo "=== test sync-operator-repo clone path ==="
CLONE_ROOT="${WORK}/fresh-clone"
export NPA_SIM2REAL_DEMO="${CLONE_ROOT}/demo"
export NPA_SIM2REAL_REPO="${CLONE_ROOT}/demo/nebius-physical-ai"
# shellcheck source=lib/sync-operator-repo.sh
source "${ROOT}/ops/private/sim2real-rtxpro/lib/sync-operator-repo.sh"
sync_operator_repo "${NPA_SIM2REAL_REPO}" "feat/sim2real-mandatory-stages"
test -f "${NPA_SIM2REAL_REPO}/ops/private/sim2real-rtxpro/operator-run.sh" \
  || test -f "${NPA_SIM2REAL_REPO}/ops/private/sim2real-rtxpro/mac-run.sh"

echo "operator paste verify OK"
