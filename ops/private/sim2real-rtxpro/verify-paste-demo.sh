#!/usr/bin/env bash
# Verify operator paste scripts (pull + install; no cluster submit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

export SIM2REAL_PASTE_SKIP_DEMO=1
export HOME="${WORK}/home"
export NPA_SIM2REAL_DEMO="${WORK}/npa-sim2real-demo"
mkdir -p "${HOME}/.npa/clusters/npa-rtxpro-mk8s" "${NPA_SIM2REAL_DEMO}"

if [[ ! -d "${ROOT}/.git" ]]; then
  echo "SKIP: not a git checkout"
  exit 0
fi

echo "=== test paste-customer-demo (existing repo) ==="
cp -a "${ROOT}" "${NPA_SIM2REAL_DEMO}/nebius-physical-ai"
bash "${ROOT}/ops/private/sim2real-rtxpro/paste-customer-demo.sh"
test -x "${NPA_SIM2REAL_DEMO}/run.sh"
grep -q 'NPA_SIM2REAL_REPO' "${NPA_SIM2REAL_DEMO}/run.sh"

echo "=== test PASTE-NEW-TERMINAL bootstrap (no repo yet) ==="
rm -rf "${NPA_SIM2REAL_DEMO}/nebius-physical-ai"
export NPA_SIM2REAL_DEMO
export SIM2REAL_PASTE_SKIP_DEMO=1
# Run only the bootstrap+paste path; PASTE script honors SIM2REAL_PASTE_SKIP_DEMO.
bash "${ROOT}/ops/private/sim2real-rtxpro/PASTE-NEW-TERMINAL.sh" 2>&1 | tail -5
test -x "${NPA_SIM2REAL_DEMO}/run.sh"

echo "=== test sync-operator-repo clone path ==="
CLONE_ROOT="${WORK}/fresh-clone"
export NPA_SIM2REAL_DEMO="${CLONE_ROOT}/demo"
export NPA_SIM2REAL_REPO="${CLONE_ROOT}/demo/nebius-physical-ai"
# shellcheck source=lib/sync-operator-repo.sh
source "${ROOT}/ops/private/sim2real-rtxpro/lib/sync-operator-repo.sh"
sync_operator_repo "${NPA_SIM2REAL_REPO}" "feat/sim2real-mandatory-stages"
test -f "${NPA_SIM2REAL_REPO}/ops/private/sim2real-rtxpro/mac-run.sh"

echo "operator paste verify OK"
