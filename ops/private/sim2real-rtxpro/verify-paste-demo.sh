#!/usr/bin/env bash
# Verify paste-customer-demo.sh (pull + install only; no cluster submit).
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

cp -a "${ROOT}" "${NPA_SIM2REAL_DEMO}/nebius-physical-ai"
bash "${ROOT}/ops/private/sim2real-rtxpro/paste-customer-demo.sh"
test -x "${NPA_SIM2REAL_DEMO}/run.sh"
grep -q 'NPA_SIM2REAL_REPO' "${NPA_SIM2REAL_DEMO}/run.sh"
echo "paste-customer-demo verify OK"
