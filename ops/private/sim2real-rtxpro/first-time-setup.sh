#!/usr/bin/env bash
# First-time setup (Mac or Linux): prerequisites → private/ templates → virtualenv → run.sh
#
# Usage:
#   bash ops/private/sim2real-rtxpro/first-time-setup.sh
#   NPA_SIM2REAL_DEMO=~/my-demo bash ops/private/sim2real-rtxpro/first-time-setup.sh
#   bash ops/private/sim2real-rtxpro/first-time-setup.sh /path/to/nebius-physical-ai
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:/opt/homebrew/opt/python@3.12/libexec/bin:${HOME}/.nebius/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO="${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}"
BRANCH="${NPA_BRANCH:-feat/sim2real-mandatory-stages}"

if [ -n "${1:-}" ]; then
  REPO="$(cd "$1" && pwd)"
else
  REPO="${DEMO}/nebius-physical-ai"
fi

echo "=== Sim2Real first-time setup ==="
echo "  demo: ${DEMO}"
echo "  repo: ${REPO}"

mkdir -p "${DEMO}"
if [ ! -d "${REPO}/.git" ]; then
  echo "=== git clone --branch ${BRANCH} ==="
  git clone --branch "${BRANCH}" https://github.com/nebius/nebius-physical-ai.git "${REPO}"
fi

OPS="${REPO}/ops/private/sim2real-rtxpro"
for required in install-prereqs.sh setup-customer-demo.sh bootstrap-npa-venv.sh; do
  if [ ! -f "${OPS}/${required}" ]; then
    echo "ERROR: missing ${OPS}/${required}" >&2
    echo "       Pull latest branch ${BRANCH} and re-run, or set NPA_BRANCH to a branch that includes the operator pack." >&2
    exit 1
  fi
done

export NPA_SIM2REAL_DEMO="${DEMO}"

bash "${OPS}/install-prereqs.sh"
bash "${OPS}/setup-customer-demo.sh" "${DEMO}"
bash "${OPS}/bootstrap-npa-venv.sh" "${REPO}"

RUN_SRC="${OPS}/operator-run.sh"
if [ ! -f "${RUN_SRC}" ]; then
  RUN_SRC="${OPS}/mac-run.sh"
fi
cp "${RUN_SRC}" "${DEMO}/run.sh"
chmod +x "${DEMO}/run.sh"

cat <<EOF

=== Done — edit YOUR secrets (once) ===

  ${DEMO}/private/config.yaml
  ${DEMO}/private/credentials.yaml
  ${DEMO}/private/clusters/<k8s-context>/kubeconfig

Then:
  cd ${DEMO} && ./run.sh seed-trigger
  cd ${DEMO} && ./run.sh demo

Daily:
  cd ${DEMO} && ./run.sh demo

See: ${OPS}/QUICKSTART.md
EOF
