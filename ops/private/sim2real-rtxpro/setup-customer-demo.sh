#!/usr/bin/env bash
# Scaffold ~/npa-sim2real-demo/private/ from templates (customer fills YOUR-* values).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEMO="${1:-${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}}"
PRIV="${DEMO}/private"
TEMPLATE="${SCRIPT_DIR}/private"

echo "=== Sim2Real customer demo setup ==="
echo "Demo root: ${DEMO}"
echo ""

mkdir -p "${PRIV}/clusters" "${DEMO}"

_copy_if_missing() {
  local src="$1" dst="$2"
  if [ -f "${dst}" ]; then
    echo "  keep existing ${dst}"
    return 0
  fi
  cp "${src}" "${dst}"
  chmod 600 "${dst}" 2>/dev/null || true
  echo "  created ${dst}"
}

echo "--- Scaffolding private/ from templates ---"
_copy_if_missing "${TEMPLATE}/config.yaml.example" "${PRIV}/config.yaml"
_copy_if_missing "${TEMPLATE}/credentials.yaml.example" "${PRIV}/credentials.yaml"
if [ ! -f "${PRIV}/operator.env" ]; then
  cp "${TEMPLATE}/operator.env.example" "${PRIV}/operator.env"
  chmod 600 "${PRIV}/operator.env"
  echo "  created ${PRIV}/operator.env"
fi
if [ -f "${TEMPLATE}/.gitignore.example" ]; then
  _copy_if_missing "${TEMPLATE}/.gitignore.example" "${PRIV}/.gitignore"
fi

cat <<EOF

=== Next steps ===

1. Edit YOUR files:
   ${PRIV}/config.yaml
   ${PRIV}/credentials.yaml
   ${PRIV}/clusters/<k8s-context>/kubeconfig

2. Seed stock trigger:
   cd ${DEMO} && ./run.sh seed-trigger

3. Run pipeline:
   cd ${DEMO} && ./run.sh demo

Guide: ${SCRIPT_DIR}/QUICKSTART.md
EOF
