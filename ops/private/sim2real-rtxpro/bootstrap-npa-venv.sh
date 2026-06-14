#!/usr/bin/env bash
# Create npa virtualenv with pip (Mac/Linux). Idempotent.
# Usage: bootstrap-npa-venv.sh /path/to/nebius-physical-ai
set -euo pipefail

ROOT="${1:?repo root required — e.g. ~/npa-sim2real-demo/nebius-physical-ai}"
VENV="${ROOT}/npa/.venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

if [ ! -f "${ROOT}/npa/pyproject.toml" ]; then
  echo "ERROR: missing ${ROOT}/npa/pyproject.toml" >&2
  exit 1
fi

if [ -x "${PY}" ]; then
  echo "venv ok: ${VENV}"
  exit 0
fi

if ! command -v python3 >/dev/null; then
  echo "ERROR: python3 not found — Mac: brew install python@3.12" >&2
  exit 1
fi

echo "=== Creating virtualenv (python3 -m venv + pip) ==="
echo "  ${VENV}"
python3 -m venv "${VENV}"
"${PIP}" install -U pip
"${PIP}" install -e "${ROOT}/npa"
echo ""
echo "=== Ready ==="
echo "  python: ${PY}"
echo "  npa:    ${VENV}/bin/npa"
echo "  pip:    ${PIP}"
