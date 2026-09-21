#!/usr/bin/env bash
# Regenerate RoboCasa's reviewed CPython 3.12 linux/amd64 hash locks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UV_BIN="${UV_BIN:-uv}"

if [[ "$("${UV_BIN}" --version)" != "uv 0.12.5 (x86_64-unknown-linux-gnu)" ]]; then
  echo "ERROR: RoboCasa locks require uv 0.12.5 for linux/amd64" >&2
  exit 2
fi

# Ignore workstation and CI index configuration. Both selected indexes are
# anonymous, keyring access is disabled, and generated locks emit no credentials.
unset UV_INDEX UV_DEFAULT_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_FIND_LINKS
unset UV_KEYRING_PROVIDER PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS

common=(
  --config-file /dev/null
  --python-version 3.12
  --python-platform x86_64-manylinux_2_28
  --default-index https://pypi.org/simple
  --keyring-provider disabled
  --no-sources
  --generate-hashes
  --quiet
  --custom-compile-command npa/docker/workbench/robocasa/generate-locks.sh
)

"${UV_BIN}" pip compile \
  "${SCRIPT_DIR}/build-requirements.in" \
  "${common[@]}" \
  --output-file "${SCRIPT_DIR}/build-requirements.lock"

"${UV_BIN}" pip compile \
  "${SCRIPT_DIR}/requirements.in" \
  "${common[@]}" \
  --torch-backend cu129 \
  --output-file "${SCRIPT_DIR}/requirements.lock"
