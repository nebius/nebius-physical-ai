#!/usr/bin/env bash
set -euo pipefail

changed="$(git diff --name-only -- npa/src/ || true)"
if [[ -n "${changed}" ]]; then
  echo "Warning: uncommitted source changes under npa/src/:" >&2
  echo "${changed}" >&2
fi

