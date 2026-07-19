#!/usr/bin/env bash
# Fails when the test run left uncommitted modifications under npa/src/ —
# tests must not mutate checked-in source.
set -euo pipefail

changed="$(git diff --name-only -- npa/src/ || true)"
if [[ -n "${changed}" ]]; then
  echo "Error: test run modified checked-in source under npa/src/:" >&2
  echo "${changed}" >&2
  git diff --stat -- npa/src/ >&2
  exit 1
fi
