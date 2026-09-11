#!/usr/bin/env bash
# Phase A refusal gate. It uses only POSIX-userland tools present in the exact
# Ubuntu base and performs no network operation or package installation.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

verify_phase_a_refusal() {
  incomplete=()
  for name in source-lock.json apt-runtime.lock.json corresponding-source.lock.json; do
    if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "$ROOT/$name"; then
      incomplete+=("$name")
    fi
  done
  if ! grep -Fq '# status: complete' "$ROOT/requirements.lock"; then
    incomplete+=("requirements.lock")
  fi
  if [ "${#incomplete[@]}" -gt 0 ]; then
    printf 'Phase A packaging refusal: incomplete evidence locks: %s\n' \
      "${incomplete[*]}" >&2
    return 65
  fi
  echo "Phase A packaging refusal: complete locks require a new manager-authorized build implementation" >&2
  return 65
}

case "${1:-}" in
  verify-locks|install)
    verify_phase_a_refusal
    ;;
  "")
    exec /bin/sh
    ;;
  *)
    exec "$@"
    ;;
esac
