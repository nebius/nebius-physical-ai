#!/usr/bin/env bash
# Overlap full test collection with smoke/guardrail execution on one runner.
set -euo pipefail

cd "$(dirname "$0")/../.."
unset NPA_CI_SHARD_INDEX NPA_CI_TOTAL_SHARDS
precheck_python="$PWD/npa/.venv/bin/python"
precheck_logs="$(mktemp -d "${TMPDIR:-/tmp}/npa-precheck.XXXXXXXX")"
collection_pid=""

cleanup_collection() {
  if [ -n "$collection_pid" ]; then
    kill "$collection_pid" 2>/dev/null || true
    wait "$collection_pid" 2>/dev/null || true
  fi
  rm -rf "$precheck_logs"
}
trap cleanup_collection EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$precheck_python" -m ruff check npa
"$precheck_python" -m ruff format --check npa

# Collection needs no test fixtures and reads the same immutable checkout.
"$precheck_python" -m pytest npa/tests --collect-only -q > "$precheck_logs/collection.log" 2>&1 &
collection_pid=$!
"$precheck_python" -m pytest npa/tests/smoke npa/tests/guardrails -n auto --dist worksteal -q

if wait "$collection_pid"; then
  collection_pid=""
  tail -1 "$precheck_logs/collection.log"
else
  collection_status=$?
  collection_pid=""
  cat "$precheck_logs/collection.log"
  exit "$collection_status"
fi
