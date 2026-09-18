#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  exec sleep infinity
fi

case "$1" in
  assert-refusal)
    exec robomimic-runtime assert-refusal
    ;;
  verify-runtime)
    exec robomimic-runtime verify
    ;;
  smoke)
    # No authenticated run/Pod/node/provider allocation observer is wired yet.
    # Refuse with builtins before snapshot/cache mutation or runtime execution.
    printf '%s\n' 'STRICT capacity qualification deferred: authenticated run/Pod-bound provider allocation observation is unavailable' >&2
    exit 78
    ;;
  *)
    exec "$@"
    ;;
esac
