#!/usr/bin/env bash
set -euo pipefail

readonly root=/opt/npa/embodiedgen
readonly cache_default=/workspace/.cache/npa/embodiedgen

case "${1:-health}" in
  health)
    exec /usr/bin/python3 "$root/runtime-bootstrap.py" health --manifest "$root/runtime-manifest.json"
    ;;
  run-smoke)
    shift
    exec /usr/bin/python3 "$root/runtime-bootstrap.py" run-smoke \
      --manifest "$root/runtime-manifest.json" \
      --smoke "$root/capability_smoke.py" \
      --cache-root "${NPA_EMBODIEDGEN_RUNTIME_CACHE:-$cache_default}" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
