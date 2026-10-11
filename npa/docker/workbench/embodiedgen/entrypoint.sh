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
    # SkyPilot needs the image's constrained sudo contract during bootstrap.
    # Runtime-fetched source and its installers must not inherit that power.
    exec /usr/bin/setpriv --no-new-privs /usr/bin/python3 "$root/runtime-bootstrap.py" run-smoke \
      --manifest "$root/runtime-manifest.json" \
      --smoke "$root/capability_smoke.py" \
      --cache-root "${NPA_EMBODIEDGEN_RUNTIME_CACHE:-$cache_default}" "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
