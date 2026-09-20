#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 ]]; then exec /bin/bash; fi
if [[ "$(basename -- "$1")" == python* && "${2:-}" == -m &&
      "${3:-}" == npa.workflows.habitat_sim_smoke && " $* " != *' --help '* ]]; then
  exec /usr/local/libexec/npa-habitat-runtime "${@:2}"
fi
exec "$@"
