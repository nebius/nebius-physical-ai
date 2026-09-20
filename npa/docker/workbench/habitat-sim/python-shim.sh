#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -m && "${2:-}" == npa.workflows.habitat_sim_smoke &&
      " $* " != *' --help '* ]]; then
  exec /usr/local/libexec/npa-habitat-runtime "$@"
fi
exec /usr/bin/python3 "$@"
