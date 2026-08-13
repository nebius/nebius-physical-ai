#!/usr/bin/env bash
# Every path into the container funnels through ltx-runtime, so the licensing
# gate cannot be sidestepped by passing a command. A bare shell is still
# available for debugging, but it has no LTX code or weights to reach.
set -euo pipefail

case "${1:-}" in
  ltx-runtime|health|version|status|terms|provenance|ensure|warm|fetch-weights|assert-refusal)
    exec /usr/local/bin/ltx-runtime "$@"
    ;;
  "")
    exec /bin/bash
    ;;
  *)
    exec /usr/local/bin/ltx-runtime exec "$@"
    ;;
esac
