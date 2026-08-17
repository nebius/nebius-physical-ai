#!/usr/bin/env bash
# Every path into the container funnels through ltx-runtime, so the entitlement
# checks cannot be sidestepped by passing a command. A bare shell is still
# available for debugging, but it has no LTX code or weights to reach.
set -euo pipefail

case "${1:-}" in
  ltx-runtime)
    # `docker run <image> ltx-runtime <mode>` is how the runbook and the golden
    # eval invoke this, so the literal has to be dropped before forwarding:
    # passing it through makes the mode itself "ltx-runtime" and every such
    # command dies as an unknown mode.
    shift
    exec /usr/local/bin/ltx-runtime "$@"
    ;;
  health|version|status|terms|ensure|warm|fetch-weights|assert-refusal)
    exec /usr/local/bin/ltx-runtime "$@"
    ;;
  "")
    exec /bin/bash
    ;;
  *)
    exec /usr/local/bin/ltx-runtime exec "$@"
    ;;
esac
