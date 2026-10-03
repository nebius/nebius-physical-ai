#!/usr/bin/env bash
# Explicit LTX modes funnel through ltx-runtime. Infrastructure bootstrap
# commands must run unchanged: SkyPilot injects task secrets only after its pod
# bootstrap, so gating arbitrary argv here would refuse before the operator's
# scoped entitlement can exist in the task environment. This remains safe
# because the image contains no LTX source or weights; the workflow invokes
# ltx-runtime explicitly before it can use either payload.
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
    exec "$@"
    ;;
esac
