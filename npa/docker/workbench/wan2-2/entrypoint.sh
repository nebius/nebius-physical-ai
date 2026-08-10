#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  wan-runtime)
    shift
    exec /usr/local/bin/wan-runtime "$@"
    ;;
  health|version|status|ensure|warm)
    exec /usr/local/bin/wan-runtime "$@"
    ;;
  "")
    exec /bin/bash
    ;;
  *)
    exec /usr/local/bin/wan-runtime exec "$@"
    ;;
esac
