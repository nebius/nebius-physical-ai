#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  hy-world-runtime)
    shift
    exec /usr/local/bin/hy-world-runtime "$@"
    ;;
  health|status|terms|bootstrap-integrity|ensure|fetch-models|run-image-to-world)
    exec /usr/local/bin/hy-world-runtime "$@"
    ;;
  "") exec /bin/bash ;;
  *) exec "$@" ;;
esac
