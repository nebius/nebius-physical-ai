#!/usr/bin/env bash
# Forward infrastructure bootstrap commands without downloading model payloads.
set -euo pipefail
case "${1:-health}" in
  sam3-runtime) shift; exec /usr/local/bin/sam3-runtime "$@" ;;
  health|version|access|ensure|segment|exec|assert-refusal)
    exec /usr/local/bin/sam3-runtime "$@" ;;
  *) exec "$@" ;;
esac
