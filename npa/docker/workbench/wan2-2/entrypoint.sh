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
    # SkyPilot's bootstrap runs before task-level runtime secrets exist. Pass
    # infrastructure commands through unchanged; the image has no CUDA Python
    # or model bytes to expose, and the workflow invokes wan-runtime explicitly
    # before importing or executing Wan on a GPU.
    exec "$@"
    ;;
esac
