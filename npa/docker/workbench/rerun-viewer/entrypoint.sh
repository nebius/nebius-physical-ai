#!/usr/bin/env bash
# Preserve an orchestrator-supplied argv exactly. With no argv, keep the image
# useful as an interactive Rerun server.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  exec rerun --serve-web --web-viewer --bind 0.0.0.0 --web-viewer-port 9090 --port 9876
fi

exec "$@"
