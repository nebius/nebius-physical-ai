#!/usr/bin/env bash
set -euo pipefail

case "${1:-serve}" in
  serve)
    exec /opt/npa-venv/bin/uvicorn npa.workbench.alpamayo2_super.service:app \
      --host 0.0.0.0 --port "${PORT:-8080}"
    ;;
  npa)
    exec /opt/npa-venv/bin/npa "${@:2}"
    ;;
  shell)
    exec /bin/bash "${@:2}"
    ;;
  *)
    exec "$@"
    ;;
esac
