#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-serve}" == serve ]]; then
  [[ $# == 0 ]] || shift
  exec uvicorn npa.workbench.curobo.service:create_app --factory --host 0.0.0.0 --port 8080 "$@"
fi
exec "$@"
