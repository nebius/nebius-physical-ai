#!/usr/bin/env bash
# SkyPilot's Kubernetes backend supplies the worker command at pod creation time.
# Keep the image useful as an interactive shell when run without arguments, but
# never replace or swallow an orchestrator-supplied command.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  exec /bin/bash
fi

exec "$@"
