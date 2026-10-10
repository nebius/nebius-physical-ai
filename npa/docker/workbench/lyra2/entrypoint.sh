#!/usr/bin/env bash
# SkyPilot starts an idle pod before its SSH/bootstrap command arrives.
set -euo pipefail
if [ "$#" -eq 0 ]; then
  exec sleep infinity
fi
exec "$@"
