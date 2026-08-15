#!/usr/bin/env bash
# Preserve Cosmos3's convenient mode-based CLI while forwarding orchestrator
# commands exactly. SkyPilot Kubernetes supplies its worker shell as container
# args, so prepending the Typer CLI here would make every worker exit before its
# SSH runtime can be installed.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  exec npa workbench cosmos3 generate --help
fi

MODE="$1"
shift

case "$MODE" in
  checkpoint-eval|generate|reason|text-to-image)
    exec npa workbench cosmos3 "$MODE" "$@"
    ;;
  -h|--help)
    exec npa workbench cosmos3 --help
    ;;
  shell)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
