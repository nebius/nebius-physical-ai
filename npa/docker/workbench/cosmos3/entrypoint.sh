#!/usr/bin/env bash
# Preserve Cosmos3's convenient mode-based CLI while forwarding orchestrator
# commands exactly. SkyPilot Kubernetes supplies its worker shell as container
# args, so prepending the Typer CLI here would make every worker exit before its
# SSH runtime can be installed.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  # SkyPilot creates the worker pod before it execs bootstrap/setup commands.
  # Keep a no-argv pod alive across that gap instead of exiting after help.
  exec sleep infinity
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
