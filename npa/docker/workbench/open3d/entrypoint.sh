#!/usr/bin/env bash
# Preserve an orchestrator-supplied argv exactly. SkyPilot's Kubernetes bootstrap
# runs its own shell commands in this container, so the entrypoint must forward
# argv untouched rather than prefixing `npa` onto it. With no argv, report what
# the image can do.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  exec npa workbench open3d --help
fi

exec "$@"
