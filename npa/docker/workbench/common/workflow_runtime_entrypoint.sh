#!/usr/bin/env bash
# Preserve an orchestrator-supplied argv exactly.
#
# SkyPilot's Kubernetes provisioner passes its keep-alive command as container
# args.  A tool CLI ENTRYPOINT prepends itself to those args and exits before
# Sky can install its runtime; a bare bash ENTRYPOINT treats the first /bin/bash
# argument as a script path.  Exec the supplied argv so both setup and the
# eventual workflow command run in the immutable image that was requested.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  exec /bin/bash
fi

exec "$@"
