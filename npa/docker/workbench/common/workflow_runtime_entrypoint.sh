#!/usr/bin/env bash
# Preserve an orchestrator-supplied argv exactly.
#
# SkyPilot's Kubernetes provisioner passes its keep-alive command as container
# args.  A tool CLI ENTRYPOINT prepends itself to those args and exits before
# Sky can install its runtime; a bare bash ENTRYPOINT treats the first /bin/bash
# argument as a script path.  Exec the supplied argv so both setup and the
# eventual workflow command run in the immutable image that was requested.
set -euo pipefail

# Images that include OpenSSH deliberately remove package-generated host keys from
# their layers.  Recreate machine identity only in this container's writable runtime
# layer before SkyPilot asks the service to start.  Non-SSH images pass through.
if command -v ssh-keygen >/dev/null 2>&1 \
  && command -v sudo >/dev/null 2>&1 \
  && sudo -n true >/dev/null 2>&1 \
  && ! compgen -G '/etc/ssh/ssh_host_*_key' >/dev/null; then
  sudo -n ssh-keygen -A >/dev/null
fi

if [ "$#" -eq 0 ]; then
  exec /bin/bash
fi

exec "$@"
