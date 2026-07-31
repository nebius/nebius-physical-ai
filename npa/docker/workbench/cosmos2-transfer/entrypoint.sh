#!/usr/bin/env bash
# Entrypoint for npa-cosmos2-transfer.
#
# The image previously declared `ENTRYPOINT ["/bin/bash"]`. Kubernetes passes a
# container's `args` to the image ENTRYPOINT, so an orchestrator that supplies only
# args — SkyPilot's Kubernetes provisioner does exactly this, with a keep-alive
# `/bin/bash -c 'trap : TERM INT; sleep infinity & wait'` — ended up running
# `/bin/bash /bin/bash -c ...`, which bash treats as a script path. The container
# exited 126 and SkyPilot reported the confusing `container not found ("ray-node")`
# while setting up its runtime, so every GPU stage pinned to this image failed to
# provision.
#
# Exec'ing the arguments keeps both callers working: an orchestrator's keep-alive
# command runs as given, and a bare `docker run <image>` still lands in a shell.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  exec /bin/bash
fi
exec "$@"
