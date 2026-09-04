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

# Never bake an SSH host identity into a public image.  Materialize a fresh key
# pair in the pod writable layer before SkyPilot's forwarded keep-alive command
# allows its bootstrap to start sshd.
if [[ -n "${NPA_IMAGE_SOURCE_SHA:-}" ]] \
  && [[ ! -s /etc/ssh/ssh_host_ed25519_key ]]; then
  sudo -n /usr/bin/ssh-keygen -A
fi
if [[ -n "${NPA_IMAGE_SOURCE_SHA:-}" ]]; then
  test -s /etc/ssh/ssh_host_ed25519_key
  test -s /etc/ssh/ssh_host_ed25519_key.pub
fi

if [ "$#" -eq 0 ]; then
  exec /bin/bash
fi

# Direct inference must fail before Hugging Face can start a partial or anonymous
# gated-model download. Check only presence; never print the credential.
for arg in "$@"; do
  if [[ "${arg}" == *"examples/inference.py"* ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is required at run time for gated Cosmos Transfer weights; no download was attempted." >&2
    exit 78
  fi
done
exec "$@"
