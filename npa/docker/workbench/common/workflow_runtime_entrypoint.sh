#!/usr/bin/env bash
# Preserve an orchestrator-supplied argv exactly.
#
# SkyPilot's Kubernetes provisioner passes its keep-alive command as container
# args.  A tool CLI ENTRYPOINT prepends itself to those args and exits before
# Sky can install its runtime; a bare bash ENTRYPOINT treats the first /bin/bash
# argument as a script path.  Exec the supplied argv so both setup and the
# eventual workflow command run in the immutable image that was requested.
set -euo pipefail

# The trusted build deliberately removes package-install host keys so no private
# host identity is shared by every public image consumer.  SkyPilot starts sshd
# after the container entrypoint has forwarded its keep-alive argv, so recreate
# the keys in this pod's writable layer before handing control to that argv.
# Running ssh-keygen only when the baked image identity is present keeps this
# source script safe to exercise directly in hermetic host-side tests.
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

exec "$@"
