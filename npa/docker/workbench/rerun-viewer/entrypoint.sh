#!/usr/bin/env bash
# Preserve an orchestrator-supplied argv exactly. With no argv, keep the image
# useful as an interactive Rerun server.
set -euo pipefail

# Build layers contain no SSH host private keys.  SkyPilot supplies the argv
# that keeps this pod alive and starts sshd later, so generate a unique runtime
# identity before forwarding those arguments.
if [[ -n "${NPA_IMAGE_SOURCE_SHA:-}" ]] \
  && [[ ! -s /etc/ssh/ssh_host_ed25519_key ]]; then
  sudo -n /usr/bin/ssh-keygen -A
fi
if [[ -n "${NPA_IMAGE_SOURCE_SHA:-}" ]]; then
  test -s /etc/ssh/ssh_host_ed25519_key
  test -s /etc/ssh/ssh_host_ed25519_key.pub
fi

if [[ $# -eq 0 ]]; then
  exec rerun --serve-web --web-viewer --bind 0.0.0.0 --web-viewer-port 9090 --port 9876
fi

exec "$@"
