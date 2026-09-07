#!/usr/bin/env bash
set -euo pipefail
# The Kubernetes worker needs fresh per-pod SSH keys, never keys baked in a layer.
sudo -n ssh-keygen -A
if [[ "$#" -eq 0 ]]; then
  set -- /opt/venv/bin/python /opt/ncore/bin/verify-packaging.py
fi
exec "$@"
