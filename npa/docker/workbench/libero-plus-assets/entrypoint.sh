#!/usr/bin/env bash
set -euo pipefail

# The workflow creates a unique LIBERO_CONFIG_PATH before each native import.
# Do not populate a shared LIBERO configuration or a source/asset cache here.
# The image deliberately contains no SSH host keys. SkyPilot can start SSH for
# a worker after this creates keys that exist only for the lifetime of this pod.
if [ "$(id -u)" -eq 0 ]; then
  ssh-keygen -A
else
  sudo -n ssh-keygen -A
fi

exec "$@"
