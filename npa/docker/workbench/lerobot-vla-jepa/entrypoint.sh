#!/bin/sh
# Generate unique host keys only in the running pod, then preserve the stage
# command/arguments selected by NPA or SkyPilot.
set -eu

if [ "$(id -u)" = 0 ]; then
  /usr/bin/ssh-keygen -A
else
  sudo -n /usr/bin/ssh-keygen -A
fi

exec "$@"
