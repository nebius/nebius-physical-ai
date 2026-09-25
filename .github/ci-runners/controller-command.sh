#!/usr/bin/env bash
# Restrict operator SSH commands to the installed controller's pool operations.
set -euo pipefail
case "${1:-}" in
  up|status|enable|disable|down) ;;
  *) echo 'Expected up, status, enable, disable, or down' >&2; exit 2 ;;
esac
test "$#" -eq 1
export HOME=/var/lib/npa-ci-runners
export PATH=/usr/local/bin:/usr/bin:/bin
exec /opt/npa-ci-controller/npa/.venv/bin/python \
  /opt/npa-ci-controller/npa/scripts/ci_cpu_runners.py "$1" \
  --state-dir /var/lib/npa-ci-runners
