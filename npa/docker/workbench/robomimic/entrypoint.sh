#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  exec sleep infinity
fi

case "$1" in
  assert-refusal)
    exec robomimic-runtime assert-refusal
    ;;
  verify-runtime)
    exec robomimic-runtime verify
    ;;
  smoke)
    exec robomimic-runtime exec /opt/npa/robomimic/smoke.py
    ;;
  *)
    exec "$@"
    ;;
esac
