#!/usr/bin/env bash
# Expose the native job runtime while keeping CUDA and weights runtime-only.
set -euo pipefail

case "${1:-health}" in
  health|version)
    test -s /opt/byof/npa_source_metadata.json
    test -r /opt/npa/wan2-2/runtime-requirements.txt
    cat /opt/byof/npa_source_metadata.json
    ;;
  ensure|status)
    exec wan-runtime "$@"
    ;;
  exec)
    shift
    test "$#" -gt 0
    wan-runtime ensure
    export PATH="${NPA_WAN_RUNTIME_CACHE}/current/venv/bin:$PATH"
    export PYTHONPATH="/opt/byof${PYTHONPATH:+:$PYTHONPATH}"
    exec "$@"
    ;;
  golden)
    shift
    wan-runtime ensure
    export PYTHONPATH="/opt/npa-native:/opt/byof${PYTHONPATH:+:$PYTHONPATH}"
    exec "${NPA_WAN_RUNTIME_CACHE}/current/venv/bin/python" \
      -m npa.solutions.native_image_smoke "$@"
    ;;
  *)
    printf 'Usage: model-runtime {health|version|status|ensure|exec COMMAND|golden OPTIONS}\n' >&2
    exit 64
    ;;
esac
