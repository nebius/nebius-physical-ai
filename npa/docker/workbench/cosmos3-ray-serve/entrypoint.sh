#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  args=(
    npa workbench cosmos3 ray-serve
    --world-size "${NPA_COSMOS3_RAY_WORLD_SIZE:-1}" \
    --max-batch-size "${NPA_COSMOS3_RAY_MAX_BATCH_SIZE:-4}" \
    --batch-wait-timeout-s "${NPA_COSMOS3_RAY_BATCH_WAIT_TIMEOUT_S:-0.05}" \
    --host "${NPA_COSMOS3_RAY_HOST:-0.0.0.0}" \
    --port "${NPA_COSMOS3_RAY_PORT:-8000}" \
    --output-path "${NPA_COSMOS3_RAY_OUTPUT_DIR:-/outputs}" \
    --parallelism-preset "${NPA_COSMOS3_RAY_PARALLELISM_PRESET:-throughput}"
  )
  case "${NPA_COSMOS3_RAY_GUARDRAILS:-true}" in
    0|false|FALSE|no|NO|off|OFF) args+=(--no-guardrails) ;;
  esac
  exec "${args[@]}"
fi

case "$1" in
  ray-batch|ray-health|ray-serve)
    command="$1"
    shift
    exec npa workbench cosmos3 "${command}" "$@"
    ;;
  -h|--help)
    exec npa workbench cosmos3 ray-serve --help
    ;;
  shell)
    shift
    exec /bin/bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
