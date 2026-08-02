#!/usr/bin/env bash
# Mode-based entrypoint for the npa-cosmos-evaluator workbench image.
#
# Modes:
#   evaluate          grade a run's augmented variants (default workflow stage)
#   hallucination     score hallucinated motion for one clip pair
#   attribute-verify  verify one clip's attributes with an LLM + VLM pass
#   engine            report which evaluator engine this image resolves to
#   smoke             the image's golden eval: a real hallucination run
#   shell             interactive shell for debugging
#
# Any other first argument is executed as-is, so `docker run <image> npa ...`
# still works.
set -euo pipefail

MODE="${1:-engine}"
shift || true

case "$MODE" in
  evaluate|hallucination|attribute-verify|engine)
    exec npa workbench cosmos-evaluator "$MODE" "$@"
    ;;
  smoke)
    exec python /opt/npa/docker/workbench/cosmos-evaluator/smoke_functional.py "$@"
    ;;
  shell)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
