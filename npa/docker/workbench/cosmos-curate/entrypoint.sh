#!/usr/bin/env bash
# Mode-based entrypoint for the npa-cosmos-curate workbench image.
#
# Modes:
#   curate-augmented  curate a run's augmented variants (default workflow stage)
#   curate-videos     curate a local directory of videos
#   fetch-models      download curator model weights with the operator's HF token
#   models            report the model sets and what is present locally
#   engine            report whether the upstream curator can run here
#   smoke             the image's golden eval: a real curation run
#   shell             interactive shell for debugging
#
# Any other first argument is executed as-is, so `docker run <image> npa ...`
# still works.
set -euo pipefail

MODE="${1:-engine}"
shift || true

case "$MODE" in
  curate-augmented|curate-videos|fetch-models|models|engine|plan-pipeline)
    exec npa workbench cosmos-curate "$MODE" "$@"
    ;;
  smoke)
    exec python /opt/npa/docker/workbench/cosmos-curate/smoke_functional.py "$@"
    ;;
  shell)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
