#!/usr/bin/env bash
# KubeRay replaces image entrypoints, so its Ray command must bootstrap too.
set -euo pipefail
exec /usr/local/bin/npa-cosmos3-nano-bootstrap ray "$@"
