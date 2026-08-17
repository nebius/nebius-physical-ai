#!/usr/bin/env bash
set -euo pipefail

# The thin Isaac image keeps the exact NPA source in /opt/npa/src without
# installing package metadata, so provide the same executable contract as the
# installed console script while retaining that digest-bound source tree.
exec "${NPA_BAKED_PYTHON:-/opt/npa/sim/venv/bin/python}" -m npa "$@"
