#!/usr/bin/env bash
# Submit the complete public-data VLA pipeline from an installed Workbench checkout.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$repo_root/npa/.venv/bin/python" -m npa.workflows.policy_training.public_vla_launch "$@"
