#!/usr/bin/env bash
# SkyPilot supplies its task argv. Preserve it exactly; the default retains a
# useful, non-networked module help command for private image inspection.
set -euo pipefail

exec "$@"
