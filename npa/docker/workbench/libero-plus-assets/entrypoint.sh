#!/usr/bin/env bash
set -euo pipefail

# The workflow creates a unique LIBERO_CONFIG_PATH before each native import.
# Do not populate a shared LIBERO configuration or a source/asset cache here.
exec "$@"
