#!/usr/bin/env bash
# Preserve the old build entrypoint without a symlink in source archives.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../cuda13-blackwell/build.sh" "$@"
