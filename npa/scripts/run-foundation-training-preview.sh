#!/usr/bin/env bash
# Build the public-data foundation architecture preview and its MP4 walkthrough.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash npa/scripts/run-foundation-training-preview.sh OUTPUT_DIRECTORY [--html-only]" >&2
  exit 2
fi
command -v uv >/dev/null || { echo "Install uv before running this preview." >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "Install ffmpeg before running this preview." >&2; exit 1; }
if [[ ! -x npa/.venv/bin/python ]]; then
  uv venv --python 3.12 npa/.venv
fi
uv pip install --python npa/.venv/bin/python -e 'npa[adapter,policy-demo]'
npa/.venv/bin/python -m playwright install chromium
output="$1"
shift
exec npa/.venv/bin/python -m npa.workflows.policy_training.foundation_report \
  --output-path "$output" --cache-path "${XDG_CACHE_HOME:-$HOME/.cache}/npa/foundation-preview" "$@"
