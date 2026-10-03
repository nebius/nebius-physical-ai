#!/usr/bin/env bash
# Bootstrap and execute the generated policy reference with portable HTML and MP4.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash npa/scripts/run_policy_training_demo.sh OUTPUT_DIRECTORY [--html-only]" >&2
  exit 2
fi
command -v uv >/dev/null || { echo "Install uv before running this demo." >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "Install ffmpeg before running this demo." >&2; exit 1; }
if [[ ! -x npa/.venv/bin/python ]]; then
  uv venv --python 3.12 npa/.venv
fi
uv pip install --python npa/.venv/bin/python -e 'npa[adapter,policy-demo]'
npa/.venv/bin/python -m playwright install chromium
output="$1"
shift
exec npa/.venv/bin/python -m npa.workflows.policy_training.demo \
  --spec workflows/testing/policy-training-slurm.yaml --output "$output" "$@"
