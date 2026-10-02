#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
npa_root="$(cd "$script_dir/../../.." && pwd)"
image="${1:-npa-lerobot-flux3:local}"
docker build -f "$script_dir/Dockerfile" -t "$image" "$npa_root"
