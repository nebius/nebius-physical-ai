#!/usr/bin/env bash
# Stage a real SOMA-CSV motion clip set to S3 for the retargeting-backed live specs.
#
# The retargeting tool feeds NVIDIA's upstream `convert_soma_csv_to_motion_lib.py`, so
# `retargeting.yaml` and `sonic-locomotion-finetuning.yaml` can only be verified live
# with a real SOMA/G1 clip — and this repo does not vendor the dual-licensed upstream
# dataset. `npa.workflows.motion_fixture` synthesizes clips that satisfy that loader's
# documented contract using only the standard library, so no container and no heavy
# dependency is involved: this script just runs it and uploads the result.
#
# The live harness calls the same module automatically when NPA_E2E_SONIC_MOTION_SRC is
# unset, so this script is only needed to stage a fixture ONCE and share it across runs.
#
# Usage:
#   scripts/stage-sonic-motion-fixture.sh --uri s3://<bucket>/<prefix>/motion/
#   export NPA_E2E_SONIC_MOTION_SRC=s3://<bucket>/<prefix>/motion/
#
# Credentials: AWS_* / AWS_ENDPOINT_URL from the environment, as usual for Nebius S3.
set -euo pipefail

URI=""
CLIPS="${NPA_MOTION_FIXTURE_CLIPS:-walk-forward,stand-sway}"
FRAMES="${NPA_MOTION_FIXTURE_FRAMES:-40}"
FPS="${NPA_MOTION_FIXTURE_FPS:-30}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${NPA_LIVE_E2E_PYTHON_BIN:-${REPO_ROOT}/npa/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uri) URI="$2"; shift 2 ;;
    --clips) CLIPS="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$URI" ]] || { echo "ERROR: --uri s3://bucket/prefix/ is required" >&2; exit 2; }
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  [[ -n "${!var:-}" ]] || { echo "ERROR: $var must be set" >&2; exit 2; }
done

WORKDIR="$(mktemp -d -t npa-motion-fixture-XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT INT TERM

PYTHONPATH="${REPO_ROOT}/npa/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m npa.workflows.motion_fixture \
    --output-dir "$WORKDIR" \
    --uri "$URI" \
    --clips "$CLIPS" \
    --frames "$FRAMES" \
    --fps "$FPS"
