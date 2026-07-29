#!/usr/bin/env bash
# Stage this working tree's `npa` package to an S3 prefix and print NPA_SRC_S3_URI.
#
# Tasks rendered by the npa.workflow renderer install npa from NPA_SRC_S3_URI when
# no workbench image is pinned (the default on this cluster, where
# NPA_E2E_CLEAR_WORKBENCH_IMAGES=1). Staging the *branch* source is what makes a
# live run execute branch code (e.g. npa.workflows.rl_sweep) instead of whatever
# was staged last.
#
# Nothing is hardcoded: bucket/prefix come from flags or environment.
#
# Usage:
#   scripts/stage-npa-src.sh --bucket <bucket> [--prefix npa-workflow-e2e/npa-src] [--repo-root .]
#   NPA_SRC_S3_BUCKET=<bucket> scripts/stage-npa-src.sh
#
# Requires AWS_* credentials + AWS_ENDPOINT_URL (as usual for Nebius S3).
set -euo pipefail

BUCKET="${NPA_SRC_S3_BUCKET:-}"
PREFIX="${NPA_SRC_S3_PREFIX:-npa-workflow-e2e/npa-src}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "ERROR: pass --bucket <bucket> or set NPA_SRC_S3_BUCKET" >&2
  exit 2
fi

PY="${NPA_LIVE_E2E_PYTHON_BIN:-${REPO_ROOT}/npa/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

SRC_DIR="${REPO_ROOT}/npa"
DEST="s3://${BUCKET}/${PREFIX%/}/npa"

"$PY" - "$SRC_DIR" "$BUCKET" "${PREFIX%/}/npa" <<'PY'
import os
import pathlib
import sys

import boto3
from botocore.client import Config

src = pathlib.Path(sys.argv[1]).resolve()
bucket, prefix = sys.argv[2], sys.argv[3].strip("/")

# Noise directories, skipped at any depth.
skip_anywhere = {".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}
# Heavy trees that a rendered task never executes, skipped only at the TOP level.
# (Must not match nested packages such as src/npa/workflows/ — dropping those
# breaks `run.shell` stages that import npa.workflows.*.)
skip_top_level = {"tests", "docker"}
if os.environ.get("NPA_STAGE_INCLUDE_ALL") == "1":
    skip_top_level = set()
kwargs = {"config": Config(signature_version="s3v4")}
endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT")
if endpoint:
    kwargs["endpoint_url"] = endpoint
s3 = boto3.client("s3", **kwargs)

uploaded = 0
for path in src.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(src)
    if any(part in skip_anywhere for part in rel.parts):
        continue
    if rel.parts and rel.parts[0] in skip_top_level:
        continue
    if rel.suffix in {".pyc", ".pyo"}:
        continue
    s3.upload_file(str(path), bucket, f"{prefix}/{rel.as_posix()}")
    uploaded += 1
print(f"staged {uploaded} files -> s3://{bucket}/{prefix}", file=sys.stderr)
PY

echo "$DEST"
