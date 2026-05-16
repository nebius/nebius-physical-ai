#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$NPA_DIR/.venv/bin/npa"
PYTHON_BIN="$NPA_DIR/.venv/bin/python"

TOTAL=0
FAILED=0
WORK_DIR=""

record_pass() {
  TOTAL=$((TOTAL + 1))
  printf 'PASS: %s\n' "$1"
}

record_fail() {
  TOTAL=$((TOTAL + 1))
  FAILED=$((FAILED + 1))
  printf 'FAIL: %s\n' "$1"
}

print_summary() {
  local passed=$((TOTAL - FAILED))
  printf 'SUMMARY: %d/%d steps passed\n' "$passed" "$TOTAL"
}

print_command() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_step() {
  local label="$1"
  shift

  print_command "$@"
  if "$@"; then
    record_pass "$label"
    return 0
  fi

  record_fail "$label"
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT

  if [ -n "$WORK_DIR" ]; then
    rm -rf "$WORK_DIR"
  fi

  print_summary
  if [ "$FAILED" -ne 0 ] || [ "$status" -ne 0 ]; then
    exit 1
  fi
  exit 0
}
trap cleanup EXIT

create_synthetic_dataset() {
  "$PYTHON_BIN" - "$INPUT_DIR" <<'PY'
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
rng = np.random.default_rng(7)

for ep_idx, frames in enumerate((4, 5)):
    ep_dir = root / f"episode_{ep_idx:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    workspace = rng.integers(0, 255, size=(frames, 16, 16, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, size=(frames, 16, 16, 3), dtype=np.uint8)
    state = rng.normal(size=(frames, 8)).astype(np.float32)
    actions = rng.normal(size=(frames, 4)).astype(np.float32)

    np.save(ep_dir / "obs_workspace.npy", workspace)
    np.save(ep_dir / "obs_wrist.npy", wrist)
    np.save(ep_dir / "state.npy", state)
    np.save(ep_dir / "actions.npy", actions)
PY
}

verify_lerobot_dataset() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import json
import sys

import pyarrow.parquet as pq

root = Path(sys.argv[1])
expected = [
    root / "meta" / "info.json",
    root / "meta" / "stats.json",
    root / "meta" / "tasks.parquet",
    root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    root / "data" / "chunk-000" / "file-000.parquet",
    root / "videos" / "observation.images.workspace" / "chunk-000" / "file-000.mp4",
    root / "videos" / "observation.images.workspace" / "chunk-000" / "file-001.mp4",
    root / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4",
    root / "videos" / "observation.images.wrist" / "chunk-000" / "file-001.mp4",
]
missing = [str(path) for path in expected if not path.exists()]
if missing:
    raise SystemExit("missing expected files: " + ", ".join(missing))

info = json.loads((root / "meta" / "info.json").read_text())
assert info["codebase_version"] == "v3.0"
assert info["total_episodes"] == 2
assert info["total_frames"] == 9
assert info["features"]["observation.state"]["shape"] == [8]
assert info["features"]["action"]["shape"] == [4]

data = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
tasks = pq.read_table(root / "meta" / "tasks.parquet")

assert data.num_rows == 9
assert episodes.num_rows == 2
assert tasks.num_rows == 1
PY
}

run_step "npa CLI exists" test -x "$CLI" || exit 1
run_step "python exists" test -x "$PYTHON_BIN" || exit 1

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-adapter-e2e.XXXXXX")"
INPUT_DIR="$WORK_DIR/input"
OUTPUT_DIR="$WORK_DIR/output"
mkdir -p "$INPUT_DIR" || exit 1

run_step "create synthetic Genesis dataset" create_synthetic_dataset || exit 1
run_step "convert synthetic dataset" "$CLI" adapter convert \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --fps 10 \
  --robot franka_panda \
  --task "Synthetic adapter e2e" || exit 1
run_step "verify LeRobot dataset output" verify_lerobot_dataset || exit 1
