#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${NPA_SMOKE_OUTPUT_DIR:-/tmp/isaac-arena-golden}"
REPLAY_SHA256="154ebea7839ec53e6ac441e18f1404b3fe140c3f004ad7e309519ba37274fa50"
REPLAY_URL="https://media.githubusercontent.com/media/isaac-sim/IsaacLab-Arena/ed0fd12be862078be316c73eb7cf423ba9b1c5cd/isaaclab_arena/tests/test_data/test_demo_gr1_open_microwave.hdf5"
TEMP_DIR="$(mktemp -d /tmp/npa-isaac-arena-smoke.XXXXXX)"
trap 'rm -rf "$TEMP_DIR"' EXIT

REPLAY_PATH="${NPA_ISAAC_ARENA_REPLAY_PATH:-$TEMP_DIR/gr1-open-microwave.hdf5}"
if [[ -z "${NPA_ISAAC_ARENA_REPLAY_PATH:-}" ]]; then
  curl --fail --silent --show-error --location "$REPLAY_URL" --output "$REPLAY_PATH"
fi
printf '%s  %s\n' "$REPLAY_SHA256" "$REPLAY_PATH" | sha256sum -c -

exec npa workbench isaac-arena evaluate \
  --output-path "$OUTPUT_DIR" \
  --environment gr1_open_microwave \
  --policy-type replay \
  --input-path "$REPLAY_PATH" \
  --embodiment gr1_pink \
  --num-episodes 1 \
  --record-video \
  --output-format json
