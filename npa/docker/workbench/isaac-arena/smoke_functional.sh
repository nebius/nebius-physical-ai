#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${NPA_SMOKE_OUTPUT_DIR:-/tmp/isaac-arena-golden}"
exec npa workbench isaac-arena evaluate \
  --output-path "$OUTPUT_DIR" \
  --environment cube_goal_pose \
  --policy-type zero_action \
  --num-episodes 1 \
  --record-video \
  --output-format json
