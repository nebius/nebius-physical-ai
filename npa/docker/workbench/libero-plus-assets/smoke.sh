#!/usr/bin/env bash
# Execute the complete licensed-assets camera evidence path in a qualified image.
set -euo pipefail
umask 027

output_root=${NPA_SMOKE_OUTPUT_DIR:?NPA_SMOKE_OUTPUT_DIR is required}
python=${NPA_BAKED_PYTHON:-/opt/openwam-libero/bin/python}
workflow=/opt/npa/src/npa/workflows/libero_plus_assets.py
run_root="${output_root%/}/libero-plus-assets-camera"

test -x "$python"
test -f "$workflow"
mkdir -p "$run_root"
chmod 0750 "$run_root"
export MUJOCO_GL=egl

manifest="$run_root/asset-manifest.json"
scene="$run_root/scene.xml"
assembly="$run_root/scene-assembly.json"
gallery="$run_root/camera-gallery.json"
agentview="$run_root/agentview.png"
agentview_60="$run_root/agentview_60.png"
metrics="$run_root/camera-metrics.json"
report="$run_root/report.json"
rrd="$run_root/gallery.rrd"

"$python" "$workflow" acquire \
  --output-uri "file://$manifest" --scene-uri "file://$scene"
"$python" "$workflow" assemble \
  --asset-manifest-uri "file://$manifest" --output-uri "file://$assembly"
"$python" "$workflow" render \
  --assembly-uri "file://$assembly" --output-uri "file://$gallery" \
  --camera-a-uri "file://$agentview" --camera-b-uri "file://$agentview_60"
"$python" "$workflow" validate \
  --gallery-uri "file://$gallery" --output-uri "file://$metrics"
"$python" "$workflow" report \
  --assembly-uri "file://$assembly" --gallery-uri "file://$gallery" \
  --metrics-uri "file://$metrics" --output-uri "file://$report" --rrd-uri "file://$rrd"

"$python" - "$report" "$rrd" <<'PY'
import json
import sys
from pathlib import Path

report_path, rrd_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
assert report["capability"] == "libero-plus-licensed-assets-camera-compatibility"
assert report["metrics"]["rgb_mean_absolute_difference"] > 0.0
assert report["benchmark_equivalence"] is False
assert report["policy_rollout"] is False
assert report["physical_robot"] is False
assert rrd_path.is_file() and rrd_path.stat().st_size > 0
PY
