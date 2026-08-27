#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required for guarded Cosmos3-Nano serving}"
: "${NPA_COSMOS3_RAY_TOKEN:?NPA_COSMOS3_RAY_TOKEN is required}"

work="$(mktemp -d /tmp/npa-cosmos3-ray-smoke.XXXXXX)"
server_log="${work}/server.log"
cleanup() {
  if [ -n "${server_pid:-}" ]; then kill "${server_pid}" 2>/dev/null || true; fi
}
trap cleanup EXIT

/usr/local/bin/cosmos3-ray-serve-entrypoint >"${server_log}" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if npa workbench cosmos3 ray-health --endpoint http://127.0.0.1:8000 >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -n 200 "${server_log}" >&2
    exit 1
  fi
  sleep 10
done
npa workbench cosmos3 ray-health --endpoint http://127.0.0.1:8000 >/dev/null

printf '%s\n' '{"model":"Cosmos3-Nano","samples":[{"name":"ray-smoke-a","model_mode":"text2image","prompt":"a red cube on a robotics workbench","seed":17,"num_steps":4},{"name":"ray-smoke-b","model_mode":"text2image","prompt":"a blue cube on a robotics workbench","seed":23,"num_steps":4}]}' >"${work}/batch.json"
npa workbench cosmos3 ray-batch \
  --input-path "${work}/batch.json" \
  --output-path "${work}/result" \
  --endpoint http://127.0.0.1:8000 \
  --timeout 1800 >"${work}/result.json"

/opt/npa/.venv/bin/python - "${work}" <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
result = json.loads((root / "result.json").read_text())
assert result["status"] == "completed"
assert result["backend"] == "cosmos-framework-native-ray-serve"
assert result["batch_size"] == 2
assert result["guardrails"] is True
assert result["max_batch_size"] >= 2
images = list((root / "result" / "artifacts").rglob("*.jpg")) + list(
    (root / "result" / "artifacts").rglob("*.png")
)
assert len(images) >= 2, images
for path in images:
    with Image.open(path) as image:
        image.verify()
print(json.dumps({"status": "PASS", "batch_size": 2, "artifacts": len(images)}))
PY
