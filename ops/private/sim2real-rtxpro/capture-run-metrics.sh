#!/usr/bin/env bash
# Capture per-stage timings, sibling job count, and S3 artifact sizes for a sim2real staged run.
# Usage: capture-run-metrics.sh <run-id>
set -euo pipefail

RUN_ID="${1:?usage: capture-run-metrics.sh <run-id>}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-$HOME/.npa/clusters/npa-rtxpro-mk8s/kubeconfig}"
CTX="${KUBECONTEXT:-npa-rtxpro-mk8s}"
NS="${KUBENS:-default}"
BUCKET="${S3_BUCKET:-}"
PREFIX="${S3_PREFIX:-sim2real-b}"
if [ -z "${BUCKET}" ]; then
  readarray -t _npa_cfg < <("${ROOT}/npa/.venv/bin/python" - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path.home().joinpath(".npa/config.yaml").read_text())
storage = cfg.get("storage") or {}
bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/")[0]
endpoint = storage.get("endpoint_url", "https://storage.eu-north1.nebius.cloud")
print(bucket)
print(endpoint)
PY
)
  BUCKET="${_npa_cfg[0]:-}"
  ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"
else
  ENDPOINT="${S3_ENDPOINT:-https://storage.eu-north1.nebius.cloud}"
fi
if [ -z "${BUCKET}" ]; then
  echo "Set S3_BUCKET or configure storage.bucket in ~/.npa/config.yaml" >&2
  exit 1
fi
OUT="/tmp/sim2real-cluster/${RUN_ID}-metrics.json"
mkdir -p /tmp/sim2real-cluster

echo "=== Sim2Real run metrics: ${RUN_ID} ==="

# --- K8s orchestrator job ---
ORCH_JOB="sim2real-${RUN_ID}"
orch_status="$(kubectl --context "${CTX}" get "job/${ORCH_JOB}" -n "${NS}" \
  -o jsonpath='{.status.conditions[0].type}{" "}{.status.startTime}{" "}{.status.completionTime}' 2>/dev/null || echo 'NOT_FOUND')"
echo "Orchestrator job ${ORCH_JOB}: ${orch_status}"

# --- Sibling component jobs (GPU work) ---
echo ""
echo "--- Sibling K8s jobs (real GPU work) ---"
sibling_jobs="$(kubectl --context "${CTX}" get jobs -n "${NS}" \
  -o json 2>/dev/null | "${ROOT}/npa/.venv/bin/python" -c "
import json, sys
data = json.load(sys.stdin)
run_id = '${RUN_ID}'
rows = []
for item in data.get('items', []):
    name = item['metadata']['name']
    if not name.startswith('s2r-') or run_id[:20] not in name:
        continue
    labels = item['metadata'].get('labels', {})
    component = labels.get('app.kubernetes.io/component', labels.get('component', 'unknown'))
    start = item['status'].get('startTime', '')
    end = item['status'].get('completionTime', '')
    succeeded = item['status'].get('succeeded', 0)
    failed = item['status'].get('failed', 0)
    duration = ''
    if start and end:
        from datetime import datetime
        s = datetime.fromisoformat(start.replace('Z', '+00:00'))
        e = datetime.fromisoformat(end.replace('Z', '+00:00'))
        duration = f'{int((e - s).total_seconds())}s'
    rows.append({'name': name, 'component': component, 'duration': duration, 'succeeded': succeeded, 'failed': failed})
for r in sorted(rows, key=lambda x: x['name']):
    print(f'{r[\"name\"]:60} component={r[\"component\"]:20} duration={r[\"duration\"]:6} ok={r[\"succeeded\"]}')
print(f'SIBLING_COUNT={len(rows)}')
" 2>/dev/null || echo "WARN: could not list sibling jobs")"
echo "${sibling_jobs}"

# --- S3 artifact sizes ---
echo ""
echo "--- S3 artifacts (${BUCKET}/${PREFIX}/${RUN_ID}/) ---"
"${ROOT}/npa/.venv/bin/python" - <<PY
import json, yaml, boto3
from collections import defaultdict
from pathlib import Path
from botocore.config import Config

run_id = "${RUN_ID}"
bucket = "${BUCKET}"
prefix = f"${PREFIX}/{run_id}/"
c = yaml.safe_load(Path.home().joinpath(".npa/credentials.yaml").read_text())
s = c.get("storage") or {}
client = boto3.client(
    "s3",
    endpoint_url="${ENDPOINT}",
    aws_access_key_id=s["aws_access_key_id"],
    aws_secret_access_key=s["aws_secret_access_key"],
    config=Config(signature_version="s3v4"),
    region_name="eu-north1",
)

# GPU sibling stages spawn real work; orchestrator stages 1-2,4-6,9,11-14 are JSON manifests
GPU_COMPONENTS = {
    "cosmos2-transfer", "cosmos2_transfer", "isaac", "genesis",
    "vlm-eval", "vlm_eval", "heldout-eval", "heldout_eval",
    "trainer", "policy", "lerobot",
}
JSON_ONLY_PREFIXES = (
    "stage_01_trigger", "stage_02_assets", "stage_12_external_validation",
    "stage_13_retrigger", "envs/", "tokens/", "augment/manifest.json",
    "outer_loop/", "state/", "training_signal/",
)

cats = defaultdict(lambda: {"count": 0, "bytes": 0})
totals = {"json_manifest": 0, "binary_media": 0, "other": 0, "count": 0}
gpu_io = {"count": 0, "bytes": 0}
json_only = {"count": 0, "bytes": 0}

paginator = client.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        size = obj["Size"]
        rel = key.replace(prefix, "")
        ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
        top = rel.split("/")[0]

        if ext in ("ppm", "png", "jpg", "jpeg", "mp4") or "component-io" in rel:
            kind = "binary_media"
        elif rel.endswith(".json") or ext == "json":
            kind = "json_manifest"
        else:
            kind = "other"

        totals[kind] += size
        totals["count"] += 1
        cats[top]["count"] += 1
        cats[top]["bytes"] += size

        is_gpu_io = "component-io" in rel or top in ("actions", "vlm_eval")
        is_json_stub = any(rel.startswith(p) or rel == p for p in JSON_ONLY_PREFIXES)
        if is_gpu_io:
            gpu_io["count"] += 1
            gpu_io["bytes"] += size
        elif is_json_stub or kind == "json_manifest":
            json_only["count"] += 1
            json_only["bytes"] += size

print(f"Total objects: {totals['count']}")
print(f"  JSON manifests: {totals['json_manifest']:,} bytes")
print(f"  Binary media:   {totals['binary_media']:,} bytes")
print(f"  Other:          {totals['other']:,} bytes")
print()
print("Classification:")
print(f"  GPU sibling I/O + rollouts: {gpu_io['count']} objects, {gpu_io['bytes']:,} bytes (real GPU work artifacts)")
print(f"  JSON manifests / stubs:     {json_only['count']} objects, {json_only['bytes']:,} bytes (orchestrator-only)")
print()
print("Per-prefix breakdown:")
for top in sorted(cats.keys()):
    d = cats[top]
    label = "GPU-IO" if top in ("actions", "component-io", "vlm_eval") else "JSON"
    print(f"  {top:30} [{label:6}] objects={d['count']:4}  bytes={d['bytes']:,}")

# Report summary if available
report_key = prefix + "reports/sim2real-report.json"
try:
    report = json.loads(client.get_object(Bucket=bucket, Key=report_key)["Body"].read())
    summary = {
        "run_id": run_id,
        "sim_backend": report.get("config", {}).get("sim_backend"),
        "reward_trend": report.get("inner_loop", {}).get("reward_trend"),
        "decision": report.get("outer_loop", {}).get("latest_decision"),
        "env_count": report.get("config", {}).get("env_count"),
        "train_fraction": report.get("config", {}).get("train_fraction"),
        "stage_14_rerun_viz_tier": next(
            (c.get("tier") for c in report.get("components", []) if c.get("name") == "stage_14_rerun_viz"),
            None,
        ),
        "visualization_status": report.get("visualization", {}).get("status"),
        "rrd_s3": report.get("s3_artifacts", {}).get("stage_14_rerun_viz_rrd"),
    }
    print()
    print("Report summary:", json.dumps(summary, indent=2))
    Path("/tmp/sim2real-cluster/${RUN_ID}-metrics.json").write_text(
        json.dumps({"summary": summary, "s3_totals": totals, "gpu_io": gpu_io, "json_only": json_only}, indent=2)
    )
except Exception as e:
    print(f"Report not yet available: {e}")
PY

echo ""
echo "Metrics written to ${OUT}"
