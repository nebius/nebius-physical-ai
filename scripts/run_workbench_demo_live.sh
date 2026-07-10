#!/usr/bin/env bash
# Live Nebius Physical AI workbench demo (Cosmos + Isaac Lab + GR00T + FiftyOne).
# Run on the operator/dev VM with ~/.npa configured and nebius profile active.
#
# Adapts docs/demo/8gpu-h200.md to the rtxpro project shape:
# - 8x H200 managed VM hosts Cosmos / GR00T / FiftyOne (shared host, distinct ports)
# - 1x RTX PRO 6000 managed VM hosts Isaac Lab (RT cores)
# - Regenerates pipeline artifacts when the historical demo-prestage bucket is absent
set -euo pipefail

REPO_PATH="${REPO_PATH:-$HOME/nebius-physical-ai}"
NPA_BIN="${NPA_BIN:-$REPO_PATH/npa/.venv/bin/npa}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_PATH/npa/.venv/bin/python}"

PROJECT_ALIAS="${PROJECT_ALIAS:-rtxpro}"
REGION="${REGION:-us-central1}"
# Required — set from your Nebius project / ~/.npa config (never commit values):
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
TENANT_ID="${TENANT_ID:?set TENANT_ID}"
TARGET_BUCKET="${TARGET_BUCKET:?set TARGET_BUCKET}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DEMO_PREFIX="${DEMO_PREFIX:-demo-arch-${RUN_TAG}}"

COSMOS_ALIAS="${COSMOS_ALIAS:-${DEMO_PREFIX}-cosmos}"
GROOT_ALIAS="${GROOT_ALIAS:-${DEMO_PREFIX}-groot}"
FIFTYONE_ALIAS="${FIFTYONE_ALIAS:-${DEMO_PREFIX}-fiftyone}"
ISAAC_ALIAS="${ISAAC_ALIAS:-${DEMO_PREFIX}-isaac}"

COSMOS_PORT="${COSMOS_PORT:-8081}"
GROOT_PORT="${GROOT_PORT:-8082}"
FIFTYONE_PORT="${FIFTYONE_PORT:-5151}"

H200_GPU_TYPE="${H200_GPU_TYPE:-gpu-h200-sxm}"
H200_GPU_PRESET="${H200_GPU_PRESET:-8gpu-128vcpu-1600gb}"
H200_GPU_COUNT="${H200_GPU_COUNT:-8}"

ISAAC_GPU_TYPE="${ISAAC_GPU_TYPE:-gpu-rtx6000}"
ISAAC_GPU_PRESET="${ISAAC_GPU_PRESET:-1gpu-24vcpu-218gb}"
ISAAC_GPU_COUNT="${ISAAC_GPU_COUNT:-1}"

COSMOS_MODEL="${COSMOS_MODEL:-nvidia/Cosmos-1.0-Diffusion-7B-Text2World}"
GROOT_MODEL="${GROOT_MODEL:-nvidia/GR00T-N1.7-3B}"
GROOT_EMBODIMENT="${GROOT_EMBODIMENT:-REAL_G1}"
ISAAC_TASK="${ISAAC_TASK:-Isaac-Velocity-Flat-G1-v0}"
COSMOS_PROMPT="${COSMOS_PROMPT:-humanoid robot carries a red cube around obstacles}"

TARGET_BUCKET_URI="${TARGET_BUCKET_URI:-s3://${TARGET_BUCKET}/demo-8gpu-h200/${RUN_TAG}}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SOURCE_CIDR="${SOURCE_CIDR:-}"
NPA_NEBIUS_PROFILE="${NPA_NEBIUS_PROFILE:-npa-mk8s}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
TEARDOWN="${TEARDOWN:-0}"
STATE_DIR="${STATE_DIR:-$HOME/.npa/demo-runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
cd "$REPO_PATH"

export NPA_NEBIUS_PROFILE
if command -v nebius >/dev/null 2>&1; then
  nebius profile activate "$NPA_NEBIUS_PROFILE" >/dev/null || true
fi

if [[ -f "$HOME/.npa/live-e2e.env" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$HOME/.npa/live-e2e.env"; set +a
fi

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG_DIR/demo.log"; }
fail() { log "ERROR: $*"; exit 1; }

require_bin() {
  [[ -x "$1" ]] || fail "missing executable: $1"
}

workbench_field() {
  local tool="$1" alias="$2" field="$3"
  "$PYTHON_BIN" - "$tool" "$alias" "$field" <<'PY'
import sys, yaml
from pathlib import Path
tool, alias, field = sys.argv[1:4]
cfg = yaml.safe_load(Path.home().joinpath(".npa/config.yaml").read_text())
projects = cfg.get("projects") or {}
# Prefer explicit project alias from env via NPA_DEMO_PROJECT if set later; scan all.
hits = []
for pname, proj in projects.items():
    wbs = (proj.get("workbenches") or {})
    # tool-specific nesting varies; also check flat aliases under projects.
    for key, wb in wbs.items():
        if key == alias or (isinstance(wb, dict) and wb.get("alias") == alias):
            hits.append(wb)
# Fallback: npa stores per-tool under ~/.npa sometimes only via CLI status.
if not hits:
    print("")
    raise SystemExit(0)
wb = hits[-1]
val = wb.get(field) if isinstance(wb, dict) else ""
print(val or "")
PY
}

resolve_source_cidr() {
  if [[ -n "$SOURCE_CIDR" ]]; then
    echo "$SOURCE_CIDR"
    return
  fi
  local ip
  ip="$(curl -4 -fsS --max-time 5 ifconfig.me 2>/dev/null || true)"
  if [[ -n "$ip" ]]; then
    echo "${ip}/32"
  else
    echo "0.0.0.0/0"
  fi
}

save_env() {
  cat >"$STATE_DIR/demo.env" <<EOF
REPO_PATH=$REPO_PATH
PROJECT_ALIAS=$PROJECT_ALIAS
PROJECT_ID=$PROJECT_ID
TENANT_ID=$TENANT_ID
REGION=$REGION
RUN_TAG=$RUN_TAG
TARGET_BUCKET_URI=$TARGET_BUCKET_URI
COSMOS_ALIAS=$COSMOS_ALIAS
GROOT_ALIAS=$GROOT_ALIAS
FIFTYONE_ALIAS=$FIFTYONE_ALIAS
ISAAC_ALIAS=$ISAAC_ALIAS
COSMOS_PORT=$COSMOS_PORT
GROOT_PORT=$GROOT_PORT
FIFTYONE_PORT=$FIFTYONE_PORT
H200_GPU_TYPE=$H200_GPU_TYPE
H200_GPU_PRESET=$H200_GPU_PRESET
ISAAC_GPU_TYPE=$ISAAC_GPU_TYPE
ISAAC_GPU_PRESET=$ISAAC_GPU_PRESET
ISAAC_TASK=$ISAAC_TASK
GROOT_EMBODIMENT=$GROOT_EMBODIMENT
COSMOS_MODEL=$COSMOS_MODEL
GROOT_MODEL=$GROOT_MODEL
EOF
}

status_json() {
  local tool="$1" alias="$2"
  "$NPA_BIN" workbench "$tool" -p "$PROJECT_ALIAS" -n "$alias" status --output-format json 2>/dev/null \
    || "$NPA_BIN" workbench "$tool" -p "$PROJECT_ALIAS" -n "$alias" status --json 2>/dev/null \
    || "$NPA_BIN" workbench "$tool" -p "$PROJECT_ALIAS" -n "$alias" status 2>&1 | tee -a "$LOG_DIR/${tool}-status.txt"
}

deploy_cosmos_h200() {
  log "Deploying Cosmos on ${H200_GPU_COUNT}x H200 (${COSMOS_ALIAS})"
  "$NPA_BIN" workbench cosmos -p "$PROJECT_ALIAS" -n "$COSMOS_ALIAS" deploy \
    --runtime vm \
    --project-id "$PROJECT_ID" \
    --tenant-id "$TENANT_ID" \
    --region "$REGION" \
    --gpu-type "$H200_GPU_TYPE" \
    --gpu-preset "$H200_GPU_PRESET" \
    --gpu-count "$H200_GPU_COUNT" \
    --server-port "$COSMOS_PORT" \
    --model "$COSMOS_MODEL" \
    --yes \
    2>&1 | tee "$LOG_DIR/cosmos-deploy.log"
}

deploy_isaac_rtx() {
  log "Deploying Isaac Lab on RTX PRO (${ISAAC_ALIAS})"
  "$NPA_BIN" workbench isaac-lab -p "$PROJECT_ALIAS" -n "$ISAAC_ALIAS" deploy \
    --runtime vm \
    --project-id "$PROJECT_ID" \
    --tenant-id "$TENANT_ID" \
    --region "$REGION" \
    --gpu-type "$ISAAC_GPU_TYPE" \
    --gpu-preset "$ISAAC_GPU_PRESET" \
    --yes \
    2>&1 | tee "$LOG_DIR/isaac-deploy.log"
}

wait_for_host() {
  local tool="$1" alias="$2" tries="${3:-90}"
  local host=""
  for ((i=1; i<=tries; i++)); do
    host="$("$NPA_BIN" workbench "$tool" -p "$PROJECT_ALIAS" -n "$alias" status --output-format json 2>/dev/null \
      | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("host") or d.get("public_ip") or d.get("ip") or "")' 2>/dev/null || true)"
    if [[ -z "$host" ]]; then
      host="$("$PYTHON_BIN" - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path.home().joinpath(".npa/config.yaml").read_text())
# Search nested workbench records for alias
alias="$alias"
found=""
def walk(obj):
    global found
    if isinstance(obj, dict):
        if obj.get("alias")==alias or False:
            found = obj.get("host") or obj.get("public_ip") or obj.get("ip") or found
        for k,v in obj.items():
            if k==alias and isinstance(v, dict):
                found = v.get("host") or v.get("public_ip") or v.get("ip") or found
            walk(v)
    elif isinstance(obj, list):
        for i in obj: walk(i)
walk(cfg)
print(found or "")
PY
)"
    fi
    if [[ -n "$host" && "$host" != "<pending>" ]]; then
      echo "$host"
      return 0
    fi
    sleep 20
  done
  return 1
}

deploy_shared_h200_services() {
  local host="$1"
  local instance_id="$2"
  local cidr
  cidr="$(resolve_source_cidr)"
  log "Ensuring ingress on $instance_id for ports ${FIFTYONE_PORT},${COSMOS_PORT},${GROOT_PORT} from $cidr"
  "$NPA_BIN" network ensure-ingress \
    --vm "$instance_id" \
    --ports "${FIFTYONE_PORT},${COSMOS_PORT},${GROOT_PORT}" \
    --source "$cidr" \
    --tool physical-ai-demo \
    2>&1 | tee "$LOG_DIR/ingress.log" || log "WARN: ensure-ingress returned non-zero (continuing)"

  log "Deploying GR00T BYOVM on $host"
  "$NPA_BIN" workbench groot -p "$PROJECT_ALIAS" -n "$GROOT_ALIAS" deploy \
    --runtime byovm \
    --host "$host" \
    --ssh-user "$SSH_USER" \
    --ssh-key "$SSH_KEY" \
    --project-id "$PROJECT_ID" \
    --tenant-id "$TENANT_ID" \
    --region "$REGION" \
    --gpu-type "$H200_GPU_TYPE" \
    --gpu-preset "$H200_GPU_PRESET" \
    --gpu-count "$H200_GPU_COUNT" \
    --server-port "$GROOT_PORT" \
    --model "$GROOT_MODEL" \
    --robot-embodiment "$GROOT_EMBODIMENT" \
    --yes \
    2>&1 | tee "$LOG_DIR/groot-deploy.log"

  log "Deploying FiftyOne BYOVM on $host"
  "$NPA_BIN" workbench fiftyone -p "$PROJECT_ALIAS" -n "$FIFTYONE_ALIAS" deploy \
    --runtime byovm \
    --host "$host" \
    --ssh-user "$SSH_USER" \
    --ssh-key "$SSH_KEY" \
    --project-id "$PROJECT_ID" \
    --tenant-id "$TENANT_ID" \
    --region "$REGION" \
    --gpu-count "$H200_GPU_COUNT" \
    --port "$FIFTYONE_PORT" \
    --yes \
    2>&1 | tee "$LOG_DIR/fiftyone-deploy.log"
}

run_pipeline() {
  local isaac_export="${TARGET_BUCKET_URI}/demo-prestage/isaac-lab/${ISAAC_TASK}/"
  local groot_dataset="${TARGET_BUCKET_URI}/demo-prestage/groot-lerobot/${ISAAC_TASK}/"
  local groot_output="${TARGET_BUCKET_URI}/demo-live/groot-predictions/stage-demo/"
  local cosmos_output="${TARGET_BUCKET_URI}/demo-live/cosmos/stage-demo.mp4"
  local fiftyone_source="${TARGET_BUCKET_URI}/demo-prestage/cosmos/fiftyone-ranked/"
  local fiftyone_dataset="demo_cosmos_ranked_${RUN_TAG}"

  log "Isaac Lab short train"
  "$NPA_BIN" workbench isaac-lab -p "$PROJECT_ALIAS" -n "$ISAAC_ALIAS" train \
    --task "$ISAAC_TASK" \
    --num-envs 64 \
    --steps 1000 \
    --output-path "${TARGET_BUCKET_URI}/isaac-lab-train/${ISAAC_TASK}/" \
    --output-format json \
    2>&1 | tee "$LOG_DIR/isaac-train.log" || log "WARN: isaac train returned non-zero"

  log "Isaac Lab export-lerobot"
  "$NPA_BIN" workbench isaac-lab -p "$PROJECT_ALIAS" -n "$ISAAC_ALIAS" export-lerobot \
    --task "$ISAAC_TASK" \
    --num-episodes 10 \
    --steps-per-episode 30 \
    --output-path "$isaac_export" \
    --output-format json \
    2>&1 | tee "$LOG_DIR/isaac-export.log"

  log "GR00T convert lerobot-to-groot"
  "$NPA_BIN" workbench groot -p "$PROJECT_ALIAS" -n "$GROOT_ALIAS" convert \
    --input-path "$isaac_export" \
    --output-path "$groot_dataset" \
    --direction lerobot-to-groot \
    --robot-embodiment "$GROOT_EMBODIMENT" \
    --output-format json \
    2>&1 | tee "$LOG_DIR/groot-convert.log"

  log "GR00T infer"
  "$NPA_BIN" workbench groot -p "$PROJECT_ALIAS" -n "$GROOT_ALIAS" infer \
    --input-path "$GROOT_MODEL" \
    --dataset-path "$groot_dataset" \
    --output-path "$groot_output" \
    --embodiment-tag "$GROOT_EMBODIMENT" \
    --steps 32 \
    --output json \
    2>&1 | tee "$LOG_DIR/groot-infer.log"

  log "Cosmos infer"
  "$NPA_BIN" workbench cosmos -p "$PROJECT_ALIAS" -n "$COSMOS_ALIAS" infer \
    --prompt "$COSMOS_PROMPT" \
    --output-path "$cosmos_output" \
    --output-format json \
    2>&1 | tee "$LOG_DIR/cosmos-infer.log" || \
  "$NPA_BIN" workbench cosmos -p "$PROJECT_ALIAS" -n "$COSMOS_ALIAS" infer \
    --prompt "$COSMOS_PROMPT" \
    --output-path "$cosmos_output" \
    2>&1 | tee -a "$LOG_DIR/cosmos-infer.log"

  # Seed a tiny FiftyOne-ranked prefix from the cosmos output when present.
  mkdir -p "$STATE_DIR/fiftyone-ranked"
  if command -v aws >/dev/null 2>&1; then
    local endpoint
    endpoint="$(grep -E '^[[:space:]]*endpoint_url:' "$HOME/.npa/credentials.yaml" | head -1 | awk '{print $2}')"
    aws --endpoint-url "${endpoint:-${AWS_ENDPOINT_URL:-https://storage.us-central1.nebius.cloud}}" \
      s3 cp "$cosmos_output" "s3://${TARGET_BUCKET}/demo-8gpu-h200/${RUN_TAG}/demo-prestage/cosmos/fiftyone-ranked/sample-000.mp4" \
      2>&1 | tee "$LOG_DIR/fiftyone-seed.log" || true
  fi

  log "FiftyOne load-dataset"
  "$NPA_BIN" workbench fiftyone -p "$PROJECT_ALIAS" -n "$FIFTYONE_ALIAS" load-dataset \
    --name "$fiftyone_dataset" \
    --input-path "${TARGET_BUCKET_URI}/demo-prestage/cosmos/fiftyone-ranked/" \
    --format auto \
    2>&1 | tee "$LOG_DIR/fiftyone-load.log" || log "WARN: fiftyone load returned non-zero"

  cat >"$STATE_DIR/artifacts.txt" <<EOF
TARGET_BUCKET_URI=$TARGET_BUCKET_URI
ISAAC_EXPORT=$isaac_export
GROOT_DATASET=$groot_dataset
GROOT_OUTPUT=$groot_output
COSMOS_OUTPUT=$cosmos_output
FIFTYONE_DATASET=$fiftyone_dataset
EOF
}

teardown_all() {
  log "Teardown requested"
  "$NPA_BIN" workbench fiftyone -p "$PROJECT_ALIAS" -n "$FIFTYONE_ALIAS" deploy --destroy --yes 2>&1 | tee "$LOG_DIR/fiftyone-destroy.log" || true
  "$NPA_BIN" workbench groot -p "$PROJECT_ALIAS" -n "$GROOT_ALIAS" deploy --destroy --yes 2>&1 | tee "$LOG_DIR/groot-destroy.log" || true
  "$NPA_BIN" workbench cosmos -p "$PROJECT_ALIAS" -n "$COSMOS_ALIAS" deploy --destroy --yes 2>&1 | tee "$LOG_DIR/cosmos-destroy.log" || true
  "$NPA_BIN" workbench isaac-lab -p "$PROJECT_ALIAS" -n "$ISAAC_ALIAS" deploy --destroy --yes 2>&1 | tee "$LOG_DIR/isaac-destroy.log" || true
}

main() {
  require_bin "$NPA_BIN"
  require_bin "$PYTHON_BIN"
  save_env
  log "Starting workbench demo RUN_TAG=$RUN_TAG project=$PROJECT_ALIAS ($PROJECT_ID)"
  log "Artifacts -> $TARGET_BUCKET_URI"
  log "State dir -> $STATE_DIR"

  if [[ "$TEARDOWN" == "1" ]]; then
    teardown_all
    exit 0
  fi

  if [[ "$SKIP_DEPLOY" != "1" ]]; then
    # Sequential infra bring-up avoids racing the tenant compute.disk.count quota.
    deploy_cosmos_h200 || fail "Cosmos H200 deploy failed (see $LOG_DIR/cosmos-deploy.log)"
    deploy_isaac_rtx || fail "Isaac deploy failed (see $LOG_DIR/isaac-deploy.log)"

    local host instance_id
    host="$(wait_for_host cosmos "$COSMOS_ALIAS" 30 || true)"
    instance_id="$("$PYTHON_BIN" - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path.home().joinpath(".npa/config.yaml").read_text())
alias="$COSMOS_ALIAS"
found=""
def walk(obj):
    global found
    if isinstance(obj, dict):
        if obj.get("alias")==alias:
            found = obj.get("instance_id") or found
        for k,v in obj.items():
            if k==alias and isinstance(v, dict):
                found = v.get("instance_id") or found
            walk(v)
    elif isinstance(obj, list):
        for i in obj: walk(i)
walk(cfg)
print(found or "")
PY
)"
    [[ -n "$host" ]] || fail "Could not resolve Cosmos VM host after deploy"
    [[ -n "$instance_id" ]] || log "WARN: instance_id not found in config; ingress may need manual VM id"
    echo "$host" >"$STATE_DIR/h200_host.txt"
    echo "$instance_id" >"$STATE_DIR/h200_instance_id.txt"
    if [[ -n "$instance_id" ]]; then
      deploy_shared_h200_services "$host" "$instance_id"
    else
      log "Skipping BYOVM groot/fiftyone until instance_id is known; host=$host"
    fi
  fi

  log "Service status snapshot"
  status_json cosmos "$COSMOS_ALIAS" | tee "$LOG_DIR/cosmos-status.out" || true
  status_json isaac-lab "$ISAAC_ALIAS" | tee "$LOG_DIR/isaac-status.out" || true
  status_json groot "$GROOT_ALIAS" | tee "$LOG_DIR/groot-status.out" || true
  status_json fiftyone "$FIFTYONE_ALIAS" | tee "$LOG_DIR/fiftyone-status.out" || true

  if [[ "$SKIP_PIPELINE" != "1" ]]; then
    run_pipeline
  fi

  log "Demo packaging complete. Review $STATE_DIR"
  cat "$STATE_DIR/demo.env"
}

main "$@"
