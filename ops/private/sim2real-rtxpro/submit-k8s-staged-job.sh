#!/usr/bin/env bash
# Direct Kubernetes submit for sim2real staged runbook.
# Bypasses SkyPilot kubeconfig mismatch (workbench vs cluster contexts).
# Credentials: cluster secretRef only — never embedded in generated manifests.
#
# Storage env (see runbook.yaml header for full map):
#   S3_BUCKET              — artifact + trigger parent bucket (fallback: storage.bucket)
#   AWS_ENDPOINT_URL       — S3-compatible endpoint (fallback: storage.endpoint_url)
#   NPA_SIM2REAL_TRIGGER_DATASET_URI — LeRobot trigger prefix (required at submit)
#   NPA_SIM2REAL_TRIGGER_DATASET_ID — source dataset id (default lerobot/pusht)
# Aliases: TRIGGER_DATASET_URI, TRIGGER_DATASET_ID. External object-store example: endpoint
# https://storage.googleapis.com + GCS HMAC keys in ~/.npa/credentials.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
# shellcheck source=lib/registry-pull-secret.sh
source "${SCRIPT_DIR}/lib/registry-pull-secret.sh"
# shellcheck source=lib/customer-asset-profile.sh
source "${SCRIPT_DIR}/lib/customer-asset-profile.sh"
# shellcheck source=lib/lerobot-byo-trainer.sh
source "${SCRIPT_DIR}/lib/lerobot-byo-trainer.sh"
# shellcheck source=lib/asset-profile-guard.sh
source "${SCRIPT_DIR}/lib/asset-profile-guard.sh"
# shellcheck source=lib/trigger-preflight.sh
source "${SCRIPT_DIR}/lib/trigger-preflight.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
export NPA_SIM2REAL_REPO="${ROOT}"

npa_read_lines _npa_cfg operator_read_config "${ROOT}"
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.us-central1.nebius.cloud}}"
REG="${REGISTRY:-${_npa_cfg[2]:-}}"
CTX="${KUBECONTEXT:-${_npa_cfg[3]:-}}"
export S3_BUCKET="${BUCKET}"
export NPA_SIM2REAL_BUCKET="${BUCKET}"
if [[ "${NPA_SIM2REAL_TRIGGER_DATASET_URI:-}" == *YOUR-BUCKET* ]] && [ -n "${BUCKET}" ]; then
  export NPA_SIM2REAL_TRIGGER_DATASET_URI="${NPA_SIM2REAL_TRIGGER_DATASET_URI/YOUR-BUCKET/${BUCKET}}"
fi
if [ -z "${CTX}" ]; then
  echo "Set k8s_context in ~/.npa/config.yaml (storage or projects.<alias>)" >&2
  exit 1
fi
export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${CTX}")}"
operator_export_kubeconfig "${CTX}" "${ROOT}" || exit 1
RUN_ID="${RUN_ID:-sim2real-staged-$(date -u +%Y%m%dT%H%M%Sz | tr '[:upper:]' '[:lower:]')}"
if [ -z "${BUCKET}" ]; then
  echo "Set S3_BUCKET or configure storage.bucket in ~/.npa/config.yaml" >&2
  exit 1
fi
if [ -z "${REG}" ]; then
  echo "Set REGISTRY or configure storage.registry in ~/.npa/config.yaml" >&2
  exit 1
fi

customer_asset_prepare_for_submit
customer_asset_profile_apply "${SCRIPT_DIR}" "${BUCKET}" "${CUSTOMER_TASK_ID:-}" || exit 1
customer_asset_guard_placeholders || exit 1
if [ -n "${CUSTOMER_ASSET_PROFILE_APPLIED:-}" ]; then
  echo "Customer asset profile: ${CUSTOMER_ASSET_PROFILE_APPLIED} (${CUSTOMER_ASSET_PROFILE_PATH})"
  customer_asset_profile_print
fi

STOCK_TRIGGER_URI="${_npa_cfg[4]:-}"
TRIGGER_URI="${NPA_SIM2REAL_TRIGGER_DATASET_URI:-${TRIGGER_DATASET_URI:-${STOCK_TRIGGER_URI:-s3://${BUCKET}/sim2real-triggers/${RUN_ID}/lerobot-pusht/}}}"
TRIGGER_ID="${NPA_SIM2REAL_TRIGGER_DATASET_ID:-${TRIGGER_DATASET_ID:-lerobot/pusht}}"
# Normalize trailing slash for S3 prefix semantics.
if [ -n "${TRIGGER_URI}" ] && [[ "${TRIGGER_URI}" != */ ]]; then
  TRIGGER_URI="${TRIGGER_URI}/"
fi

readarray -t _tags < <("${ROOT}/npa/.venv/bin/python" - <<'PY'
from npa.deploy.images import supported_tool_version
print(supported_tool_version("lerobot-vlm-rl"))
print(supported_tool_version("sim2real-eval"))
print(supported_tool_version("cosmos3-reason"))
print(supported_tool_version("cosmos2-transfer"))
PY
)
TRAINER_TAG="${_tags[0]}"
EVAL_TAG="${_tags[1]}"
VLM_TAG="${_tags[2]}"
AUGMENT_TAG="${_tags[3]}"

# Preflight: every image must be registry-qualified before we apply the Job.
TRAINER_IMAGE="${TRAINER_IMAGE:-${REG}/npa-lerobot-vlm-rl:${TRAINER_TAG}}"
VLM_IMAGE="${VLM_IMAGE:-${REG}/npa-cosmos3-reason:${VLM_TAG}}"
EVAL_IMAGE="${EVAL_IMAGE:-${REG}/npa-sim2real-eval:${EVAL_TAG}}"
AUGMENT_IMAGE="${AUGMENT_IMAGE:-${REG}/npa-cosmos2-transfer:${AUGMENT_TAG}}"
POLICY_IMAGE="${POLICY_IMAGE:-${TRAINER_IMAGE}}"
ISAAC_IMAGE="${ISAAC_IMAGE:-${REG}/npa-isaac-lab:2.3.2.post1}"
ORCHESTRATOR_IMAGE="${ORCHESTRATOR_IMAGE:-${TRAINER_IMAGE}}"
lerobot_prod_defaults_apply

BYO_TRAINER_COMMAND_B64=""
if [[ -n "${BYO_TRAINER_COMMAND:-}" ]]; then
  BYO_TRAINER_COMMAND_B64="$(printf '%s' "${BYO_TRAINER_COMMAND}" | base64 -w0 2>/dev/null || printf '%s' "${BYO_TRAINER_COMMAND}" | base64)"
fi
# BYO held-out eval (rolls the TRAINED checkpoint for a real success_rate) — same
# base64 passthrough as the trainer so colon/space-bearing commands survive YAML.
BYO_EVAL_COMMAND_B64=""
if [[ -n "${BYO_EVAL_COMMAND:-}" ]]; then
  BYO_EVAL_COMMAND_B64="$(printf '%s' "${BYO_EVAL_COMMAND}" | base64 -w0 2>/dev/null || printf '%s' "${BYO_EVAL_COMMAND}" | base64)"
fi

"${ROOT}/npa/.venv/bin/python" - \
  "${ORCHESTRATOR_IMAGE}" "${TRAINER_IMAGE}" "${VLM_IMAGE}" "${EVAL_IMAGE}" \
  "${AUGMENT_IMAGE}" "${POLICY_IMAGE}" "${ISAAC_IMAGE}" <<'PY'
import sys
from npa.guardrails.skypilot import unresolved_image_placeholders
from npa.workflows.sim2real_health import _looks_registry_qualified

labels = (
    "orchestrator",
    "trainer",
    "vlm",
    "eval",
    "augment",
    "policy",
    "isaac",
)
bad: list[str] = []
for label, image in zip(labels, sys.argv[1:], strict=True):
    if not image or unresolved_image_placeholders(image) or not _looks_registry_qualified(image):
        bad.append(f"{label}={image!r}")
if bad:
    print("Preflight failed: images must be registry-qualified (<registry>/<name>:<tag>).", file=sys.stderr)
    for item in bad:
        print(f"  {item}", file=sys.stderr)
    print("Set REGISTRY in ~/.npa/config.yaml or export fully-qualified TRAINER_IMAGE, VLM_IMAGE, etc.", file=sys.stderr)
    sys.exit(1)
print("Preflight OK: all workflow images are registry-qualified.")
PY

echo "=== Preflight: LeRobot trigger on S3 ==="
echo "  ${TRIGGER_URI}"
trigger_preflight_s3 "${TRIGGER_URI}" "${ENDPOINT}" "${ROOT}"

echo "=== Preflight: operator S3 write access ==="
storage_preflight_write "${BUCKET}" "${ENDPOINT}" "${ROOT}"

echo "=== Preflight: cluster npa-storage-credentials endpoint ==="
storage_preflight_cluster_secret "${CTX}" "${ENDPOINT}" "${ROOT}"

# Refresh npa-nebius-registry before apply — stale IAM tokens cause ImagePullBackOff 401.
registry_refresh_for_images "${CTX}" \
  "${ORCHESTRATOR_IMAGE}" "${TRAINER_IMAGE}" "${VLM_IMAGE}" "${EVAL_IMAGE}" \
  "${AUGMENT_IMAGE}" "${POLICY_IMAGE}" "${ISAAC_IMAGE}"

LOG="/tmp/sim2real-cluster/${RUN_ID}.log"
mkdir -p /tmp/sim2real-cluster

K8S_ENV_SECRETS="${K8S_ENV_SECRETS:-hf-ngc-tokens,npa-storage-credentials}"
# Build envFrom secretRef blocks — credentials stay in cluster secrets, never in this manifest.
ENV_FROM_YAML=""
IFS=',' read -ra _secret_names <<< "${K8S_ENV_SECRETS}"
for _secret in "${_secret_names[@]}"; do
  _secret="${_secret// /}"
  [ -z "${_secret}" ] && continue
  ENV_FROM_YAML="${ENV_FROM_YAML}
            - secretRef:
                name: ${_secret}"
done

JOB="sim2real-${RUN_ID}"
MANIFEST="/tmp/sim2real-cluster/${JOB}.yaml"

cat > "${MANIFEST}" <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB}
  namespace: default
  labels:
    app: sim2real-staged-loop
    run-id: ${RUN_ID}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: sim2real-staged-loop
        run-id: ${RUN_ID}
    spec:
      restartPolicy: Never
      serviceAccountName: agent-sa
      imagePullSecrets:
        - name: agent-sa
        - name: ngc-nvcr-imagepullsecret
        - name: npa-nebius-registry
      containers:
        - name: orchestrator
          image: ${ORCHESTRATOR_IMAGE}
          imagePullPolicy: Always
          resources:
            limits:
              nvidia.com/gpu: "1"
            requests:
              nvidia.com/gpu: "1"
          env:
            - name: NPA_SIM2REAL_RUN_ID
              value: "${RUN_ID}"
            - name: NPA_SIM2REAL_BUCKET
              value: "${BUCKET}"
            - name: NPA_SIM2REAL_PREFIX
              value: "sim2real-b"
            - name: AWS_ENDPOINT_URL
              value: "${ENDPOINT}"
            - name: S3_ENDPOINT_URL
              value: "${ENDPOINT}"
            - name: TRAINER_IMAGE
              value: "${TRAINER_IMAGE}"
            - name: VLM_IMAGE
              value: "${VLM_IMAGE}"
            - name: EVAL_IMAGE
              value: "${EVAL_IMAGE}"
            - name: NPA_REGISTRY
              value: "${REG}"
            - name: AUGMENT_IMAGE
              value: "${AUGMENT_IMAGE}"
            - name: POLICY_IMAGE
              value: "${POLICY_IMAGE}"
            - name: BYO_TRAINER_COMMAND_B64
              value: "${BYO_TRAINER_COMMAND_B64}"
            - name: BYO_EVAL_COMMAND_B64
              value: "${BYO_EVAL_COMMAND_B64}"
            - name: NPA_BYO_ISAAC_OBJECT_USD
              value: "${NPA_BYO_ISAAC_OBJECT_USD:-}"
            - name: NPA_BYO_ISAAC_OBJECT_SCALE
              value: "${NPA_BYO_ISAAC_OBJECT_SCALE:-}"
            - name: NPA_BYO_ISAAC_SUCCESS_DIST_M
              value: "${NPA_BYO_ISAAC_SUCCESS_DIST_M:-0.05}"
            - name: ISAAC_IMAGE
              value: "${ISAAC_IMAGE}"
            - name: NPA_SIM2REAL_ISAAC_TASK
              value: "${NPA_SIM2REAL_ISAAC_TASK:-Isaac-Lift-Cube-Franka-v0}"
            - name: NPA_SIM2REAL_SIM_BACKEND
              value: "${NPA_SIM2REAL_SIM_BACKEND:-isaac}"
            - name: INNER_ITERATIONS
              value: "${INNER_ITERATIONS:-3}"
            - name: OUTER_ITERATIONS
              value: "${OUTER_ITERATIONS:-2}"
            - name: NPA_ENV_COUNT
              value: "${NPA_ENV_COUNT:-10000}"
            - name: NPA_TRAIN_FRACTION
              value: "${NPA_TRAIN_FRACTION:-0.8}"
            - name: NPA_ENVGEN_SHARD_COUNT
              value: "${NPA_ENVGEN_SHARD_COUNT:-16}"
            - name: NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS
              value: "${NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS:-16}"
            - name: ROLLOUT_COUNT
              value: "${ROLLOUT_COUNT:-8}"
            - name: VLM_REASON2_MODEL
              value: "${VLM_REASON2_MODEL:-nvidia/Cosmos-Reason2-8B}"
            - name: VLM_REASON3_MODEL
              value: "${VLM_REASON3_MODEL:-nvidia/Cosmos-Reason2-2B}"
            - name: NPA_SIM2REAL_VLM_DUAL_REASON
              value: "${NPA_SIM2REAL_VLM_DUAL_REASON:-1}"
            - name: HELDOUT_ENV_COUNT
              value: "${HELDOUT_ENV_COUNT:-8}"
            - name: NPA_SIM2REAL_HELDOUT_EVAL_LIMIT
              value: "${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-${HELDOUT_ENV_COUNT:-8}}"
            - name: LEARNING_RATE
              value: "${LEARNING_RATE:-0.08}"
            - name: INITIAL_QUALITY
              value: "${INITIAL_QUALITY:-0.42}"
            - name: STEPS_PER_ROLLOUT
              value: "${STEPS_PER_ROLLOUT:-6}"
            - name: NPA_SIM2REAL_COMPONENT_DOWNLOAD_RETRIES
              value: "${NPA_SIM2REAL_COMPONENT_DOWNLOAD_RETRIES:-24}"
            - name: NPA_SIM2REAL_HELDOUT_UPLOAD_GRACE_S
              value: "${NPA_SIM2REAL_HELDOUT_UPLOAD_GRACE_S:-20}"
            - name: SUCCESS_THRESHOLD
              value: "${SUCCESS_THRESHOLD:-0.50}"
            - name: NPA_SOURCE_REPO
              value: "${NPA_SOURCE_REPO:-https://github.com/nebius/nebius-physical-ai.git}"
            - name: NPA_SOURCE_REF
              value: "${NPA_SOURCE_REF:-main}"
            - name: NPA_SIM2REAL_K8S_NAMESPACE
              value: "default"
            - name: NPA_SIM2REAL_K8S_SERVICE_ACCOUNT
              value: "agent-sa"
            - name: NPA_SIM2REAL_TRIGGER_DATASET_URI
              value: "${TRIGGER_URI}"
            - name: NPA_SIM2REAL_TRIGGER_DATASET_ID
              value: "${TRIGGER_ID}"
            - name: ASSETS_URI
              value: "${ASSETS_URI:-}"
            - name: SCENE_SPEC_URI
              value: "${SCENE_SPEC_URI:-}"
            - name: NPA_SIM2REAL_ROBOT_PRESET
              value: "${NPA_SIM2REAL_ROBOT_PRESET:-${ROBOT_PRESET:-}}"
            - name: NPA_SIM2REAL_ROBOT_SOURCE
              value: "${NPA_SIM2REAL_ROBOT_SOURCE:-${ROBOT_SOURCE:-}}"
            - name: NPA_SIM2REAL_ROBOT_SPEC_URI
              value: "${NPA_SIM2REAL_ROBOT_SPEC_URI:-${ROBOT_SPEC_URI:-}}"
            - name: NPA_SIM2REAL_K8S_GPU_PRODUCT
              value: "${NPA_SIM2REAL_K8S_GPU_PRODUCT:-NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition}"
            - name: NPA_SIM2REAL_K8S_JOB_TIMEOUT_S
              value: "${NPA_SIM2REAL_K8S_JOB_TIMEOUT_S:-28800}"
          envFrom:${ENV_FROM_YAML}
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              exec > >(tee -a /tmp/run.log) 2>&1
              git clone --depth 1 --branch "\${NPA_SOURCE_REF}" "\${NPA_SOURCE_REPO}" /tmp/npa-src
              export PYTHONPATH="/tmp/npa-src/npa/src:\${PYTHONPATH:-}"
              python3 -c "import npa.workflows.sim2real as m; print(m.__file__)"
              if [[ -n "\${BYO_TRAINER_COMMAND_B64:-}" ]]; then
                export BYO_TRAINER_COMMAND="\$(printf '%s' "\${BYO_TRAINER_COMMAND_B64}" | base64 -d)"
              fi
              if [[ -n "\${BYO_EVAL_COMMAND_B64:-}" ]]; then
                export BYO_EVAL_COMMAND="\$(printf '%s' "\${BYO_EVAL_COMMAND_B64}" | base64 -d)"
              fi
              if ! command -v kubectl >/dev/null; then
                curl -fsSL -o /tmp/kubectl https://dl.k8s.io/release/v1.33.7/bin/linux/amd64/kubectl
                chmod +x /tmp/kubectl
              fi
              export PATH="/tmp:/usr/local/bin:\${PATH}"
              run_id="\${NPA_SIM2REAL_RUN_ID}"
              output_dir="/tmp/npa-sim2real-\${run_id}"
              mkdir -p "\${output_dir}"
              common_args=(
                --run-id "\${run_id}"
                --output-dir "\${output_dir}"
                --s3-bucket "\${NPA_SIM2REAL_BUCKET}"
                --s3-prefix "\${NPA_SIM2REAL_PREFIX:-sim2real-b}"
                --s3-endpoint "\${AWS_ENDPOINT_URL}"
                --inner-iterations "\${INNER_ITERATIONS:-3}"
                --outer-iterations "\${OUTER_ITERATIONS:-2}"
                --rollout-count "\${ROLLOUT_COUNT:-8}"
                --steps-per-rollout "\${STEPS_PER_ROLLOUT:-6}"
                --vlm-reason2-model "\${VLM_REASON2_MODEL:-nvidia/Cosmos-Reason2-8B}"
                --vlm-reason3-model "\${VLM_REASON3_MODEL:-nvidia/Cosmos-Reason2-2B}"
                --vlm-dual-reason
                --heldout-env-count "\${HELDOUT_ENV_COUNT:-8}"
                --heldout-eval-limit "\${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-\${HELDOUT_ENV_COUNT:-8}}"
                --threshold "\${SUCCESS_THRESHOLD:-0.50}"
                --learning-rate "\${LEARNING_RATE:-0.08}"
                --sim-backend "\${NPA_SIM2REAL_SIM_BACKEND:-isaac}"
                --isaac-image "\${ISAAC_IMAGE}"
                --isaac-task "\${NPA_SIM2REAL_ISAAC_TASK:-Isaac-Lift-Cube-Franka-v0}"
                --augment-image "\${AUGMENT_IMAGE}"
                --policy-image "\${POLICY_IMAGE}"
                --byo-trainer-command "\${BYO_TRAINER_COMMAND:-}"
                --byo-eval-command "\${BYO_EVAL_COMMAND:-}"
                --env-count "\${NPA_ENV_COUNT:-10000}"
                --train-fraction "\${NPA_TRAIN_FRACTION:-0.8}"
                --envgen-shard-count "\${NPA_ENVGEN_SHARD_COUNT:-16}"
                --k8s-max-parallel-gpus "\${NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS:-16}"
                --vlm-image "\${VLM_IMAGE}"
                --eval-image "\${EVAL_IMAGE}"
                --trainer-image "\${TRAINER_IMAGE}"
                --k8s-namespace "\${NPA_SIM2REAL_K8S_NAMESPACE:-default}"
                --k8s-service-account "\${NPA_SIM2REAL_K8S_SERVICE_ACCOUNT:-agent-sa}"
                --trigger-dataset-uri "\${NPA_SIM2REAL_TRIGGER_DATASET_URI}"
                --trigger-dataset-id "\${NPA_SIM2REAL_TRIGGER_DATASET_ID:-lerobot/pusht}"
                --assets-uri "\${ASSETS_URI:-}"
                --scene-spec-uri "\${SCENE_SPEC_URI:-}"
                --robot-preset "\${NPA_SIM2REAL_ROBOT_PRESET:-}"
                --robot-source "\${NPA_SIM2REAL_ROBOT_SOURCE:-}"
                --robot-spec-uri "\${NPA_SIM2REAL_ROBOT_SPEC_URI:-}"
                --k8s-gpu-product "\${NPA_SIM2REAL_K8S_GPU_PRODUCT:-NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition}"
                --k8s-job-timeout-s "\${NPA_SIM2REAL_K8S_JOB_TIMEOUT_S:-10800}"
                --source-repo "\${NPA_SOURCE_REPO:-https://github.com/nebius/nebius-physical-ai.git}"
                --source-ref "\${NPA_SOURCE_REF:-main}"
                --upload-artifacts
              )
              python3 -m npa.workflows.sim2real run "\${common_args[@]}" \
                --initial-quality "\${INITIAL_QUALITY:-0.42}"
              python3 -c "import json; from pathlib import Path; r=json.loads(Path('\${output_dir}/reports/sim2real-report.json').read_text()); print('CLUSTER_METRICS', json.dumps({'run_id': r['run_id'], 'decision': r['outer_loop']['latest_decision'], 'reward_trend': r['inner_loop']['reward_trend']}))"
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition
YAML

chmod 600 "${MANIFEST}"

echo "Applying job ${JOB} to context ${CTX}..." | tee "${LOG}"
kubectl --context "${CTX}" apply -f "${MANIFEST}" | tee -a "${LOG}"
echo "run_id=${RUN_ID}"
echo "job=${JOB}"
echo "manifest=${MANIFEST}"
echo "log=${LOG}"
echo "trigger_uri=${TRIGGER_URI} trigger_id=${TRIGGER_ID}"
echo "sim_backend=${NPA_SIM2REAL_SIM_BACKEND:-isaac} env_count=${NPA_ENV_COUNT:-10000} isaac_image=${ISAAC_IMAGE} augment_image=${AUGMENT_IMAGE}"

MONITOR_SCRIPT="$(cd "$(dirname "$0")" && pwd)/monitor-k8s-job.sh"
MONITOR_SESSION="${MONITOR_TMUX_SESSION:-sim2real-cluster-live}"
if [[ "${LAUNCH_MONITOR:-1}" == "1" ]] && command -v tmux >/dev/null; then
  tmux kill-session -t "${MONITOR_SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${MONITOR_SESSION}" \
    "bash -lc 'exec \"${MONITOR_SCRIPT}\" \"${JOB}\"; code=\$?; echo monitor_exit=\$code; exec bash'"
  echo "MONITOR_TMUX=${MONITOR_SESSION} attach: tmux attach -t ${MONITOR_SESSION}"
elif [[ "${LAUNCH_MONITOR:-1}" == "1" ]]; then
  nohup bash -lc "exec \"${MONITOR_SCRIPT}\" \"${JOB}\"" \
    >"/tmp/sim2real-cluster/${JOB}-monitor.nohup.log" 2>&1 &
  echo "MONITOR_PID=$! log=/tmp/sim2real-cluster/${JOB}-monitor.nohup.log"
fi
echo "MONITOR_JOB=${JOB}"
