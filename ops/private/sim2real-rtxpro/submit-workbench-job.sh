#!/usr/bin/env bash
# Submit sim2real via npa workbench workflow submit + runbook.yaml.
# Materializes operator config into the runbook, then launches through SkyPilot.
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

ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
export NPA_SIM2REAL_REPO="${ROOT}"
NPA_BIN="${ROOT}/npa/.venv/bin/npa"
PY="${ROOT}/npa/.venv/bin/python"
RUNBOOK="${ROOT}/npa/workflows/workbench/sim2real/runbook.yaml"

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
if [ ! -f "${RUNBOOK}" ]; then
  echo "ERROR: runbook missing at ${RUNBOOK} — run ./setup.sh" >&2
  exit 1
fi
if [ ! -x "${NPA_BIN}" ]; then
  echo "ERROR: npa CLI missing at ${NPA_BIN} — run ./setup.sh" >&2
  exit 1
fi

customer_asset_prepare_for_submit
customer_asset_profile_apply "${SCRIPT_DIR}" "${BUCKET}" "${CUSTOMER_TASK_ID:-}" || exit 1
customer_asset_guard_placeholders || exit 1

if ! operator_use_workbench_submit; then
  echo "WARN: NPA_USE_KUBECTL_SUBMIT=1 — bypassing npa workbench workflow submit" >&2
  exec "${SCRIPT_DIR}/submit-k8s-staged-job.sh"
fi
if [ -n "${CUSTOMER_ASSET_PROFILE_APPLIED:-}" ]; then
  echo "Customer asset profile: ${CUSTOMER_ASSET_PROFILE_APPLIED} (${CUSTOMER_ASSET_PROFILE_PATH})"
  customer_asset_profile_print
fi

TRIGGER_URI="${NPA_SIM2REAL_TRIGGER_DATASET_URI:-${TRIGGER_DATASET_URI:-s3://${BUCKET}/sim2real-triggers/${RUN_ID}/lerobot-pusht/}}"
TRIGGER_ID="${NPA_SIM2REAL_TRIGGER_DATASET_ID:-${TRIGGER_DATASET_ID:-lerobot/pusht}}"
if [ -n "${TRIGGER_URI}" ] && [[ "${TRIGGER_URI}" != */ ]]; then
  TRIGGER_URI="${TRIGGER_URI}/"
fi

TRAINER_IMAGE="${TRAINER_IMAGE:-${REG}/npa-lerobot-vlm-rl:0.1.0}"
VLM_IMAGE="${VLM_IMAGE:-${REG}/npa-cosmos3-reason:3.0.1-genuine-sm120}"
EVAL_IMAGE="${EVAL_IMAGE:-${REG}/npa-sim2real-eval:0.1.1-genuine-sm120}"
AUGMENT_IMAGE="${AUGMENT_IMAGE:-${REG}/npa-cosmos2-transfer:2.5.0}"
POLICY_IMAGE="${POLICY_IMAGE:-${TRAINER_IMAGE}}"
ISAAC_IMAGE="${ISAAC_IMAGE:-${REG}/npa-isaac-lab:2.3.2.post1}"
lerobot_prod_defaults_apply

"${PY}" - \
  "${TRAINER_IMAGE}" "${TRAINER_IMAGE}" "${VLM_IMAGE}" "${EVAL_IMAGE}" \
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
    sys.exit(1)
print("Preflight OK: all workflow images are registry-qualified.")
PY

registry_refresh_for_images "${CTX}" \
  "${TRAINER_IMAGE}" "${VLM_IMAGE}" "${EVAL_IMAGE}" \
  "${AUGMENT_IMAGE}" "${POLICY_IMAGE}" "${ISAAC_IMAGE}"

LOG="/tmp/sim2real-cluster/${RUN_ID}.log"
mkdir -p /tmp/sim2real-cluster
MATERIALIZED="/tmp/sim2real-cluster/${RUN_ID}-runbook.yaml"
SKY_CONFIG="/tmp/sim2real-cluster/${RUN_ID}-skypilot-config.yaml"

S3_PREFIX="${S3_PREFIX:-sim2real-b}"
INNER_ITERATIONS="${INNER_ITERATIONS:-1}"
OUTER_ITERATIONS="${OUTER_ITERATIONS:-2}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
HELDOUT_ENV_COUNT="${HELDOUT_ENV_COUNT:-4}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.45}"
NPA_SOURCE_REPO="${NPA_SOURCE_REPO:-https://github.com/nebius/nebius-physical-ai.git}"
NPA_SOURCE_REF="${NPA_SOURCE_REF:-${SETUP_NPA_BRANCH:-feat/sim2real-mandatory-stages}}"

export RUN_ID TRIGGER_URI="${TRIGGER_URI}" TRIGGER_ID="${TRIGGER_ID}"
export S3_BUCKET="${BUCKET}" REGISTRY="${REG}"
export S3_PREFIX S3_ENDPOINT="${ENDPOINT}"
export TRAINER_IMAGE VLM_IMAGE EVAL_IMAGE AUGMENT_IMAGE POLICY_IMAGE ISAAC_IMAGE
export INNER_ITERATIONS OUTER_ITERATIONS ROLLOUT_COUNT HELDOUT_ENV_COUNT SUCCESS_THRESHOLD
export NPA_SOURCE_REPO NPA_SOURCE_REF
export ASSETS_URI="${ASSETS_URI:-}" SCENE_SPEC_URI="${SCENE_SPEC_URI:-}"
export NPA_SIM2REAL_ROBOT_PRESET="${NPA_SIM2REAL_ROBOT_PRESET:-${ROBOT_PRESET:-}}"
export NPA_SIM2REAL_ROBOT_SOURCE="${NPA_SIM2REAL_ROBOT_SOURCE:-${ROBOT_SOURCE:-}}"
export NPA_SIM2REAL_ROBOT_SPEC_URI="${NPA_SIM2REAL_ROBOT_SPEC_URI:-${ROBOT_SPEC_URI:-}}"

"${PY}" - "${RUNBOOK}" "${MATERIALIZED}" "${SKY_CONFIG}" <<'PY'
import os
import re
import sys
from pathlib import Path

import yaml

runbook_path, materialized_path, sky_config_path = map(Path, sys.argv[1:4])

def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

bucket = env("S3_BUCKET")
endpoint = env("S3_ENDPOINT", "https://storage.us-central1.nebius.cloud")
registry = env("REGISTRY", "").rstrip("/")
run_id = env("RUN_ID")
trigger_uri = env("TRIGGER_URI")
trigger_id = env("TRIGGER_ID")
trainer = env("TRAINER_IMAGE")
vlm = env("VLM_IMAGE")
eval_image = env("EVAL_IMAGE")
augment = env("AUGMENT_IMAGE")
policy = env("POLICY_IMAGE")
isaac = env("ISAAC_IMAGE")
prefix = env("S3_PREFIX", "sim2real-b")

docs = [doc for doc in yaml.safe_load_all(runbook_path.read_text(encoding="utf-8")) if doc]
if not docs:
    raise SystemExit(f"empty runbook: {runbook_path}")
task = docs[0]
task.setdefault("resources", {})
task["resources"]["image_id"] = f"docker:{trainer}"
envs = task.setdefault("envs", {})
updates = {
    "NPA_SIM2REAL_RUN_ID": run_id,
    "NPA_SIM2REAL_BUCKET": bucket,
    "NPA_SIM2REAL_PREFIX": prefix,
    "S3_BUCKET": bucket,
    "AWS_ENDPOINT_URL": endpoint,
    "S3_ENDPOINT_URL": endpoint,
    "NPA_SIM2REAL_TRIGGER_DATASET_URI": trigger_uri,
    "NPA_SIM2REAL_TRIGGER_DATASET_ID": trigger_id,
    "TRAINER_IMAGE": trainer,
    "VLM_IMAGE": vlm,
    "EVAL_IMAGE": eval_image,
    "AUGMENT_IMAGE": augment,
    "POLICY_IMAGE": policy,
    "ISAAC_IMAGE": isaac,
    "NPA_REGISTRY": registry,
    "INNER_ITERATIONS": env("INNER_ITERATIONS", "1"),
    "OUTER_ITERATIONS": env("OUTER_ITERATIONS", "2"),
    "LOOP_OF_LOOPS_ITERATIONS": env("LOOP_OF_LOOPS_ITERATIONS", "1"),
    "ROLLOUT_COUNT": env("ROLLOUT_COUNT", "8"),
    "STEPS_PER_ROLLOUT": env("STEPS_PER_ROLLOUT", "4"),
    "HELDOUT_ENV_COUNT": env("HELDOUT_ENV_COUNT", "4"),
    "NPA_SIM2REAL_HELDOUT_EVAL_LIMIT": env("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", env("HELDOUT_ENV_COUNT", "4")),
    "SUCCESS_THRESHOLD": env("SUCCESS_THRESHOLD", "0.45"),
    "VLM_MODEL": env("VLM_MODEL", "nvidia/Cosmos-Reason1-7B"),
    "SIGNAL_LOSS_WEIGHT": env("SIGNAL_LOSS_WEIGHT", "1.0"),
    "LEARNING_RATE": env("LEARNING_RATE", "0.05"),
    "NO_GUARDRAILS": env("NO_GUARDRAILS", "0"),
    "NPA_SIM2REAL_RERUN": env("NPA_SIM2REAL_RERUN", "1"),
    "NPA_SIM2REAL_SIM_BACKEND": env("NPA_SIM2REAL_SIM_BACKEND", "isaac"),
    "NPA_SIM2REAL_ISAAC_TASK": env("NPA_SIM2REAL_ISAAC_TASK", "Isaac-Lift-Cube-Franka-v0"),
    "NPA_SOURCE_REPO": env("NPA_SOURCE_REPO"),
    "NPA_SOURCE_REF": env("NPA_SOURCE_REF"),
    "ACTION_ROLLOUTS_URI": env("ACTION_ROLLOUTS_URI", ""),
    "TRAIN_ENVS_URI": env("TRAIN_ENVS_URI", ""),
    "HELDOUT_ENVS_URI": env("HELDOUT_ENVS_URI", ""),
    "NPA_SIM2REAL_K8S_NAMESPACE": env("NPA_SIM2REAL_K8S_NAMESPACE", "default"),
    "NPA_SIM2REAL_K8S_SERVICE_ACCOUNT": env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa"),
    "NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS": env(
        "NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS",
        "agent-sa,ngc-nvcr-imagepullsecret,npa-nebius-registry",
    ),
    "NPA_SIM2REAL_K8S_ENV_SECRET_NAMES": env(
        "NPA_SIM2REAL_K8S_ENV_SECRET_NAMES",
        "hf-ngc-tokens,npa-storage-credentials",
    ),
    "NPA_SIM2REAL_K8S_GPU_RESOURCE": env("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu"),
    "NPA_SIM2REAL_K8S_GPU_PRODUCT": env(
        "NPA_SIM2REAL_K8S_GPU_PRODUCT",
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    ),
    "NPA_SIM2REAL_K8S_JOB_TIMEOUT_S": env("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", "10800"),
    "BYO_TRAINER_COMMAND": env("BYO_TRAINER_COMMAND", ""),
    "BYO_SIGNAL_CONVERTER": env("BYO_SIGNAL_CONVERTER", ""),
    "BYO_VLM_COMMAND": env("BYO_VLM_COMMAND", ""),
    "BYO_EVAL_COMMAND": env("BYO_EVAL_COMMAND", ""),
    "BYO_RERUN_COMMAND": env("BYO_RERUN_COMMAND", ""),
}
for key in ("ASSETS_URI", "SCENE_SPEC_URI"):
    val = env(key, "")
    if val:
        updates[key] = val
for key in (
    "NPA_SIM2REAL_ROBOT_PRESET",
    "NPA_SIM2REAL_ROBOT_SOURCE",
    "NPA_SIM2REAL_ROBOT_SPEC_URI",
):
    val = env(key, env(key.removeprefix("NPA_SIM2REAL_"), ""))
    if val:
        updates[key] = val
envs.update({key: str(value) for key, value in updates.items() if value is not None})

def materialize_simple_placeholders(block: str) -> str:
    if not block:
        return block
    keys = sorted(updates, key=len, reverse=True)
    pattern = re.compile(r"\$\{(" + "|".join(re.escape(k) for k in keys) + r")\}")

    def repl(match: re.Match[str]) -> str:
        return str(updates.get(match.group(1), match.group(0)))

    return pattern.sub(repl, block)

for block_key in ("setup", "run"):
    if block_key in task and isinstance(task[block_key], str):
        task[block_key] = materialize_simple_placeholders(task[block_key])

materialized_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

bad_envs = [
    key
    for key, value in envs.items()
    if isinstance(value, str) and "${" in value
]
if bad_envs:
    raise SystemExit(
        "materialized runbook envs still have unresolved placeholders: "
        + ", ".join(sorted(bad_envs)[:12])
    )

sky_config = {
    "kubernetes": {
        "pod_config": {
            "spec": {
                "serviceAccountName": env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa"),
                "imagePullSecrets": [
                    {"name": "agent-sa"},
                    {"name": "ngc-nvcr-imagepullsecret"},
                    {"name": "npa-nebius-registry"},
                ],
                "envFrom": [
                    {"secretRef": {"name": "hf-ngc-tokens"}},
                    {"secretRef": {"name": "npa-storage-credentials"}},
                ],
            }
        }
    }
}
sky_config_path.write_text(yaml.safe_dump(sky_config, sort_keys=False), encoding="utf-8")
print(f"materialized_runbook={materialized_path}")
print(f"skypilot_config={sky_config_path}")
PY

operator_export_storage_env "${ROOT}" || true

export NPA_SIM2REAL_SUBMIT_SCRIPT="${SCRIPT_DIR}/submit-k8s-staged-job.sh"

if [[ "${SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  echo "SUBMIT_DRY_RUN=1 — materialized runbook only; no cluster submit"
  echo "run_id=${RUN_ID}"
  echo "job=${RUN_ID}"
  echo "manifest=${MATERIALIZED}"
  echo "log=${LOG}"
  echo "trigger_uri=${TRIGGER_URI} trigger_id=${TRIGGER_ID}"
  exit 0
fi

SUBMIT_CMD=(
  "${NPA_BIN}" workbench workflow submit
  "${RUNBOOK}"
  --run-id "${RUN_ID}"
  --s3-bucket "${BUCKET}"
  --s3-prefix "${S3_PREFIX}"
  --var "NPA_SIM2REAL_TRIGGER_DATASET_URI=${TRIGGER_URI}"
  --var "NPA_SIM2REAL_TRIGGER_DATASET_ID=${TRIGGER_ID}"
  --var "INNER_ITERATIONS=${INNER_ITERATIONS}"
  --var "OUTER_ITERATIONS=${OUTER_ITERATIONS}"
)

echo "=== Submit sim2real via npa workbench workflow submit ===" | tee "${LOG}"
echo "context=${CTX} bucket=${BUCKET} run_id=${RUN_ID}" | tee -a "${LOG}"
echo "runbook=${RUNBOOK}" | tee -a "${LOG}"
echo "materialized=${MATERIALIZED}" | tee -a "${LOG}"
printf 'command:' | tee -a "${LOG}"
printf ' %q' "${SUBMIT_CMD[@]}" | tee -a "${LOG}"
echo | tee -a "${LOG}"

if ! "${SUBMIT_CMD[@]}" 2>&1 | tee -a "${LOG}"; then
  echo "ERROR: npa workbench workflow submit failed (see ${LOG})" >&2
  exit 1
fi

if PARSED_RUN_ID="$(operator_parse_submit_run_id "${LOG}" 2>/dev/null)"; then
  RUN_ID="${PARSED_RUN_ID}"
else
  RUN_ID="$(operator_normalize_staged_run_id "${RUN_ID}")"
fi
JOB="$(operator_parse_submit_job "${LOG}" "${RUN_ID}" 2>/dev/null || true)"
JOB="${JOB:-$(operator_orchestrator_job_name "${RUN_ID}")}"

echo "run_id=${RUN_ID}"
echo "job=${JOB}"
echo "manifest=${MATERIALIZED}"
echo "log=${LOG}"
echo "trigger_uri=${TRIGGER_URI} trigger_id=${TRIGGER_ID}"
echo "sim_backend=${NPA_SIM2REAL_SIM_BACKEND:-isaac} isaac_image=${ISAAC_IMAGE} augment_image=${AUGMENT_IMAGE}"

MONITOR_SCRIPT="${SCRIPT_DIR}/status-run-npa.sh"
if [[ "${LAUNCH_MONITOR:-1}" == "1" ]] && command -v tmux >/dev/null; then
  MONITOR_SESSION="${MONITOR_TMUX_SESSION:-sim2real-cluster-live}"
  tmux kill-session -t "${MONITOR_SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${MONITOR_SESSION}" \
    "bash -lc 'exec \"${MONITOR_SCRIPT}\" \"${RUN_ID}\" --watch; code=\$?; echo monitor_exit=\$code; exec bash'"
  echo "MONITOR_TMUX=${MONITOR_SESSION} attach: tmux attach -t ${MONITOR_SESSION}"
elif [[ "${LAUNCH_MONITOR:-1}" == "1" ]]; then
  nohup bash -lc "exec \"${MONITOR_SCRIPT}\" \"${RUN_ID}\" --watch" \
    >"/tmp/sim2real-cluster/${RUN_ID}-monitor.nohup.log" 2>&1 &
  echo "MONITOR_PID=$! log=/tmp/sim2real-cluster/${RUN_ID}-monitor.nohup.log"
fi
echo "MONITOR_RUN_ID=${RUN_ID}"
