#!/usr/bin/env bash
# Detached Sim2Real live run launcher.
# Owns a single run end-to-end: preflight, source tarball staging, K8s Job,
# status polling, final success validation, and Rerun serve deployment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/npa/.venv/bin/python"
if [ ! -x "${PY}" ]; then
  echo "Missing repo venv python: ${PY}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-sim2real-real-success-$(date -u +%Y%m%dt%H%M%Sz)}"
RUN_ID="$(printf '%s' "${RUN_ID}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9.-' '-')"
RUN_ID="${RUN_ID%-}"
if [ "${#RUN_ID}" -gt 52 ]; then
  RUN_ID="${RUN_ID:0:52}"
  RUN_ID="${RUN_ID%-}"
fi

OUT_ROOT="${OUT_ROOT:-/tmp/sim2real-real-success}"
OUT="${OUT_ROOT}/${RUN_ID}"
mkdir -p "${OUT}"
ln -sfn "${OUT}" "${OUT_ROOT}/latest"
LOG="${OUT}/launcher.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== sim2real real success launcher ==="
echo "run_id=${RUN_ID}"
echo "worktree=${ROOT}"
echo "log=${LOG}"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

export PYTHONPATH="${ROOT}/npa/src:${PYTHONPATH:-}"
export RUN_ID

for env_file in "${HOME}/.npa/sim2real-operator.env" "${HOME}/.npa/live-e2e.env"; do
  if [ -f "${env_file}" ]; then
    echo "sourcing ${env_file}"
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
done
# Some local env files can carry a short-lived NEBIUS_IAM_TOKEN. A stale value
# poisons provider API calls, so let the CLI use its durable configured profile.
unset NEBIUS_IAM_TOKEN

CONFIG_JSON="${OUT}/launcher-config.json"
"${PY}" - "${ROOT}" "${RUN_ID}" "${CONFIG_JSON}" <<'PY'
import json
import os
import tarfile
import tempfile
from pathlib import Path

from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient
from npa.deploy.images import supported_tool_version
from npa.workflows.sim2real.monitor import load_operator_config, resolve_kubeconfig

root = Path(os.sys.argv[1]).resolve()
run_id = os.sys.argv[2]
config_path = Path(os.sys.argv[3])
operator = load_operator_config()
creds = load_credentials()

bucket = operator.bucket
endpoint = operator.endpoint_url
registry = operator.registry.rstrip("/")
context = operator.k8s_context
kubeconfig = str(resolve_kubeconfig(context))

if not registry:
    raise SystemExit("storage.registry is not set in ~/.npa/config.yaml")

trigger_uri = (
    os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
    or os.environ.get("TRIGGER_DATASET_URI")
    or f"s3://{bucket}/sim2real-triggers/trigger-validate-20260611T154016Z/lerobot-pusht/"
)
if trigger_uri and not trigger_uri.endswith("/"):
    trigger_uri += "/"
trigger_id = (
    os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
    or os.environ.get("TRIGGER_DATASET_ID")
    or "lerobot/pusht"
)

tags = {
    "trainer": supported_tool_version("lerobot-vlm-rl"),
    "eval": supported_tool_version("loop-eval"),
    "vlm": supported_tool_version("cosmos3-reason"),
    "augment": supported_tool_version("cosmos2-transfer"),
    "isaac": supported_tool_version("isaac-lab"),
    "rerun": supported_tool_version("rerun-viewer"),
}
images = {
    "TRAINER_IMAGE": os.environ.get("TRAINER_IMAGE")
    or f"{registry}/npa-lerobot-vlm-rl:{tags['trainer']}",
    "EVAL_IMAGE": os.environ.get("EVAL_IMAGE")
    or f"{registry}/npa-loop-eval:{tags['eval']}",
    "VLM_IMAGE": os.environ.get("VLM_IMAGE")
    or f"{registry}/npa-cosmos3-reason:{tags['vlm']}",
    "AUGMENT_IMAGE": os.environ.get("AUGMENT_IMAGE")
    or f"{registry}/npa-cosmos2-transfer:{tags['augment']}",
    "ISAAC_IMAGE": os.environ.get("ISAAC_IMAGE")
    or f"{registry}/npa-isaac-lab:{tags['isaac']}",
    "RERUN_IMAGE": os.environ.get("RERUN_IMAGE")
    or os.environ.get("NPA_RERUN_VIEWER_IMAGE")
    or f"{registry}/npa-rerun-viewer:{tags['rerun']}",
}
images["POLICY_IMAGE"] = os.environ.get("POLICY_IMAGE") or images["TRAINER_IMAGE"]
images["ORCHESTRATOR_IMAGE"] = os.environ.get("ORCHESTRATOR_IMAGE") or images["TRAINER_IMAGE"]

access_key = os.environ.get("AWS_ACCESS_KEY_ID") or creds.s3_access_key_id
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or creds.s3_secret_access_key
if not access_key or not secret_key:
    raise SystemExit("S3 credentials missing; configure ~/.npa/credentials.yaml or AWS_*")

def tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = tarinfo.name
    if "__pycache__" in name or name.endswith(".pyc"):
        return None
    return tarinfo

with tempfile.TemporaryDirectory(prefix="npa-orchestrator-src-") as tmp:
    tarball = Path(tmp) / "npa-source.tgz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(root / "npa" / "src", arcname="npa/src", filter=tar_filter)
        archive.add(root / "npa" / "pyproject.toml", arcname="npa/pyproject.toml", filter=tar_filter)
    destination = f"s3://{bucket}/sim2real-b/{run_id}/source/orchestrator-{run_id}.tgz"
    client = StorageClient(
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    source_uri = client.upload_file(str(tarball), destination)

payload = {
    "BUCKET": bucket,
    "ENDPOINT": endpoint,
    "REGISTRY": registry,
    "CTX": context,
    "KUBECONFIG_PATH": kubeconfig,
    "TRIGGER_URI": trigger_uri,
    "TRIGGER_ID": trigger_id,
    "SOURCE_TARBALL_URI": source_uri,
    **images,
}
config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(config_path)
PY

eval "$("${PY}" - "${CONFIG_JSON}" <<'PY'
import json
import shlex
import sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
for key, value in data.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

export AWS_ENDPOINT_URL="${ENDPOINT}"
export S3_ENDPOINT_URL="${ENDPOINT}"
export S3_BUCKET="${BUCKET}"
export NPA_SIM2REAL_BUCKET="${BUCKET}"
export KUBECONFIG="${KUBECONFIG_PATH}"
export BUCKET ENDPOINT REGISTRY CTX KUBECONFIG_PATH SOURCE_TARBALL_URI
export TRIGGER_URI TRIGGER_ID

echo "operator_context=${CTX}"
echo "kubeconfig=${KUBECONFIG_PATH}"
echo "bucket=${BUCKET}"
echo "endpoint=${ENDPOINT}"
echo "source_tarball=${SOURCE_TARBALL_URI}"
echo "trigger_uri=${TRIGGER_URI}"
echo "images:"
echo "  orchestrator=${ORCHESTRATOR_IMAGE}"
echo "  trainer=${TRAINER_IMAGE}"
echo "  policy=${POLICY_IMAGE}"
echo "  eval=${EVAL_IMAGE}"
echo "  isaac=${ISAAC_IMAGE}"
echo "  vlm=${VLM_IMAGE}"
echo "  augment=${AUGMENT_IMAGE}"
echo "  rerun=${RERUN_IMAGE}"

echo "=== preflight: sim2real health ==="
"${PY}" -m npa.cli.main workbench health sim2real \
  --run-id "${RUN_ID}" \
  --s3-bucket "${BUCKET}" \
  --s3-prefix sim2real-b \
  --s3-endpoint "${ENDPOINT}" \
  --trigger-dataset-uri "${TRIGGER_URI}" \
  --trigger-dataset-id "${TRIGGER_ID}" \
  --augment-image "${AUGMENT_IMAGE}" \
  --policy-image "${POLICY_IMAGE}" \
  --trainer-image "${TRAINER_IMAGE}" \
  --vlm-image "${VLM_IMAGE}" \
  --eval-image "${EVAL_IMAGE}" \
  --threshold "${SUCCESS_THRESHOLD:-0.50}" \
  --inner-iterations "${INNER_ITERATIONS:-1}" \
  --outer-iterations "${OUTER_ITERATIONS:-2}" \
  --rollout-count "${ROLLOUT_COUNT:-8}" \
  --steps-per-rollout "${STEPS_PER_ROLLOUT:-6}" \
  --heldout-env-count "${HELDOUT_ENV_COUNT:-4}" \
  --k8s-context "${CTX}" \
  --k8s-kubeconfig "${KUBECONFIG_PATH}" \
  --checks all \
  --warn-only \
  --json | tee "${OUT}/health.json"

echo "=== preflight: cluster quick view ==="
kubectl --context "${CTX}" -n default get nodes -o wide | tee "${OUT}/nodes.txt"
kubectl --context "${CTX}" -n default get secret npa-storage-credentials hf-ngc-tokens | tee "${OUT}/required-secrets.txt"

BYO_TRAINER_COMMAND="${BYO_TRAINER_COMMAND:-NPA_BYO_ISAAC_ITERATIONS=1000 NPA_BYO_ISAAC_NUM_ENVS=1024 python3 -m npa.workflows.sim2real.byo_isaac_trainer}"
BYO_POLICY_COMMAND="${BYO_POLICY_COMMAND:-python3 -m npa.workflows.sim2real.byo_isaac_policy_rollout}"
BYO_EVAL_COMMAND="${BYO_EVAL_COMMAND:-python3 -m npa.workflows.sim2real.byo_isaac_eval}"

BYO_TRAINER_COMMAND_B64="$(printf '%s' "${BYO_TRAINER_COMMAND}" | base64 -w0 2>/dev/null || printf '%s' "${BYO_TRAINER_COMMAND}" | base64)"
BYO_POLICY_COMMAND_B64="$(printf '%s' "${BYO_POLICY_COMMAND}" | base64 -w0 2>/dev/null || printf '%s' "${BYO_POLICY_COMMAND}" | base64)"
BYO_EVAL_COMMAND_B64="$(printf '%s' "${BYO_EVAL_COMMAND}" | base64 -w0 2>/dev/null || printf '%s' "${BYO_EVAL_COMMAND}" | base64)"

JOB="sim2real-${RUN_ID}"
MANIFEST="${OUT}/${JOB}.json"
export JOB MANIFEST SOURCE_TARBALL_URI
export TRAINER_IMAGE POLICY_IMAGE EVAL_IMAGE ISAAC_IMAGE VLM_IMAGE AUGMENT_IMAGE ORCHESTRATOR_IMAGE
export TRIGGER_URI TRIGGER_ID BYO_TRAINER_COMMAND_B64 BYO_POLICY_COMMAND_B64 BYO_EVAL_COMMAND_B64

"${PY}" - "${MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

env = os.environ
job = env["JOB"]

container_script = r'''set -euo pipefail
exec > >(tee -a /tmp/run.log) 2>&1
echo "orchestrator_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 -c "import boto3" 2>/dev/null || python3 -m pip install --quiet boto3 botocore
rm -rf /tmp/npa-source && mkdir -p /tmp/npa-source
python3 - "${NPA_SIM2REAL_SOURCE_TARBALL_URI}" <<'PYB'
import os
import sys
import tarfile
import urllib.parse

import boto3

uri = urllib.parse.urlparse(sys.argv[1])
endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL") or None
boto3.client("s3", endpoint_url=endpoint).download_file(
    uri.netloc,
    uri.path.lstrip("/"),
    "/tmp/npa-source.tgz",
)
with tarfile.open("/tmp/npa-source.tgz") as tar:
    tar.extractall("/tmp/npa-source")
PYB
export PYTHONPATH="/tmp/npa-source/npa/src:${PYTHONPATH:-}"
python3 -c "import npa.workflows.sim2real as m; print('npa_source=' + m.__file__)"

if [[ -n "${BYO_TRAINER_COMMAND_B64:-}" ]]; then
  export BYO_TRAINER_COMMAND="$(printf '%s' "${BYO_TRAINER_COMMAND_B64}" | base64 -d)"
fi
if [[ -n "${BYO_POLICY_COMMAND_B64:-}" ]]; then
  export BYO_POLICY_COMMAND="$(printf '%s' "${BYO_POLICY_COMMAND_B64}" | base64 -d)"
fi
if [[ -n "${BYO_EVAL_COMMAND_B64:-}" ]]; then
  export BYO_EVAL_COMMAND="$(printf '%s' "${BYO_EVAL_COMMAND_B64}" | base64 -d)"
fi

if ! command -v kubectl >/dev/null; then
  curl -fsSL -o /tmp/kubectl https://dl.k8s.io/release/v1.33.7/bin/linux/amd64/kubectl
  chmod +x /tmp/kubectl
fi
export PATH="/tmp:/usr/local/bin:${PATH}"

run_id="${NPA_SIM2REAL_RUN_ID}"
output_dir="/tmp/npa-sim2real-${run_id}"
mkdir -p "${output_dir}"

common_args=(
  --run-id "${run_id}"
  --output-dir "${output_dir}"
  --s3-bucket "${NPA_SIM2REAL_BUCKET}"
  --s3-prefix "${NPA_SIM2REAL_PREFIX:-sim2real-b}"
  --s3-endpoint "${AWS_ENDPOINT_URL}"
  --trigger-dataset-uri "${NPA_SIM2REAL_TRIGGER_DATASET_URI}"
  --trigger-dataset-id "${NPA_SIM2REAL_TRIGGER_DATASET_ID:-lerobot/pusht}"
  --augment-image "${AUGMENT_IMAGE}"
  --policy-image "${POLICY_IMAGE}"
  --trainer-image "${TRAINER_IMAGE}"
  --vlm-image "${VLM_IMAGE}"
  --eval-image "${EVAL_IMAGE}"
  --isaac-image "${ISAAC_IMAGE}"
  --sim-backend "${NPA_SIM2REAL_SIM_BACKEND:-isaac}"
  --isaac-task "${NPA_SIM2REAL_ISAAC_TASK:-Isaac-Lift-Cube-Franka-v0}"
  --threshold "${SUCCESS_THRESHOLD:-0.50}"
  --inner-iterations "${INNER_ITERATIONS:-1}"
  --outer-iterations "${OUTER_ITERATIONS:-2}"
  --rollout-count "${ROLLOUT_COUNT:-8}"
  --steps-per-rollout "${STEPS_PER_ROLLOUT:-6}"
  --heldout-env-count "${HELDOUT_ENV_COUNT:-4}"
  --heldout-eval-limit "${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-4}"
  --env-count "${NPA_ENV_COUNT:-10000}"
  --train-fraction "${NPA_TRAIN_FRACTION:-0.8}"
  --envgen-shard-count "${NPA_ENVGEN_SHARD_COUNT:-16}"
  --k8s-max-parallel-gpus "${NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS:-16}"
  --k8s-namespace "${NPA_SIM2REAL_K8S_NAMESPACE:-default}"
  --k8s-service-account "${NPA_SIM2REAL_K8S_SERVICE_ACCOUNT:-agent-sa}"
  --k8s-image-pull-secrets "${NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS:-ngc-nvcr-imagepullsecret}"
  --k8s-env-secret-names "${NPA_SIM2REAL_K8S_ENV_SECRET_NAMES:-hf-ngc-tokens,npa-storage-credentials}"
  --k8s-gpu-resource "${NPA_SIM2REAL_K8S_GPU_RESOURCE:-nvidia.com/gpu}"
  --k8s-gpu-product "${NPA_SIM2REAL_K8S_GPU_PRODUCT:-NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition}"
  --k8s-job-timeout-s "${NPA_SIM2REAL_K8S_JOB_TIMEOUT_S:-28800}"
  --learning-rate "${LEARNING_RATE:-0.08}"
  --byo-trainer-command "${BYO_TRAINER_COMMAND:-}"
  --byo-policy-command "${BYO_POLICY_COMMAND:-}"
  --byo-eval-command "${BYO_EVAL_COMMAND:-}"
  --source-repo "local-source-tarball"
  --source-ref "${NPA_SIM2REAL_RUN_ID}"
  --vlm-reason2-model "${VLM_REASON2_MODEL:-nvidia/Cosmos-Reason2-8B}"
  --vlm-reason3-model "${VLM_REASON3_MODEL:-nvidia/Cosmos-Reason2-2B}"
  --vlm-dual-reason
  --rerun
  --upload-artifacts
)

python3 -m npa.workflows.sim2real run "${common_args[@]}" \
  --initial-quality "${INITIAL_QUALITY:-0.42}"

python3 - "${output_dir}/reports/sim2real-report.json" <<'PYC'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
decision = (report.get("outer_loop") or {}).get("latest_decision") or {}
heldout = (report.get("outer_loop") or {}).get("latest_heldout_report") or {}
print("CLUSTER_METRICS", json.dumps({
    "run_id": report.get("run_id"),
    "status": report.get("status"),
    "decision": decision.get("decision"),
    "success_rate": decision.get("success_rate", heldout.get("success_rate")),
    "threshold": decision.get("threshold", heldout.get("threshold")),
    "checkpoint_uri": decision.get("checkpoint_uri"),
}, sort_keys=True))
PYC
'''

def env_entry(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": str(value)}

run_env = [
    env_entry("NPA_SIM2REAL_RUN_ID", env["RUN_ID"]),
    env_entry("NPA_SIM2REAL_BUCKET", env["BUCKET"]),
    env_entry("NPA_SIM2REAL_PREFIX", "sim2real-b"),
    env_entry("AWS_ENDPOINT_URL", env["ENDPOINT"]),
    env_entry("S3_ENDPOINT_URL", env["ENDPOINT"]),
    env_entry("NPA_SIM2REAL_SOURCE_TARBALL_URI", env["SOURCE_TARBALL_URI"]),
    env_entry("TRAINER_IMAGE", env["TRAINER_IMAGE"]),
    env_entry("POLICY_IMAGE", env["POLICY_IMAGE"]),
    env_entry("EVAL_IMAGE", env["EVAL_IMAGE"]),
    env_entry("VLM_IMAGE", env["VLM_IMAGE"]),
    env_entry("AUGMENT_IMAGE", env["AUGMENT_IMAGE"]),
    env_entry("ISAAC_IMAGE", env["ISAAC_IMAGE"]),
    env_entry("NPA_SIM2REAL_ISAAC_IMAGE", env["ISAAC_IMAGE"]),
    env_entry("NPA_REGISTRY", env["REGISTRY"]),
    env_entry("BYO_TRAINER_COMMAND_B64", env["BYO_TRAINER_COMMAND_B64"]),
    env_entry("BYO_POLICY_COMMAND_B64", env["BYO_POLICY_COMMAND_B64"]),
    env_entry("BYO_EVAL_COMMAND_B64", env["BYO_EVAL_COMMAND_B64"]),
    env_entry("NPA_SIM2REAL_ISAAC_TASK", env.get("NPA_SIM2REAL_ISAAC_TASK", "Isaac-Lift-Cube-Franka-v0")),
    env_entry("NPA_SIM2REAL_SIM_BACKEND", env.get("NPA_SIM2REAL_SIM_BACKEND", "isaac")),
    env_entry("INNER_ITERATIONS", env.get("INNER_ITERATIONS", "1")),
    env_entry("OUTER_ITERATIONS", env.get("OUTER_ITERATIONS", "2")),
    env_entry("NPA_ENV_COUNT", env.get("NPA_ENV_COUNT", "10000")),
    env_entry("NPA_TRAIN_FRACTION", env.get("NPA_TRAIN_FRACTION", "0.8")),
    env_entry("NPA_ENVGEN_SHARD_COUNT", env.get("NPA_ENVGEN_SHARD_COUNT", "16")),
    env_entry("NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS", env.get("NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS", "16")),
    env_entry("ROLLOUT_COUNT", env.get("ROLLOUT_COUNT", "8")),
    env_entry("STEPS_PER_ROLLOUT", env.get("STEPS_PER_ROLLOUT", "6")),
    env_entry("HELDOUT_ENV_COUNT", env.get("HELDOUT_ENV_COUNT", "4")),
    env_entry("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", env.get("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", "4")),
    env_entry("LEARNING_RATE", env.get("LEARNING_RATE", "0.08")),
    env_entry("INITIAL_QUALITY", env.get("INITIAL_QUALITY", "0.42")),
    env_entry("SUCCESS_THRESHOLD", env.get("SUCCESS_THRESHOLD", "0.50")),
    env_entry("VLM_REASON2_MODEL", env.get("VLM_REASON2_MODEL", "nvidia/Cosmos-Reason2-8B")),
    env_entry("VLM_REASON3_MODEL", env.get("VLM_REASON3_MODEL", "nvidia/Cosmos-Reason2-2B")),
    env_entry("NPA_SIM2REAL_VLM_DUAL_REASON", env.get("NPA_SIM2REAL_VLM_DUAL_REASON", "1")),
    env_entry("NPA_SIM2REAL_RERUN", "1"),
    env_entry("NPA_SIM2REAL_RERUN_SERVE", "0"),
    env_entry("NPA_SIM2REAL_COMPONENT_DOWNLOAD_RETRIES", env.get("NPA_SIM2REAL_COMPONENT_DOWNLOAD_RETRIES", "24")),
    env_entry("NPA_SIM2REAL_HELDOUT_UPLOAD_GRACE_S", env.get("NPA_SIM2REAL_HELDOUT_UPLOAD_GRACE_S", "20")),
    env_entry("NPA_BYO_ISAAC_JOB_TIMEOUT_S", env.get("NPA_BYO_ISAAC_JOB_TIMEOUT_S", "7200")),
    env_entry("NPA_BYO_ISAAC_SUCCESS_DIST_M", env.get("NPA_BYO_ISAAC_SUCCESS_DIST_M", "0.10")),
    env_entry("NPA_SIM2REAL_K8S_NAMESPACE", "default"),
    env_entry("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa"),
    env_entry("NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS", "ngc-nvcr-imagepullsecret"),
    env_entry("NPA_SIM2REAL_K8S_ENV_SECRET_NAMES", "hf-ngc-tokens,npa-storage-credentials"),
    env_entry("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu"),
    env_entry("NPA_SIM2REAL_K8S_GPU_PRODUCT", env.get("NPA_SIM2REAL_K8S_GPU_PRODUCT", "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition")),
    env_entry("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", env.get("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", "28800")),
    env_entry("NPA_SIM2REAL_TRIGGER_DATASET_URI", env["TRIGGER_URI"]),
    env_entry("NPA_SIM2REAL_TRIGGER_DATASET_ID", env["TRIGGER_ID"]),
    env_entry("ASSETS_URI", env.get("ASSETS_URI", "")),
    env_entry("SCENE_SPEC_URI", env.get("SCENE_SPEC_URI", "")),
    env_entry("NPA_SIM2REAL_ROBOT_PRESET", env.get("NPA_SIM2REAL_ROBOT_PRESET", env.get("ROBOT_PRESET", ""))),
    env_entry("NPA_SIM2REAL_ROBOT_SOURCE", env.get("NPA_SIM2REAL_ROBOT_SOURCE", env.get("ROBOT_SOURCE", ""))),
    env_entry("NPA_SIM2REAL_ROBOT_SPEC_URI", env.get("NPA_SIM2REAL_ROBOT_SPEC_URI", env.get("ROBOT_SPEC_URI", ""))),
]

manifest = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {
        "name": job,
        "namespace": "default",
        "labels": {"app": "sim2real-staged-loop", "run-id": env["RUN_ID"]},
    },
    "spec": {
        "backoffLimit": 0,
        "ttlSecondsAfterFinished": 86400,
        "activeDeadlineSeconds": 28800,
        "template": {
            "metadata": {
                "labels": {"app": "sim2real-staged-loop", "run-id": env["RUN_ID"]},
            },
            "spec": {
                "restartPolicy": "Never",
                "serviceAccountName": "agent-sa",
                "imagePullSecrets": [
                    {"name": "ngc-nvcr-imagepullsecret"},
                ],
                "containers": [
                    {
                        "name": "orchestrator",
                        "image": env["ORCHESTRATOR_IMAGE"],
                        "imagePullPolicy": "Always",
                        "resources": {
                            "limits": {"nvidia.com/gpu": "1"},
                            "requests": {"nvidia.com/gpu": "1"},
                        },
                        "env": run_env,
                        "envFrom": [
                            {"secretRef": {"name": "hf-ngc-tokens"}},
                            {"secretRef": {"name": "npa-storage-credentials"}},
                        ],
                        "command": ["/bin/bash", "-lc"],
                        "args": [container_script],
                    }
                ],
                "nodeSelector": {
                    "nvidia.com/gpu.product": env.get(
                        "NPA_SIM2REAL_K8S_GPU_PRODUCT",
                        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
                    )
                },
            },
        },
    },
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(sys.argv[1])
PY

chmod 600 "${MANIFEST}"
echo "=== apply orchestrator job ==="
kubectl --context "${CTX}" -n default apply -f "${MANIFEST}" | tee "${OUT}/kubectl-apply.txt"
echo "job=${JOB}"
echo "manifest=${MANIFEST}"
echo "run_prefix=s3://${BUCKET}/sim2real-b/${RUN_ID}/"

POD=""
for _ in $(seq 1 90); do
  POD="$(kubectl --context "${CTX}" -n default get pods -l "job-name=${JOB}" --sort-by=.metadata.creationTimestamp -o name 2>/dev/null | tail -1 | sed 's#^pod/##')"
  if [ -n "${POD}" ]; then
    break
  fi
  sleep 2
done
if [ -n "${POD}" ]; then
  echo "orchestrator_pod=${POD}"
  (kubectl --context "${CTX}" -n default logs -f "${POD}" > "${OUT}/orchestrator.log" 2>&1 & echo $! > "${OUT}/orchestrator-logs.pid") || true
else
  echo "orchestrator_pod=pending"
fi

echo "=== monitor job ==="
DEADLINE_SECONDS="${DEADLINE_SECONDS:-28800}"
start_epoch="$(date +%s)"
while true; do
  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  succeeded="$(kubectl --context "${CTX}" -n default get job "${JOB}" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$(kubectl --context "${CTX}" -n default get job "${JOB}" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$(kubectl --context "${CTX}" -n default get job "${JOB}" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  echo "job_status elapsed=${elapsed}s active=${active:-0} succeeded=${succeeded:-0} failed=${failed:-0}"
  "${PY}" -m npa.cli.main workbench sim2real status \
    --run-id "${RUN_ID}" \
    --s3-bucket "${BUCKET}" \
    --s3-prefix sim2real-b \
    --s3-endpoint "${ENDPOINT}" \
    --k8s-context "${CTX}" \
    --k8s-namespace default \
    --json > "${OUT}/status-current.json" 2>"${OUT}/status-current.err" || true
  if [ "${succeeded:-0}" != "0" ]; then
    echo "job_complete=true"
    break
  fi
  if [ "${failed:-0}" != "0" ]; then
    echo "job_failed=true"
    kubectl --context "${CTX}" -n default describe job "${JOB}" > "${OUT}/job-describe.txt" 2>&1 || true
    kubectl --context "${CTX}" -n default logs "job/${JOB}" --tail=300 > "${OUT}/job-failed-tail.log" 2>&1 || true
    exit 1
  fi
  if [ "${elapsed}" -gt "${DEADLINE_SECONDS}" ]; then
    echo "job_timeout=true"
    kubectl --context "${CTX}" -n default describe job "${JOB}" > "${OUT}/job-timeout-describe.txt" 2>&1 || true
    kubectl --context "${CTX}" -n default logs "job/${JOB}" --tail=300 > "${OUT}/job-timeout-tail.log" 2>&1 || true
    exit 1
  fi
  sleep "${MONITOR_INTERVAL_SECONDS:-60}"
done

echo "=== download and validate final report ==="
FINAL_REPORT="${OUT}/sim2real-report.json"
"${PY}" - "${RUN_ID}" "${BUCKET}" "${ENDPOINT}" "${FINAL_REPORT}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient

run_id, bucket, endpoint, dest = sys.argv[1:]
creds = load_credentials()
client = StorageClient(
    endpoint_url=endpoint,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or creds.s3_access_key_id,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or creds.s3_secret_access_key,
)
uri = f"s3://{bucket}/sim2real-b/{run_id}/reports/sim2real-report.json"
last_error = ""
for _ in range(90):
    try:
        client.download_path(uri, dest)
        break
    except Exception as exc:
        last_error = repr(exc)
        time.sleep(10)
else:
    raise SystemExit(f"could not download final report {uri}: {last_error}")

report = json.loads(Path(dest).read_text(encoding="utf-8"))
outer = report.get("outer_loop") or {}
decision = outer.get("latest_decision") or {}
heldout = outer.get("latest_heldout_report") or {}
success_rate = decision.get("success_rate", heldout.get("success_rate"))
threshold = decision.get("threshold", heldout.get("threshold", 0.5))
payload = {
    "run_id": report.get("run_id"),
    "status": report.get("status"),
    "decision": decision.get("decision"),
    "success_rate": success_rate,
    "threshold": threshold,
    "checkpoint_uri": decision.get("checkpoint_uri") or heldout.get("policy_checkpoint"),
    "heldout_envs": heldout.get("generated_envs_tested"),
    "report_uri": uri,
}
Path(str(dest) + ".summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("FINAL_RESULT", json.dumps(payload, sort_keys=True))
if report.get("status") != "completed":
    raise SystemExit(f"report status is not completed: {report.get('status')!r}")
if decision.get("decision") != "promote_checkpoint":
    raise SystemExit(f"latest decision is not promote_checkpoint: {decision!r}")
if success_rate is None or float(success_rate) < float(threshold):
    raise SystemExit(f"success_rate {success_rate!r} is below threshold {threshold!r}")
if not payload["checkpoint_uri"]:
    raise SystemExit("missing promoted checkpoint URI")
PY

echo "=== download and validate final visual artifacts ==="
mkdir -p "${OUT}/reports"
FINAL_RRD="${OUT}/reports/sim2real.rrd"
FINAL_VISUAL_INDEX="${OUT}/reports/sim2real-visual-index.json"
"${PY}" - "${RUN_ID}" "${BUCKET}" "${ENDPOINT}" "${FINAL_RRD}" "${FINAL_VISUAL_INDEX}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient

run_id, bucket, endpoint, rrd_dest, index_dest = sys.argv[1:]
creds = load_credentials()
client = StorageClient(
    endpoint_url=endpoint,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or creds.s3_access_key_id,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or creds.s3_secret_access_key,
)


def _download_with_retry(uri: str, dest: str) -> None:
    last_error = ""
    for _ in range(90):
        try:
            client.download_path(uri, dest)
            if Path(dest).is_file() and Path(dest).stat().st_size > 0:
                return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(10)
    raise SystemExit(f"could not download non-empty artifact {uri}: {last_error}")


rrd_uri = f"s3://{bucket}/sim2real-b/{run_id}/reports/sim2real.rrd"
index_uri = f"s3://{bucket}/sim2real-b/{run_id}/reports/sim2real-visual-index.json"
_download_with_retry(rrd_uri, rrd_dest)
_download_with_retry(index_uri, index_dest)

rrd_path = Path(rrd_dest)
rrd_bytes = rrd_path.read_bytes()
if rrd_path.stat().st_size < 1_000_000:
    raise SystemExit(f"final RRD is unexpectedly small: {rrd_path.stat().st_size} bytes")
required_rrd_tokens = [
    b"synthetic",
    b"heldout/camera",
    b"summary",
    b"signal",
]
missing_tokens = [token.decode("utf-8") for token in required_rrd_tokens if token not in rrd_bytes]
if missing_tokens:
    raise SystemExit(f"final RRD is missing broad visual tokens: {missing_tokens}")

index = json.loads(Path(index_dest).read_text(encoding="utf-8"))
success = index.get("success") or {}
synthetic = index.get("synthetic") or {}
dataset = index.get("dataset") or {}
augmentation = index.get("augmentation") or {}
success_rate = success.get("success_rate")
threshold = success.get("threshold", 0.5)
if success.get("decision") != "promote_checkpoint":
    raise SystemExit(f"visual index decision is not promote_checkpoint: {success!r}")
if success_rate is None or float(success_rate) < float(threshold):
    raise SystemExit(f"visual index success_rate {success_rate!r} below threshold {threshold!r}")
if int(synthetic.get("dataset_sample_count") or 0) <= 0:
    raise SystemExit(f"visual index has no synthetic dataset samples: {synthetic!r}")
if int(synthetic.get("augmentation_sample_count") or 0) <= 0:
    raise SystemExit(f"visual index has no augmentation samples: {synthetic!r}")
if int(dataset.get("train_count") or 0) <= 0 or int(dataset.get("heldout_count") or 0) <= 0:
    raise SystemExit(f"visual index has an invalid train/heldout split: {dataset!r}")

payload = {
    "run_id": run_id,
    "rrd_uri": rrd_uri,
    "rrd_size_bytes": Path(rrd_dest).stat().st_size,
    "visual_index_uri": index_uri,
    "success_rate": success_rate,
    "threshold": threshold,
    "decision": success.get("decision"),
    "train_envs": dataset.get("train_count"),
    "heldout_envs": dataset.get("heldout_count"),
    "augmentation_frames": augmentation.get("frame_count"),
    "synthetic_dataset_samples": synthetic.get("dataset_sample_count"),
    "synthetic_dataset_camera_pngs": synthetic.get("dataset_camera_image_count"),
    "synthetic_dataset_descriptor_previews": synthetic.get("dataset_descriptor_preview_count"),
    "synthetic_augmentation_samples": synthetic.get("augmentation_sample_count"),
    "synthetic_augmentation_pngs": synthetic.get("augmentation_image_count"),
    "synthetic_augmentation_descriptor_previews": synthetic.get("augmentation_descriptor_preview_count"),
}
Path(str(index_dest) + ".summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("FINAL_VISUALS", json.dumps(payload, sort_keys=True))
PY

echo "=== serve Rerun ==="
"${PY}" -m npa.cli.main workbench sim2real rerun serve \
  --run-id "${RUN_ID}" \
  --kubeconfig "${KUBECONFIG_PATH}" \
  --namespace default \
  --s3-bucket "${BUCKET}" \
  --s3-prefix sim2real-b \
  --s3-endpoint "${ENDPOINT}" \
  --rerun-image "${RERUN_IMAGE}" \
  --local-record \
  --local-rrd-path "${OUT}/reports/sim2real-served.rrd" \
  --output json | tee "${OUT}/rerun-serve.json"

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "DONE run_id=${RUN_ID} out=${OUT}"
