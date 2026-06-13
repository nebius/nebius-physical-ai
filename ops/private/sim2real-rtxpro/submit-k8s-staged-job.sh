#!/usr/bin/env bash
# Direct Kubernetes submit for sim2real staged runbook.
# Bypasses SkyPilot kubeconfig mismatch (workbench vs cluster contexts).
# Credentials: cluster secretRef only — never embedded in generated manifests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"

readarray -t _npa_cfg < <(operator_read_config "${ROOT}")
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"
REG="${REGISTRY:-${_npa_cfg[2]:-}}"
CTX="${KUBECONTEXT:-${_npa_cfg[3]:-}}"
if [ -z "${CTX}" ]; then
  echo "Set k8s_context in ~/.npa/config.yaml (storage or projects.<alias>)" >&2
  exit 1
fi
export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${CTX}")}"
RUN_ID="${RUN_ID:-sim2real-staged-$(date -u +%Y%m%dT%H%M%Sz | tr '[:upper:]' '[:lower:]')}"
if [ -z "${BUCKET}" ]; then
  echo "Set S3_BUCKET or configure storage.bucket in ~/.npa/config.yaml" >&2
  exit 1
fi
if [ -z "${REG}" ]; then
  echo "Set REGISTRY or configure storage.registry in ~/.npa/config.yaml" >&2
  exit 1
fi

# Preflight: every image must be registry-qualified before we apply the Job.
TRAINER_IMAGE="${TRAINER_IMAGE:-${REG}/npa-lerobot-vlm-rl:0.1.0}"
VLM_IMAGE="${VLM_IMAGE:-${REG}/npa-cosmos3-reason:3.0.1-genuine-sm120}"
EVAL_IMAGE="${EVAL_IMAGE:-${REG}/npa-sim2real-eval:0.1.1-genuine-sm120}"
AUGMENT_IMAGE="${AUGMENT_IMAGE:-${REG}/npa-cosmos2-transfer:2.5.0}"
POLICY_IMAGE="${POLICY_IMAGE:-${REG}/npa-sim2real-reference-policy:0.1.1}"
ISAAC_IMAGE="${ISAAC_IMAGE:-${REG}/npa-isaac-lab:2.3.2.post1}"
ORCHESTRATOR_IMAGE="${ORCHESTRATOR_IMAGE:-${TRAINER_IMAGE}}"

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
            - name: ISAAC_IMAGE
              value: "${ISAAC_IMAGE}"
            - name: NPA_SIM2REAL_ISAAC_TASK
              value: "${NPA_SIM2REAL_ISAAC_TASK:-Isaac-Lift-Cube-Franka-v0}"
            - name: NPA_SIM2REAL_SIM_BACKEND
              value: "${NPA_SIM2REAL_SIM_BACKEND:-isaac}"
            - name: INNER_ITERATIONS
              value: "${INNER_ITERATIONS:-1}"
            - name: OUTER_ITERATIONS
              value: "${OUTER_ITERATIONS:-1}"
            - name: NPA_ENV_COUNT
              value: "${NPA_ENV_COUNT:-10000}"
            - name: NPA_TRAIN_FRACTION
              value: "${NPA_TRAIN_FRACTION:-0.8}"
            - name: ROLLOUT_COUNT
              value: "2"
            - name: HELDOUT_ENV_COUNT
              value: "${HELDOUT_ENV_COUNT:-4}"
            - name: NPA_SIM2REAL_HELDOUT_EVAL_LIMIT
              value: "${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-0}"
            - name: SUCCESS_THRESHOLD
              value: "0.45"
            - name: NPA_SOURCE_REPO
              value: "https://github.com/nebius/nebius-physical-ai.git"
            - name: NPA_SOURCE_REF
              value: "feat/sim2real-mandatory-stages"
            - name: NPA_SIM2REAL_K8S_NAMESPACE
              value: "default"
            - name: NPA_SIM2REAL_K8S_SERVICE_ACCOUNT
              value: "agent-sa"
          envFrom:${ENV_FROM_YAML}
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              exec > >(tee -a /tmp/run.log) 2>&1
              git clone --depth 1 --branch "\${NPA_SOURCE_REF}" "\${NPA_SOURCE_REPO}" /tmp/npa-src
              export PYTHONPATH="/tmp/npa-src/npa/src:\${PYTHONPATH:-}"
              python3 -c "import npa.workflows.sim2real_loop as m; print(m.__file__)"
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
                --inner-iterations "\${INNER_ITERATIONS:-1}"
                --outer-iterations "\${OUTER_ITERATIONS:-1}"
                --rollout-count "\${ROLLOUT_COUNT:-2}"
                --heldout-env-count "\${HELDOUT_ENV_COUNT:-4}"
                --heldout-eval-limit "\${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-0}"
                --threshold "\${SUCCESS_THRESHOLD:-0.45}"
                --sim-backend "\${NPA_SIM2REAL_SIM_BACKEND:-isaac}"
                --isaac-image "\${ISAAC_IMAGE}"
                --isaac-task "\${NPA_SIM2REAL_ISAAC_TASK:-Isaac-Lift-Cube-Franka-v0}"
                --augment-image "\${AUGMENT_IMAGE}"
                --policy-image "\${POLICY_IMAGE}"
                --env-count "\${NPA_ENV_COUNT:-10000}"
                --train-fraction "\${NPA_TRAIN_FRACTION:-0.8}"
                --vlm-image "\${VLM_IMAGE}"
                --eval-image "\${EVAL_IMAGE}"
                --trainer-image "\${TRAINER_IMAGE}"
                --k8s-namespace "\${NPA_SIM2REAL_K8S_NAMESPACE:-default}"
                --k8s-service-account "\${NPA_SIM2REAL_K8S_SERVICE_ACCOUNT:-agent-sa}"
                --upload-artifacts
              )
              python3 -m npa.workflows.sim2real_loop preamble "\${common_args[@]}"
              state_json="\${output_dir}/state/workflow_state.json"
              current_quality=\$(python3 -c "import json; print(json.load(open('\${state_json}'))['current_quality'])")
              for outer in \$(seq 1 "\${OUTER_ITERATIONS:-1}"); do
                python3 -m npa.workflows.sim2real_loop outer-iteration "\${common_args[@]}" \
                  --outer-iteration "\${outer}" --initial-quality "\${current_quality}"
                current_quality=\$(python3 -c "import json; print(json.load(open('\${state_json}'))['current_quality'])")
                decision=\$(python3 -c "import json; print(json.load(open('\${state_json}'))['final_decision']['decision'])")
                echo "outer=\${outer} quality=\${current_quality} decision=\${decision}"
                [ "\${decision}" = "promote_checkpoint" ] && break
              done
              python3 -m npa.workflows.sim2real_loop finalize "\${common_args[@]}"
              python3 -c "import json; from pathlib import Path; r=json.loads(Path('\${output_dir}/reports/sim2real-report.json').read_text()); print('CLUSTER_METRICS', json.dumps({'run_id': r['run_id'], 'decision': r['outer_loop']['latest_decision'], 'reward_trend': r['inner_loop']['reward_trend']}))"
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition
YAML

chmod 600 "${MANIFEST}"

echo "Applying job ${JOB} to context ${CTX}..." | tee "${LOG}"
kubectl --context "${CTX}" apply -f "${MANIFEST}" | tee -a "${LOG}"
echo "run_id=${RUN_ID} job=${JOB} manifest=${MANIFEST} log=${LOG}"
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
