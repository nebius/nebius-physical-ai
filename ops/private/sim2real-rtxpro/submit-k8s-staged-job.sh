#!/usr/bin/env bash
# Direct Kubernetes submit for sim2real staged runbook on npa-rtxpro-mk8s.
# Bypasses SkyPilot kubeconfig mismatch (workbench vs rtxpro contexts).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export KUBECONFIG="${KUBECONFIG:-$HOME/.npa/clusters/npa-rtxpro-mk8s/kubeconfig}"
CTX="${KUBECONTEXT:-npa-rtxpro-mk8s}"
RUN_ID="${RUN_ID:-rtxpro-staged-$(date -u +%Y%m%dT%H%M%Sz | tr '[:upper:]' '[:lower:]')}"

# Resolve bucket/registry/endpoint from env or ~/.npa/config.yaml (no hardcoded tenant values).
readarray -t _npa_cfg < <("${ROOT}/npa/.venv/bin/python" - <<'PY'
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path.home().joinpath(".npa/config.yaml").read_text())
storage = cfg.get("storage") or {}
bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/")[0]
endpoint = storage.get("endpoint_url", "https://storage.eu-north1.nebius.cloud")
registry = storage.get("registry", cfg.get("registry", "")).rstrip("/")
print(bucket)
print(endpoint)
print(registry)
PY
)
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"
REG="${REGISTRY:-${_npa_cfg[2]:-}}"
if [ -z "${BUCKET}" ]; then
  echo "Set S3_BUCKET or configure storage.bucket in ~/.npa/config.yaml" >&2
  exit 1
fi
if [ -z "${REG}" ]; then
  echo "Set REGISTRY or configure storage.registry in ~/.npa/config.yaml" >&2
  exit 1
fi
LOG="/tmp/sim2real-cluster/${RUN_ID}.log"
mkdir -p /tmp/sim2real-cluster

# Load secrets into env for job env vars (never log values).
eval "$("${ROOT}/npa/.venv/bin/python" - <<'PY'
import yaml
from pathlib import Path
c = yaml.safe_load(Path.home().joinpath(".npa/credentials.yaml").read_text())
s = c.get("storage") or {}
for key, env in (
    ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
    ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
):
    val = s.get(key, "")
    if val:
        print(f"export {env}={val!r}")
tok = (c.get("tokens") or {}).get("HF_TOKEN", "")
if tok:
    print(f"export HF_TOKEN={tok!r}")
ngc = (c.get("ngc") or {}).get("api_key", "")
if ngc:
    print(f"export NGC_API_KEY={ngc!r}")
PY
)"

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
          image: ${REG}/npa-lerobot-vlm-rl:0.1.0
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
              value: "${REG}/npa-lerobot-vlm-rl:0.1.0"
            - name: VLM_IMAGE
              value: "${REG}/npa-cosmos3-reason:3.0.1-genuine-sm120"
            - name: EVAL_IMAGE
              value: "${REG}/npa-sim2real-eval:0.1.1-genuine-sm120"
            - name: NPA_SIM2REAL_SIM_BACKEND
              value: "genesis"
            - name: INNER_ITERATIONS
              value: "${INNER_ITERATIONS:-1}"
            - name: OUTER_ITERATIONS
              value: "${OUTER_ITERATIONS:-1}"
            - name: ROLLOUT_COUNT
              value: "2"
            - name: HELDOUT_ENV_COUNT
              value: "4"
            - name: SUCCESS_THRESHOLD
              value: "0.45"
            - name: NPA_SOURCE_REPO
              value: "https://github.com/nebius/nebius-physical-ai.git"
            - name: NPA_SOURCE_REF
              value: "feat/sim2real-staged-runbook"
            - name: NPA_SIM2REAL_K8S_NAMESPACE
              value: "default"
            - name: NPA_SIM2REAL_K8S_SERVICE_ACCOUNT
              value: "agent-sa"
            - name: AWS_ACCESS_KEY_ID
              value: "${AWS_ACCESS_KEY_ID:-}"
            - name: AWS_SECRET_ACCESS_KEY
              value: "${AWS_SECRET_ACCESS_KEY:-}"
            - name: HF_TOKEN
              value: "${HF_TOKEN:-}"
            - name: NGC_API_KEY
              value: "${NGC_API_KEY:-}"
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
                --threshold "\${SUCCESS_THRESHOLD:-0.45}"
                --sim-backend "\${NPA_SIM2REAL_SIM_BACKEND:-genesis}"
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

echo "Applying job ${JOB} to context ${CTX}..." | tee "${LOG}"
kubectl --context "${CTX}" apply -f "${MANIFEST}" | tee -a "${LOG}"
echo "run_id=${RUN_ID} job=${JOB} manifest=${MANIFEST} log=${LOG}"
