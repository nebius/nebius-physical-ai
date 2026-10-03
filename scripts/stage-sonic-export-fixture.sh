#!/usr/bin/env bash
# Stage a real SONIC policy checkpoint fixture to S3, built IN the cluster.
#
# Why in-cluster: building the fixture needs torch, the dev/operator VM regularly runs
# at >90% disk, and the SONIC workbench image already has torch cached on the cluster
# nodes. Nothing heavy lands locally. Same reasoning as
# scripts/build-workbench-image-in-cluster.sh.
#
# The fixture is what makes the SONIC npa.workflow twins (sonic-export, sonic-eval,
# sonic-export-eval) live-testable: `npa workbench sonic export` needs a loadable torch
# policy checkpoint, and this repo deliberately does not vendor NVIDIA's gated
# nvidia/GEAR-SONIC weights. The builder is a real, unit-tested module
# (npa/src/npa/workflows/sonic_fixture.py) mounted into the pod through a ConfigMap, so
# there is exactly one source of truth for the fixture's shape.
#
# Usage:
#   scripts/stage-sonic-export-fixture.sh \
#     --image <registry>/npa-sonic:<tag> \
#     --uri   s3://<bucket>/<prefix>/sonic-export-fixture/checkpoint.pt
#
# Then point the live harness at it:
#   export NPA_E2E_SONIC_CHECKPOINT_SRC=s3://<bucket>/<prefix>/sonic-export-fixture/checkpoint.pt
#
# Credentials: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL are read
# from the environment (as usual for Nebius S3) and passed to the pod as env vars.
# Nothing is hardcoded: image, URI, namespace and pull secret all come from flags.
set -euo pipefail

IMAGE=""
URI=""
NAMESPACE="${NPA_BUILD_NAMESPACE:-default}"
PULL_SECRET="${NPA_BUILD_PULL_SECRET:-}"
POD_NAME="${NPA_SONIC_FIXTURE_POD:-npa-sonic-fixture}"
TIMEOUT_SECONDS="${NPA_SONIC_FIXTURE_TIMEOUT:-1800}"
OBS_DIM="${NPA_SONIC_FIXTURE_OBS_DIM:-48}"
ACT_DIM="${NPA_SONIC_FIXTURE_ACT_DIM:-12}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILDER="${REPO_ROOT}/npa/src/npa/workflows/sonic_fixture.py"

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --uri) URI="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --pull-secret) PULL_SECRET="$2"; shift 2 ;;
    --pod) POD_NAME="$2"; shift 2 ;;
    --obs-dim) OBS_DIM="$2"; shift 2 ;;
    --act-dim) ACT_DIM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$IMAGE" ]] || { echo "ERROR: --image <registry>/npa-sonic:<tag> is required" >&2; exit 2; }
[[ -n "$URI" ]] || { echo "ERROR: --uri s3://bucket/key is required" >&2; exit 2; }
[[ -f "$BUILDER" ]] || { echo "ERROR: builder not found: $BUILDER" >&2; exit 2; }
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  [[ -n "${!var:-}" ]] || { echo "ERROR: $var must be set" >&2; exit 2; }
done

CONFIGMAP="${POD_NAME}-src"
SECRET="${POD_NAME}-creds"

cleanup() {
  kubectl -n "$NAMESPACE" delete pod "$POD_NAME" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  kubectl -n "$NAMESPACE" delete configmap "$CONFIGMAP" --ignore-not-found >/dev/null 2>&1 || true
  kubectl -n "$NAMESPACE" delete secret "$SECRET" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cleanup
kubectl -n "$NAMESPACE" create configmap "$CONFIGMAP" \
  --from-file=sonic_fixture.py="$BUILDER" >/dev/null
# Credentials go in a Secret, never in the pod spec / process list.
kubectl -n "$NAMESPACE" create secret generic "$SECRET" \
  --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --from-literal=AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-}" >/dev/null

echo "building SONIC checkpoint fixture in-cluster (pod=$POD_NAME ns=$NAMESPACE)" >&2

PULL_SECRET_YAML=""
if [[ -n "$PULL_SECRET" ]]; then
  printf -v PULL_SECRET_YAML '  imagePullSecrets:\n    - name: %s' "$PULL_SECRET"
fi

kubectl -n "$NAMESPACE" apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
spec:
  restartPolicy: Never
${PULL_SECRET_YAML}
  volumes:
    - name: src
      configMap:
        name: ${CONFIGMAP}
  containers:
    - name: builder
      image: ${IMAGE}
      command: ["/bin/bash", "-lc"]
      args:
        - |
          set -euo pipefail
          python3 -m pip install --quiet --break-system-packages boto3 \
            || python3 -m pip install --quiet --user boto3 \
            || python3 -m pip install --quiet boto3
          python3 /npa-src/sonic_fixture.py \
            --checkpoint-uri "${URI}" \
            --obs-dim "${OBS_DIM}" \
            --act-dim "${ACT_DIM}"
      envFrom:
        - secretRef:
            name: ${SECRET}
      volumeMounts:
        - name: src
          mountPath: /npa-src
      resources:
        requests:
          cpu: "2"
          memory: 8Gi
YAML

echo "waiting for the fixture pod to finish (timeout ${TIMEOUT_SECONDS}s)" >&2
deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
phase=""
while [[ $(date +%s) -lt $deadline ]]; do
  phase="$(kubectl -n "$NAMESPACE" get pod "$POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo '')"
  case "$phase" in
    Succeeded|Failed) break ;;
  esac
  sleep 10
done

kubectl -n "$NAMESPACE" logs "$POD_NAME" 2>&1 | tail -40

if [[ "$phase" != "Succeeded" ]]; then
  echo "ERROR: fixture pod phase=${phase:-unknown}" >&2
  exit 1
fi

echo "$URI"
