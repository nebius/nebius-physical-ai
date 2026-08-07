#!/usr/bin/env bash
# Build a workbench image *in the Kubernetes cluster* with kaniko and push it to the
# Nebius registry — no local docker pull, no local disk pressure.
#
# Why this exists: the operator/dev VM cannot always pull a workbench base image
# (npa-isaac-lab is ~8 GB compressed, and the VM regularly runs at >90% disk). The
# cluster nodes already cache these layers, and the cluster already holds a registry
# pull secret, so building there is both faster and disk-safe.
#
# Two modes:
#
#   1. Derived build (default): apply a small Dockerfile ON TOP of an existing tag.
#      Used to add the Kubernetes prerequisites (system python3, rsync, group
#      membership) that let SkyPilot host a task in a heavy vendor image.
#
#        scripts/build-workbench-image-in-cluster.sh \
#          --base <registry>/npa-isaac-lab:2.3.2.post1 \
#          --tag  <registry>/npa-isaac-lab:2.3.2.post1-sky1 \
#          --dockerfile npa/docker/workbench/isaac-lab/Dockerfile.k8s-prereqs
#
#   2. Inline snippet: pass --run "apt-get update && ..." instead of --dockerfile.
#
# Nothing is hardcoded: registry, tags, namespace, pull secret and kaniko image all
# come from flags or the environment.
set -euo pipefail

BASE=""
TAG=""
DOCKERFILE=""
RUN_SNIPPET=""
NAMESPACE="${NPA_BUILD_NAMESPACE:-default}"
PULL_SECRET="${NPA_BUILD_PULL_SECRET:-npa-nebius-registry}"
# Pinned by digest: an unpinned build tool undermines the reproducibility this
# script exists for, and the repo pins its own base images the same way.
# Refresh with: crane digest gcr.io/kaniko-project/executor:<version>
KANIKO_IMAGE="${NPA_KANIKO_IMAGE:-gcr.io/kaniko-project/executor:v1.23.2@sha256:9e69fd4330ec887829c780f5126dd80edc663df6def362cd22e79bcdf00ac53f}"
POD_NAME="${NPA_BUILD_POD:-npa-image-build}"
TIMEOUT_SECONDS="${NPA_BUILD_TIMEOUT:-1800}"

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --dockerfile) DOCKERFILE="$2"; shift 2 ;;
    --run) RUN_SNIPPET="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --pull-secret) PULL_SECRET="$2"; shift 2 ;;
    --pod) POD_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TAG" ]]; then
  echo "ERROR: --tag <registry>/<image>:<tag> is required" >&2
  exit 2
fi
if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NPA_BUILD_TIMEOUT must be a non-negative integer (0 waits indefinitely)" >&2
  exit 2
fi
if [[ -z "$DOCKERFILE" && -z "$RUN_SNIPPET" ]]; then
  echo "ERROR: pass --dockerfile <path> or --run '<shell>'" >&2
  exit 2
fi
if [[ -n "$RUN_SNIPPET" && -z "$BASE" ]]; then
  echo "ERROR: --run requires --base <existing image>" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

if [[ -n "$DOCKERFILE" ]]; then
  [[ -f "$DOCKERFILE" ]] || { echo "ERROR: no such Dockerfile: $DOCKERFILE" >&2; exit 2; }
  cp "$DOCKERFILE" "$WORK_DIR/Dockerfile"
  if [[ -n "$BASE" ]]; then
    # Allow a Dockerfile that starts with `ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`.
    printf '\n' >> "$WORK_DIR/Dockerfile"
  fi
else
  cat > "$WORK_DIR/Dockerfile" <<EOF
FROM ${BASE}
USER root
RUN ${RUN_SNIPPET}
USER ubuntu
EOF
fi

echo "--- Dockerfile ---"
cat "$WORK_DIR/Dockerfile"
echo "------------------"

CM_NAME="${POD_NAME}-dockerfile"
kubectl -n "$NAMESPACE" delete pod "$POD_NAME" --ignore-not-found >/dev/null
kubectl -n "$NAMESPACE" delete configmap "$CM_NAME" --ignore-not-found >/dev/null
kubectl -n "$NAMESPACE" create configmap "$CM_NAME" \
  --from-file=Dockerfile="$WORK_DIR/Dockerfile" >/dev/null

BUILD_ARGS=()
if [[ -n "$BASE" && -n "$DOCKERFILE" ]]; then
  BUILD_ARGS+=("        - --build-arg=BASE_IMAGE=${BASE}")
fi

cat > "$WORK_DIR/pod.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
spec:
  restartPolicy: Never
  containers:
    - name: kaniko
      image: ${KANIKO_IMAGE}
      args:
        - --dockerfile=/build/Dockerfile
        - --context=dir:///build
        - --destination=${TAG}
        - --single-snapshot
        - --verbosity=info
$(printf '%s\n' "${BUILD_ARGS[@]}")
      resources:
        requests:
          cpu: "4"
          memory: 16Gi
      volumeMounts:
        - name: dockerfile
          # NOT /workspace: that is the WORKDIR of some workbench images (Isaac Lab
          # uses /workspace/isaaclab) and kaniko then fails to create it under its
          # read-only context mount.
          mountPath: /build
        - name: regcred
          mountPath: /kaniko/.docker
  volumes:
    - name: dockerfile
      configMap:
        name: ${CM_NAME}
    - name: regcred
      secret:
        secretName: ${PULL_SECRET}
        items:
          - key: .dockerconfigjson
            path: config.json
EOF

kubectl -n "$NAMESPACE" apply -f "$WORK_DIR/pod.yaml" >/dev/null
if [[ "$TIMEOUT_SECONDS" -eq 0 ]]; then
  echo "build pod ${POD_NAME} submitted; waiting without a deadline"
else
  echo "build pod ${POD_NAME} submitted; streaming progress (timeout ${TIMEOUT_SECONDS}s)"
fi

deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
phase=""
while [[ "$TIMEOUT_SECONDS" -eq 0 || $(date +%s) -lt $deadline ]]; do
  phase="$(kubectl -n "$NAMESPACE" get pod "$POD_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  case "$phase" in
    Succeeded) break ;;
    Failed) break ;;
  esac
  sleep 15
done

kubectl -n "$NAMESPACE" logs "$POD_NAME" 2>&1 | tail -25 || true
kubectl -n "$NAMESPACE" delete pod "$POD_NAME" --ignore-not-found >/dev/null
kubectl -n "$NAMESPACE" delete configmap "$CM_NAME" --ignore-not-found >/dev/null

if [[ "$phase" != "Succeeded" ]]; then
  echo "ERROR: image build did not succeed (phase=${phase:-unknown})" >&2
  cat >&2 <<'HINT'

If the log says UNAUTHORIZED / "authentication required", the cluster's registry
secret has expired - Nebius IAM tokens are short-lived, so a long-lived pull secret
goes stale. Refresh it with the same identity and retry:

  TOKEN=$(npa/.venv/bin/python -c \
    'from npa.workflows.sim2real.registry_auth import mint_nebius_registry_token; print(mint_nebius_registry_token())')
  kubectl create secret docker-registry <pull-secret> -n <namespace> \
    --docker-server=<registry-host> --docker-username=iam --docker-password="$TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -
HINT
  exit 1
fi

echo "pushed ${TAG}"
