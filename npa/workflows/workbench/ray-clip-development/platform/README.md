# Set up the SkyPilot API once

This page is for the operator who owns a Nebius GPU Kubernetes context. It
creates one private SkyPilot API and one namespace for a trusted development
team. Developers can then follow the [CLIP guide](../../../../../docs/testing/fast-source-iteration.md)
using Ray Jobs. They do not set up an API for every source edit or GPU Job.

The service is the upstream SkyPilot API, managed by Docker Compose. Its named
volume retains cluster identities and SSH keys across API restarts. Keep that
volume until every development cluster it owns has been removed.

## Prerequisites

- A Linux operator host with Docker Engine and the
  [Docker Compose plugin](https://docs.docker.com/compose/install/linux/).
  `docker compose version` must work. The tested plugin is **5.5.0**.
- `kubectl` configured for the authorized Nebius GPU cluster, and an installed,
  authenticated `nebius` CLI. `kubectl get nodes` must succeed before continuing.
  The kubeconfig must use `nebius` or `/usr/local/bin/nebius` for its exec-auth
  command; its selected profile must exist in the operator's Nebius CLI config.
- This repository with NPA installed in `npa/.venv`. From the repository root:

  ```bash
  npa/.venv/bin/npa skypilot bootstrap
  export NPA_SKYPILOT_BIN="$(npa/.venv/bin/npa skypilot status --bin-path)"
  export SKYPILOT_DISABLE_USAGE_COLLECTION=1
  ```

The API image is pinned by digest in [compose.yaml](compose.yaml) and reports
SkyPilot **0.12.2**. It runs on the operator host without a GPU. No image build,
object-storage account or managed-jobs controller is part of this setup.

## Give this platform a namespace and private configuration

Choose a fresh platform name below. It becomes the namespace and Compose
project name. These commands copy only the selected kubeconfig context, then
change the copy; your shared kubeconfig stays intact. Run them from the
repository root after selecting the intended context with `kubectl`.

```bash
umask 077
export PLATFORM_NAME=ray-workbench
export PLATFORM_DIR="$HOME/.config/$PLATFORM_NAME"
test ! -e "$PLATFORM_DIR"
mkdir -p "$PLATFORM_DIR"
kubectl config view --raw --minify --flatten > "$PLATFORM_DIR/kubeconfig"
export KUBECONFIG="$PLATFORM_DIR/kubeconfig"
kubectl config set-context --current --namespace "$PLATFORM_NAME"
kubectl create namespace "$PLATFORM_NAME"
kubectl apply -n "$PLATFORM_NAME" \
  -f npa/workflows/workbench/ray-clip-development/network-policy.yaml

cat > "$PLATFORM_DIR/platform.env" <<EOF
COMPOSE_PROJECT_NAME=$PLATFORM_NAME
KUBECONFIG=$KUBECONFIG
NEBIUS_CONFIG_DIR=${NEBIUS_CONFIG_DIR:-$HOME/.nebius}
NEBIUS_BIN=$(command -v nebius)
SKY_API_PORT=46590
EOF
chmod 600 "$PLATFORM_DIR/platform.env" "$KUBECONFIG"
```

The three bind mounts have specific purposes: the selected kubeconfig identifies
the cluster and namespace; the existing Nebius config supplies its exec-auth
profile; and the installed Nebius binary performs that authentication. All three
are read-only. No credentials are put in the repository or forwarded through
application source. Keep `platform.env` and the kubeconfig outside the checkout.

The NetworkPolicy permits ingress from peers in this namespace. It protects the
application Ray network from other namespaces; it does not isolate mutually
untrusted users sharing this platform. Kubernetes API access and port-forward
permissions remain the operator's responsibility.

If you already own a dedicated namespace and private kubeconfig, retain them
instead of creating another copy. Record their ownership, apply the same policy,
and write `platform.env` with those paths. Do not change the API's selected
namespace while it owns development clusters.

## Start and prove the API

```bash
cd npa/workflows/workbench/ray-clip-development/platform
docker compose --env-file "$PLATFORM_DIR/platform.env" up -d --wait
docker compose --env-file "$PLATFORM_DIR/platform.env" exec -T api \
  kubectl auth can-i create pods
docker compose --env-file "$PLATFORM_DIR/platform.env" exec -T api \
  kubectl auth can-i create pods/portforward

export SKYPILOT_API_SERVER_ENDPOINT=http://127.0.0.1:46590
export KUBE_CONTEXT="$(kubectl config current-context)"
"$NPA_SKYPILOT_BIN" check kubernetes \
  --config "kubernetes.allowed_contexts=[\"$KUBE_CONTEXT\"]" -o json
"$NPA_SKYPILOT_BIN" gpus list --infra "k8s/$KUBE_CONTEXT" \
  --config "kubernetes.allowed_contexts=[\"$KUBE_CONTEXT\"]"
cd ..
"$NPA_SKYPILOT_BIN" launch --dryrun --yes -c "$PLATFORM_NAME-check" \
  --infra "k8s/$KUBE_CONTEXT" \
  --config "kubernetes.allowed_contexts=[\"$KUBE_CONTEXT\"]" cluster.yaml
```

Require both permission probes to say `yes`, the check's JSON to contain
`"Kubernetes": ["compute"]`, and the dry-run to finish successfully with the
intended context. The accelerator name in `cluster.yaml` must appear in the GPU
listing; use SkyPilot's displayed spelling without changing shared node labels.
SkyPilot can return exit zero when its credential check enables
no infrastructure. An HTTP health response alone also does not prove that the
API's execution queue works. This dry-run allocates no GPU; the first real launch
in the CLIP guide proves the remaining bootstrap and CUDA boundary.

The service binds only the operator host's loopback port. It has no Docker socket,
host network or privileged mode. Root inside the container has only
`DAC_OVERRIDE`, needed to read the explicitly mounted owner-only credentials;
all other capabilities are dropped and privilege escalation is disabled. This
keeps the image's installed rsync helper writable by its owner without changing
the helper, chmodding operator credentials or rebuilding the image.

Use this API from the same trusted host. Remote operators must first use
authenticated SSH to that host or an authenticated SSH tunnel; do not publish
this HTTP port on a public interface. Ray's separate Jobs endpoint is reached
through the authenticated tunnel in the CLIP guide.

Give developers these values: the API endpoint, the fixed namespace's kubeconfig
and context, and the pinned `NPA_SKYPILOT_BIN` path. Then return to the
[first GPU run](../../../../../docs/testing/fast-source-iteration.md). Keep the
API running across source edits and across the medium and distributed examples.

## Finish the platform

Developers first stop their exact Ray Jobs, download and verify their outputs,
cancel the corresponding Sky service task and run `sky down` for their named
development cluster. Removing this API first would discard its ability to track
and clean up those clusters.

Once **all clusters owned by this API are absent**, run from the repository root
with the same private configuration:

```bash
"$NPA_SKYPILOT_BIN" status --no-show-managed-jobs -o json
kubectl get pods -n "$PLATFORM_NAME"
# Require an empty cluster list and no remaining workload pods before continuing.
docker compose --env-file "$PLATFORM_DIR/platform.env" \
  -f npa/workflows/workbench/ray-clip-development/platform/compose.yaml \
  down --volumes
unset SKYPILOT_API_SERVER_ENDPOINT
```

If this setup created the namespace, remove it using the retained kubeconfig:

```bash
kubectl delete namespace "$PLATFORM_NAME"
unset KUBECONFIG
```

Retain an operator-provided namespace. Compose removes this project's API
container, network and named volume; it does not stop another API or destroy the
Nebius Kubernetes cluster or its GPU nodes. Retain verified application outputs
and private ownership receipts.

This follows SkyPilot's [upstream API deployment](https://docs.skypilot.ai/en/latest/reference/api-server/examples/api-server-in-docker.html)
and [separate application Ray runtime](https://docs.skypilot.ai/en/latest/examples/training/ray.html)
boundaries. It is an explicit platform service, not a Ray submission wrapper.
