# Reproduce Ray + SkyPilot source iterations

The [audit](../architecture/ray-fast-development-audit.md) recommends an
independent application Ray runtime on SkyPilot-managed Nebius GPUs. The
committed recipe exercises real Workbench CLIP inference through upstream Jobs
and `runtime_env.working_dir`, with image reuse across source revisions.
**Medium and complex GPU checks passed, including durable workflow completion
on one GPU node and on two Kubernetes GPU worker nodes.**
The [medium](../architecture/ray-fast-development-audit.md#measured-medium-result)
and [complex](../architecture/ray-fast-development-audit.md#measured-complex-result)
results record final jobs, artifact and cleanup checks, and earlier
startup/client/lifecycle repairs.
The retained CPU benchmark below is historical.

## Prepare the session and client

Use this checkout's `npa/.venv`, installed through the declared `npa[dev,adapter]`
setup. Select a configured, authorized Nebius GPU environment with compatible
GPUs, image pulling, S3 read/write access, and authenticated Kubernetes
access. The [session YAML](../../npa/workflows/workbench/npa-workflows/ray-clip-development-session.yaml)
uses an existing official digest-qualified PyTorch 2.12.1/CUDA 13 runtime image,
an application Ray 2.46.0 environment, and the public `openai/clip-vit-base-patch32` snapshot at revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. It verifies the model weight hash
before accepting application jobs. It installs pinned Ray, LanceDB, Transformers
and the supporting application stack once per session. This preparation and
model download are excluded from source iteration timings. The client requires
`--udf-source` pointing to this checkout's actual Workbench CLIP UDF; that reviewed
source is delivered through Ray with the application, not baked into this image.
The actual image interpreter is `/usr/bin/python3.12`; `ensurepip` is absent.
The YAML creates its venv with `--system-site-packages --without-pip`, then runs
the base interpreter's `pip --python /tmp/npa-ray-app/bin/python` to install the
pinned stack. Do not assume a Conda path or working default `venv` bootstrap.
[pip interpreter targeting](https://pip.pypa.io/en/stable/topics/python-option/)

The first session attempt using the published non-root LanceDB image failed
during SkyPilot SSH bootstrap because it had no `sudo`; application Ray never
started. The revised PyTorch image starts as root, enabling that bootstrap in
the isolated workload pod. It does not request a privileged pod or host-root
access. Use a cluster policy that permits this image's runtime user, or select
an already compatible image; source delivery cannot repair the bootstrap
boundary. The API container remains a separate non-root component.
[SkyPilot 0.12.2 Kubernetes bootstrap](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/templates/kubernetes-ray.yml.j2)

Resolve project, storage, Kubernetes and run identities from private operator
configuration. Keep their values and receipts outside Git. Use fresh run IDs,
S3 prefixes and output directories for the medium and complex sessions. The
application advertises `config.gpus_per_node` GPUs per node: use one node with
one GPU for medium; use either two nodes with one GPU each or one node with two
GPUs for complex. Record the actual physical node/GPU scope.

```bash
set -euo pipefail
umask 077
export NPA_RAY_CLIENT_ROOT="$(mktemp -d)"
uv venv --python 3.12 "$NPA_RAY_CLIENT_ROOT/venv"
export NPA_RAY_CLIENT_PYTHON="$NPA_RAY_CLIENT_ROOT/venv/bin/python"
uv pip install --python "$NPA_RAY_CLIENT_PYTHON" 'ray[default]==2.46.0' 'numpy==2.4.6' 'boto3==1.42.9'
uv pip freeze --python "$NPA_RAY_CLIENT_PYTHON" > "$NPA_RAY_CLIENT_ROOT/client-freeze.txt"
npa/.venv/bin/npa skypilot bootstrap
npa/.venv/bin/npa workbench health preflight --checks s3 --json
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/ray-clip-development-session.yaml
```

The frozen client environment is an experiment receipt; the application image and its separately prepared Ray
environment determine the worker dependencies. Follow the normal image/model
access and GPU placement preflight before submitting.
NumPy is also required on the client for the final cross-revision comparison;
successful GPU jobs alone do not complete that client-side validation.
The session pins its named application dependencies and freezes the resolved
environment; it does not provide a hash lock for every transitive dependency.
The health command's S3 check lists the configured bucket; that result does not
prove write permission to the workflow's resolved destination. Keep submission's
stricter write preflight enabled and reconcile the selected bucket/endpoint if
it fails.

Select a separately owned SkyPilot API server endpoint for this session. Retain
both `SKYPILOT_API_SERVER_ENDPOINT` and the selected `SKYPILOT_USER_ID` in every
submission/status/cancellation shell.
`--isolated-config-dir` isolates local state but does not change SkyPilot's
default host-wide API port. The API server's fixed identity determines its jobs
controller, so a distinct client user ID alone cannot isolate controller
ownership. Keep the API endpoint on loopback, start it with the selected private
Kubernetes/storage configuration, and retain its exact process identity for
cleanup. Do not stop or reconfigure another run's API server.
[SkyPilot 0.12.2 server identity](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/utils/common.py)

Use the upstream [container deployment](https://github.com/skypilot-org/skypilot/blob/v0.12.2/docs/source/reference/api-server/examples/api-server-in-docker.rst)
with bridge networking to isolate its internal request-queue port as well as its
HTTP listener. Changing only the host process's HTTP port leaves the pinned
queue manager on port 50011. Publish only the API HTTP port on host loopback;
mount a run-owned API home, the exact selected kubeconfig and required provider
credentials, rather than the operator's entire home. Do not use host networking,
publish the queue port, or stop the shared API daemon. The following container
startup was qualified with SkyPilot 0.12.2; application GPU qualification remains
separate.
[SkyPilot 0.12.2 request queue](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/server/requests/queues/mp_queue.py),
[Docker port publishing](https://docs.docker.com/engine/network/port-publishing/)

Set `NPA_NEBIUS_CONFIG_DIR`, `NPA_NEBIUS_BIN` and `NPA_KUBECONFIG_PATH` to the
exact authorized configuration directory, static CLI executable and private
generated kubeconfig. Its exec-auth command must resolve to
`/usr/local/bin/nebius` inside the container and preserve the selected profile.
Before starting the API or submitting GPU work, select a dedicated session
namespace and configure its ingress policy. The workflow YAML does not create
network isolation. The following matches the validation configuration: all pods
in this namespace accept ingress only from pods in the same namespace. Ray's
GCS, Client and worker sockets bind pod interfaces even though Dashboard/Jobs
binds loopback. Require a CNI that enforces NetworkPolicy; inspect other applicable
policies because their allowed traffic is additive. This ingress policy leaves
egress unrestricted and does not isolate the session from trusted nodes or
privileged/host-network actors. Configuration was verified during this run;
cross-namespace penetration testing was not performed.
[Kubernetes NetworkPolicy semantics](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

Use a fresh `NPA_KUBE_NAMESPACE` owned by this session. If the operator already
prepared it, verify equivalent isolation and ownership instead of creating it
again. Point only the run-owned kubeconfig at that namespace; the API container
mounts this exact file. Do not change the global kubeconfig.

```bash
set -euo pipefail
umask 077
kube=(kubectl --kubeconfig "$NPA_KUBECONFIG_PATH" --context "$NPA_KUBE_CONTEXT")
"${kube[@]}" create namespace "$NPA_KUBE_NAMESPACE"
"${kube[@]}" label namespace "$NPA_KUBE_NAMESPACE" "npa.dev/run=$NPA_DEV_RUN_ID"
"${kube[@]}" --namespace "$NPA_KUBE_NAMESPACE" create -f - <<YAML_RAY_POLICY
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: owned-session-ingress
  labels:
    npa.dev/run: "$NPA_DEV_RUN_ID"
spec:
  podSelector: {}
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector: {}
YAML_RAY_POLICY
"${kube[@]}" config set-context "$NPA_KUBE_CONTEXT" --namespace "$NPA_KUBE_NAMESPACE"
"${kube[@]}" --namespace "$NPA_KUBE_NAMESPACE" get networkpolicies -o json \
  > "$NPA_RAY_CLIENT_ROOT/network-policy-receipt.json"
"${kube[@]}" get namespace "$NPA_KUBE_NAMESPACE" -o json \
  > "$NPA_RAY_CLIENT_ROOT/namespace-receipt.json"
```

Verify that the rendered placement and resulting controller/application pods all
use this namespace before accepting Ray jobs. The namespace groups trusted
session components; it is not a boundary between mutually untrusted customers.

Supply storage credentials through the operator process environment. The API
home below is separate from the client's isolated state and contains no symlinks
to another run's state. Image pulling is preparation, not source iteration.
SkyPilot also requires `[nebius]` in the API home's `.aws/credentials` and
`[profile nebius]` in `.aws/config`; AWS environment variables alone do not enable
its Nebius storage capability. Materialize only the already verified selected
project's S3 credentials, region and endpoint in that run-owned home. The snippet
creates fresh `0600` files without printing values or changing global config.
[SkyPilot 0.12.2 Nebius storage preflight](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/clouds/nebius.py)

```bash
set -euo pipefail
umask 077
export NPA_RAY_SKY_STATE="$NPA_RAY_CLIENT_ROOT/sky-state"
export NPA_RAY_API_HOME="$NPA_RAY_CLIENT_ROOT/api-home"
mkdir -p "$NPA_RAY_API_HOME/.kube" "$NPA_RAY_API_HOME/.nebius"
npa/.venv/bin/python - <<'PY_SKY_STORAGE'
import configparser
import os
from pathlib import Path

os.umask(0o077)
directory = Path(os.environ["NPA_RAY_API_HOME"]) / ".aws"
directory.mkdir(mode=0o700)
credentials = configparser.ConfigParser(interpolation=None)
credentials["nebius"] = {
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
}
if os.environ.get("AWS_SESSION_TOKEN"):
    credentials["nebius"]["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
config = configparser.ConfigParser(interpolation=None)
config["profile nebius"] = {
    "region": os.environ["NPA_S3_REGION"],
    "endpoint_url": os.environ["NPA_S3_ENDPOINT"],
}
for name, value in (("credentials", credentials), ("config", config)):
    with (directory / name).open("x", encoding="utf-8") as stream:
        value.write(stream)
PY_SKY_STORAGE
SKYPILOT_USER_ID="$(npa/.venv/bin/python -c \
  'import os; from pathlib import Path; from npa.orchestration.skypilot.cleanup import sky_environment; os.environ.pop("SKYPILOT_USER_ID", None); print(sky_environment(Path(os.environ["NPA_RAY_SKY_STATE"]))["SKYPILOT_USER_ID"])')"
export SKYPILOT_USER_ID
export AWS_ENDPOINT_URL="$NPA_S3_ENDPOINT"
NPA_RAY_API_IMAGE='berkeleyskypilot/skypilot@sha256:e6cacad12f78519e0f55a33a9b38c0c9672af367749e34552889f4158b3e4b3b'
docker pull "$NPA_RAY_API_IMAGE" > "$NPA_RAY_CLIENT_ROOT/api-image-pull.log" 2>&1
docker run --detach --user "$(id -u):$(id -g)" \
  --network bridge --cap-drop ALL --security-opt no-new-privileges \
  --publish 127.0.0.1::46580 \
  --mount "type=bind,src=$NPA_RAY_API_HOME,dst=/home/sky" \
  --mount "type=bind,src=$NPA_NEBIUS_CONFIG_DIR,dst=/home/sky/.nebius,readonly" \
  --mount "type=bind,src=$NPA_KUBECONFIG_PATH,dst=/home/sky/.kube/config,readonly" \
  --mount "type=bind,src=$NPA_NEBIUS_BIN,dst=/usr/local/bin/nebius,readonly" \
  --env HOME=/home/sky --env USER --env KUBECONFIG=/home/sky/.kube/config \
  --env SKYPILOT_DISABLE_USAGE_COLLECTION=1 --env SKYPILOT_USER_ID \
  --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY --env AWS_SESSION_TOKEN \
  --env AWS_ENDPOINT_URL \
  --entrypoint tini "$NPA_RAY_API_IMAGE" \
  -- sky api start --host 0.0.0.0 --foreground \
  > "$NPA_RAY_CLIENT_ROOT/api-container.id"
read -r NPA_RAY_API_CONTAINER < "$NPA_RAY_CLIENT_ROOT/api-container.id"
export NPA_RAY_API_CONTAINER
NPA_RAY_API_BINDING="$(docker port "$NPA_RAY_API_CONTAINER" 46580/tcp)"
[[ "$NPA_RAY_API_BINDING" =~ ^127\.0\.0\.1:[0-9]+$ ]]
export SKYPILOT_API_SERVER_ENDPOINT="http://$NPA_RAY_API_BINDING"
until curl --fail --silent "$SKYPILOT_API_SERVER_ENDPOINT/api/health" \
  > "$NPA_RAY_CLIENT_ROOT/api-health.json"; do
  test "$(docker inspect --format '{{.State.Running}}' "$NPA_RAY_API_CONTAINER")" = true
  sleep 1
done
# SkyPilot chmods this installed helper before Kubernetes source syncing.
# Give only this helper to the non-root API UID; retain its exact image bytes.
NPA_RAY_API_HELPER="$(docker exec "$NPA_RAY_API_CONTAINER" python -c \
  'from pathlib import Path; import sky; print(Path(sky.__file__).parent / "utils/kubernetes/rsync_helper.sh")')"
NPA_RAY_API_HELPER_SHA_BEFORE="$(docker exec "$NPA_RAY_API_CONTAINER" sha256sum "$NPA_RAY_API_HELPER")"
NPA_RAY_API_HELPER_SHA_BEFORE="${NPA_RAY_API_HELPER_SHA_BEFORE%% *}"
docker cp "$NPA_RAY_API_CONTAINER:$NPA_RAY_API_HELPER" \
  "$NPA_RAY_CLIENT_ROOT/rsync_helper.sh"
test "$(stat -c %u "$NPA_RAY_CLIENT_ROOT/rsync_helper.sh")" = "$(id -u)"
docker cp -a "$NPA_RAY_CLIENT_ROOT/rsync_helper.sh" \
  "$NPA_RAY_API_CONTAINER:$NPA_RAY_API_HELPER"
docker exec "$NPA_RAY_API_CONTAINER" chmod 755 "$NPA_RAY_API_HELPER"
NPA_RAY_API_HELPER_SHA_AFTER="$(docker exec "$NPA_RAY_API_CONTAINER" sha256sum "$NPA_RAY_API_HELPER")"
NPA_RAY_API_HELPER_SHA_AFTER="${NPA_RAY_API_HELPER_SHA_AFTER%% *}"
test "$NPA_RAY_API_HELPER_SHA_BEFORE" = "$NPA_RAY_API_HELPER_SHA_AFTER"
printf '%s\n' "$NPA_RAY_API_HELPER_SHA_AFTER" \
  > "$NPA_RAY_CLIENT_ROOT/api-rsync-helper.sha256"
docker exec "$NPA_RAY_API_CONTAINER" /usr/local/bin/nebius --help \
  > "$NPA_RAY_CLIENT_ROOT/api-provider-cli.txt"
docker exec "$NPA_RAY_API_CONTAINER" kubectl --context "$NPA_KUBE_CONTEXT" \
  get nodes -o json > "$NPA_RAY_CLIENT_ROOT/api-kubernetes-preflight.json"
docker exec "$NPA_RAY_API_CONTAINER" sky check nebius \
  > "$NPA_RAY_CLIENT_ROOT/api-nebius-preflight.log" 2>&1
```

Verify `api-health.json` reports version `0.12.2` and healthy status. The
read-only Kubernetes query must succeed through the mounted Nebius exec-auth
configuration before submission. `sky check nebius` must explicitly report both
compute and storage enabled; a successful compute check alone is insufficient.
Neither profile presence nor bucket listing proves actual object-write access.
The published API image defaults to root; the recipe selects the operator UID
and drops capabilities. SkyPilot 0.12.2 unconditionally changes the mode of its
installed `rsync_helper.sh`, which otherwise fails for that UID even when the
file is already executable. The exact-file archive copy above changes ownership
inside this owned container while verifying unchanged bytes. It does not rebuild
the image, edit helper code, or grant ownership of the package tree.
[SkyPilot 0.12.2 Kubernetes command runner](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/utils/command_runner.py)
Preserve all preflight output privately. The container
does not receive a Docker socket or a source checkout. Its HTTP port is reachable
by local host users; this recipe assumes a trusted operator host. Secret values
are inherited by environment-variable name and never embedded in Docker argv.

The same standard workflow submission starts either session. Set
`NPA_RAY_NODES=1` and `NPA_RAY_GPUS_PER_NODE=1` for medium. For complex select
`2` nodes with `1` GPU each, or `1` node with `2` GPUs. Set `NPA_RAY_ACCELERATOR`
to the supported accelerator label and the same per-node GPU count resolved by
the target cluster, without changing the declared on-demand placement.
Include the exact run ID in the session's fresh object-key prefix. Its separate
durable workflow-state URI must end in `/<run-id>/workflow` so explicit
`status` and `cancel` resolve the same run. For example:

```bash
export NPA_RAY_PREFIX="runs/$NPA_DEV_RUN_ID/ray-development"
export NPA_WORKFLOW_URI="s3://$NPA_DEV_BUCKET/runs/$NPA_DEV_RUN_ID/workflow"
```

A custom durable prefix that omits the run ID can be accepted at submission
but rejected by the later run locator. Preserve the submission receipt and use
its exact identity if recovery is necessary; do not guess a different prefix.

```bash
set -euo pipefail
session_args=(
  npa/workflows/workbench/npa-workflows/ray-clip-development-session.yaml
  --project "$NPA_PROJECT" --run-id "$NPA_DEV_RUN_ID"
  --infra "k8s/$NPA_KUBE_CONTEXT" --isolated-config-dir "$NPA_RAY_SKY_STATE"
  --var "nodes=$NPA_RAY_NODES" --var "gpus_per_node=$NPA_RAY_GPUS_PER_NODE"
  --var "accelerator=$NPA_RAY_ACCELERATOR"
  --var "bucket=$NPA_DEV_BUCKET" --var "prefix=$NPA_RAY_PREFIX"
  --s3-endpoint "$NPA_S3_ENDPOINT"
  --durable-s3 --workflow-s3-uri "$NPA_WORKFLOW_URI"
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
  --output-format json
)
npa/.venv/bin/npa workbench workflow submit "${session_args[@]}" --plan-only \
  > "$NPA_RAY_CLIENT_ROOT/session-plan.json"
npa/.venv/bin/npa workbench workflow submit "${session_args[@]}" \
  > "$NPA_RAY_CLIENT_ROOT/session-submit.json"
```

This workflow has one rank-aware session state; standard submission with
`--durable-s3` is sufficient. It deliberately stays alive while the external
Jobs client works. Its shared submit-matrix registration is render-only because
an unattended submit test cannot coordinate Jobs and the finish marker. That
matrix entry is not a replacement for the real medium/complex checks.

## Connect through authenticated forwarding

Find this exact run's application head pod and its private head address in the
SkyPilot placement/status receipts. Do not select the first Ray pod in a shared
namespace. Verify each node's prepared receipt under the configured session
prefix; these prove preparation, not that Ray has finished registering workers.
Forward the head's loopback Dashboard through authenticated
Kubernetes access. Keep the forwarding process attached to this development
session and bound to local loopback:

```bash
set -euo pipefail
kubectl --context "$NPA_KUBE_CONTEXT" --namespace "$NPA_KUBE_NAMESPACE" \
  port-forward --address 127.0.0.1 "pod/$NPA_RAY_HEAD_POD" 18265:8265
```

Run the remaining commands in another shell with the same private configuration.
The Jobs URL is `http://127.0.0.1:18265`; the application GCS address is the
run's private head address on port 6381. Before starting the sequence, require the
Jobs API to report Ray 2.46.0 and the State API to report the expected live GPU
nodes and GPU resources. A published prepared receipt or open TCP port alone
does not satisfy this readiness check.

Run this check through the same loopback tunnel. It verifies the actual Ray
release separately from the Jobs protocol version, requires exactly one head,
and checks the GPU allocation on every live application node. The full node
receipt contains private addresses and stays in the private client directory.

```bash
set -euo pipefail
export NPA_RAY_JOBS_URL=http://127.0.0.1:18265
"$NPA_RAY_CLIENT_PYTHON" - "$NPA_RAY_JOBS_URL" \
  "$NPA_RAY_NODES" "$NPA_RAY_GPUS_PER_NODE" \
  "$NPA_RAY_CLIENT_ROOT/ray-readiness.json" <<'PY_RAY_READY'
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from urllib.parse import urlparse

import ray
from ray.job_submission import JobSubmissionClient
from ray.util.state import list_nodes
from ray.util.state.exception import RayStateApiException

os.umask(0o077)
address, node_count, gpus_per_node, receipt_path = sys.argv[1:]
expected_nodes, expected_gpus = int(node_count), int(gpus_per_node)
parsed = urlparse(address)
if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in {"", "/"} or parsed.port == 8266):
    raise ValueError("Use the loopback application Jobs tunnel")
if expected_nodes < 1 or expected_gpus < 1:
    raise ValueError("Expected application node/GPU counts must be positive")
os.environ["RAY_ADDRESS"] = address
started = time.monotonic()
observations = []
while True:
    try:
        with urllib.request.urlopen(address.rstrip("/") + "/api/version") as response:
            version = json.load(response)
        if version["ray_version"] != ray.__version__ or ray.__version__ != "2.46.0":
            raise ValueError("Application and client must both use Ray 2.46.0")
        # get_version() returns the Jobs protocol version, not the Ray release.
        jobs_protocol = JobSubmissionClient(address).get_version()
        nodes = [asdict(node) for node in list_nodes(address=address, detail=True)]
        alive = [node for node in nodes if node["state"] == "ALIVE"]
        ready = (len(alive) == expected_nodes
                 and sum(bool(node["is_head_node"]) for node in alive) == 1
                 and all(node["resources_total"].get("GPU", 0) == expected_gpus
                         for node in alive))
        if ready:
            receipt = {"ready": True, "version": version,
                       "jobs_api_version": jobs_protocol, "nodes": nodes,
                       "expected_nodes": expected_nodes,
                       "expected_gpus_per_node": expected_gpus,
                       "elapsed_seconds": time.monotonic() - started,
                       "observations": observations}
            Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"ready": True, "ray": version["ray_version"],
                              "nodes": len(alive), "gpus_per_node": expected_gpus}))
            break
        observations.append({"alive_nodes": len(alive),
                             "gpus_per_node": [node["resources_total"].get("GPU", 0)
                                               for node in alive]})
    except (OSError, RuntimeError, RayStateApiException) as error:
        observations.append({"error_type": type(error).__name__})
    time.sleep(1)
PY_RAY_READY
```

Both live sessions passed these version/node checks, with one and two physical
GPU nodes respectively. This readiness result
does not establish completed inference or source propagation; the sequence below
provides that evidence.
[Ray 2.46 Jobs version method](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/dashboard/modules/dashboard_sdk.py),
[Ray 2.46 node-state schema](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/util/state/common.py)

The session places all application ports below 11002 and application workers
on 10010–10999. SkyPilot's management head can use 11002–65535, and non-head
management workers can use OS-assigned ports; the session verifies the OS
ephemeral range begins above 10999. It also avoids the fixed management ports.
[Ray 2.9.3 worker-port allocation](https://github.com/ray-project/ray/blob/ray-2.9.3/src/ray/raylet/worker_pool.cc#L122-L138)

Keep the dedicated namespace and enforced ingress policy above in place for the
whole session. Kubernetes credentials authenticate the tunnel; Ray 2.46
namespaces and runtime environments do not provide tenant isolation.
[Ray 2.46 security model](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/ray-security/index.md)

## Execute baseline, changed source and restoration

Use the [reviewed application client](../../npa/workflows/workbench/ray-clip-development/README.md).
Its application, worker and validation files, plus the explicitly selected
Workbench UDF, travel through upstream `runtime_env.working_dir`.
The client selects the application Jobs address explicitly despite any inherited
`RAY_ADDRESS`. It uses unique submission IDs and records source hashes, status,
logs, submission and workload intervals.
[Ray 2.46 Jobs SDK](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/dashboard/modules/job/sdk.py)

For medium, set `NPA_RAY_ACTORS=1`, `NPA_RAY_RECORDS=4096`, and use the one-node
session. For complex set `NPA_RAY_ACTORS=2`, `NPA_RAY_RECORDS=16384`, and use the
session with two application GPUs. A one-node run does not establish multi-node
validation. The command runs baseline, changed crop policy, restored
source, and a separate controlled cancellation job. Every comparison uses
identical synthetic inputs, model files, image and prepared dependencies.

```bash
set -euo pipefail
export NPA_RAY_JOBS_URL=http://127.0.0.1:18265
export NPA_RAY_APP_ADDRESS="$NPA_RAY_HEAD_IP:6381"
export NPA_RAY_OUTPUT_PATH="/tmp/npa-ray-results/$NPA_DEV_RUN_ID"
"$NPA_RAY_CLIENT_PYTHON" npa/workflows/workbench/ray-clip-development/submit.py \
  --address "$NPA_RAY_JOBS_URL" --app-address "$NPA_RAY_APP_ADDRESS" \
  --udf-source npa/src/npa/workbench/lancedb/bdd100k_udfs.py \
  --python /tmp/npa-ray-app/bin/python \
  --model-path /tmp/npa-clip-model \
  --model-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --output-path "$NPA_RAY_OUTPUT_PATH" \
  --evidence-dir "$NPA_RAY_CLIENT_ROOT/workload" \
  --records "$NPA_RAY_RECORDS" --actors "$NPA_RAY_ACTORS" --batch-size 64 \
  --recovery-check --cancel-check
```

A passing medium check establishes that a source edit reached real GPU inference. The complex
check adds CPU partitions, multiple GPU actors processing concurrent batches,
Lance/Parquet aggregation and retrieval checks. Baseline/restored source crops
the left half of each rendered image; changed source crops the right half.
CPU/GPU workers verify imported source hashes, including the submitted UDF on
each model actor. The client verifies unchanged model/runtime provenance across
revisions and matches driver/actor imports against its source manifest.
Completeness checks cover every
record ID. Per-vector checks require a meaningful output change and numerical
restoration, not just a changed log message.

Actor recovery occurs after a shard commit: the client-owned application kills
one exact actor, replaces it, and checks idempotent replay of the committed
shard. The replacement reloads the model. The cancellation job first performs
actual GPU inference and then continues until the client stops its exact Job ID
and observes `STOPPED`. This does not test recovery after loss of the driver
node or restoration of GPU memory. Each new source job creates fresh actors;
model reuse is across batches within an actor, not across source redeployment.
[Ray 2.46 actor recovery](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/ray-core/fault_tolerance/actors.rst)

The session fixes Jobs drivers to the application head and the clients request
no separate entrypoint resources. This keeps baseline comparisons, shard
checkpoints and the finish upload on the same filesystem; GPU actors still run
on the allocated worker GPUs. Changing driver placement requires a shared
durable artifact design and is outside this recipe.

### Adapt an application with the upstream Jobs SDK

The qualification client automates standard Ray APIs. Customers can also submit
their own reviewed source folder directly to the already verified application
Jobs endpoint. This adaptation example is separate from the measured CLIP
workload; it does not qualify an arbitrary application or its dependencies.
Keep credentials, model weights, inputs and outputs outside the submitted folder.

In your existing `main.py`, connect explicitly before running the workload:

```python
import os
import ray

ray.init(address=os.environ["NPA_RAY_APP_ADDRESS"])
```

Set `NPA_RAY_CUSTOM_APP_DIR` to that source-only folder, keep the authenticated
loopback tunnel and session environment from above, and submit with the pinned
client. The entrypoint uses the session's prepared application interpreter:

```bash
set -euo pipefail
umask 077
"$NPA_RAY_CLIENT_PYTHON" - <<'PY_CUSTOM_RAY_JOB'
import os
from pathlib import Path
import time
import uuid
from ray.job_submission import JobSubmissionClient

source = Path(os.environ["NPA_RAY_CUSTOM_APP_DIR"]).resolve(strict=True)
assert source.is_dir() and (source / "main.py").is_file()
jobs_url = os.environ["NPA_RAY_JOBS_URL"]
# Ray 2.46's SDK reads ambient RAY_ADDRESS when selecting its Jobs endpoint.
os.environ["RAY_ADDRESS"] = jobs_url
client = JobSubmissionClient(jobs_url)
job_id = client.submit_job(
    submission_id="application-" + uuid.uuid4().hex,
    entrypoint="/tmp/npa-ray-app/bin/python main.py",
    runtime_env={
        "working_dir": str(source),
        "env_vars": {"NPA_RAY_APP_ADDRESS": os.environ["NPA_RAY_APP_ADDRESS"]},
    },
)
while True:
    status = client.get_job_status(job_id)
    if status.is_terminal():
        break
    time.sleep(1)
Path(os.environ["NPA_RAY_CLIENT_ROOT"], job_id + ".log").write_text(
    client.get_job_logs(job_id)
)
print(job_id, status)
if str(status) != "SUCCEEDED":
    raise SystemExit(1)
PY_CUSTOM_RAY_JOB
```

Edit the reviewed application source and run the same submission again: Ray
packages the changed folder under a new Jobs submission ID while the image and
prepared dependencies remain fixed. Existing actors do not acquire edited source
automatically. The measured CLIP recipe creates new actors and loads weights for
each source Job; a separately managed resident service needs its own update
contract. Preserve source and output hashes, adapt artifact validation to your
application, and follow the same finish/cancellation ordering below.
[Ray 2.46 Jobs SDK](https://github.com/ray-project/ray/blob/ray-2.46.0/python/ray/dashboard/modules/job/sdk.py),
[Ray runtime environments](https://github.com/ray-project/ray/blob/ray-2.46.0/doc/source/ray-core/handling-dependencies.rst)

## Persist outputs, finish and verify cleanup

Set `NPA_RAY_ARTIFACT_URI` to a fresh prefix under this session and
`NPA_RAY_STOP_URI` to its configured finish-marker URI, outside the artifact
prefix. The finish client sends one reviewed upload worker through Jobs,
verifies every object and the manifest by read-after-write hashes, waits for
that Job to finish, then writes the finish marker from the operator client.
It does not signal completion after an unverified artifact upload.

The finish client's operator-side marker write uses boto3 directly. Supply the
authorized `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and, when applicable,
`AWS_SESSION_TOKEN` through its private environment; set `AWS_ENDPOINT_URL` to
the selected S3 endpoint. It does not resolve the NPA project alias itself.

```bash
set -euo pipefail
export AWS_ENDPOINT_URL="$NPA_S3_ENDPOINT"
"$NPA_RAY_CLIENT_PYTHON" npa/workflows/workbench/ray-clip-development/finish.py \
  --address "$NPA_RAY_JOBS_URL" --python /tmp/npa-ray-app/bin/python \
  --output-path "$NPA_RAY_OUTPUT_PATH" \
  --artifact-uri "$NPA_RAY_ARTIFACT_URI" --stop-uri "$NPA_RAY_STOP_URI" \
  --evidence-dir "$NPA_RAY_CLIENT_ROOT/finish"
npa/.venv/bin/npa workbench workflow status "$NPA_DEV_RUN_ID" \
  --project "$NPA_PROJECT" --workflow-s3-uri "$NPA_WORKFLOW_URI" --json
```

Require live-verified workflow `SUCCEEDED` and durable
`stages.application-session.state: SUCCEEDED` with a nonempty end timestamp.
Inspect every node's cleanup receipt: finish requested, owned Ray CLI terminated,
no surviving owned children or shutdown errors. Verify
that the management runtime was not disrupted. Preserve client logs and model,
source, input/output and package receipts with the durable results. Stop the
local forwarding process and clean up only resources allocated for this run,
using the normal [cancel-before-destroy ordering](../../skills/atomic/teardown-and-cost/SKILL.md).
If the application fails, preserve diagnostics, stop/reconcile its exact Jobs,
and cancel the exact NPA workflow before infrastructure teardown:

```bash
npa/.venv/bin/npa workbench workflow cancel "$NPA_DEV_RUN_ID" \
  --project "$NPA_PROJECT" --workflow-s3-uri "$NPA_WORKFLOW_URI" --json
```

Never use `ray stop` or broad process/pod deletion on a SkyPilot node. Namespaces
and unique job IDs help target owned resources but do not replace ownership
receipts or network isolation.

After all owned jobs are terminal and the owned controller/infrastructure have
completed their normal teardown, preserve API logs and stop only the captured
container. Retain its run-owned home and evidence until they are archived:

```bash
set -euo pipefail
read -r NPA_RAY_API_CONTAINER < "$NPA_RAY_CLIENT_ROOT/api-container.id"
[[ "$NPA_RAY_API_CONTAINER" =~ ^[a-f0-9]{64}$ ]]
docker logs "$NPA_RAY_API_CONTAINER" > "$NPA_RAY_CLIENT_ROOT/api-server.log" 2>&1
docker stop "$NPA_RAY_API_CONTAINER"
docker rm "$NPA_RAY_API_CONTAINER"
```

## Measurements and support boundary

The final medium sequence processed three sets of 4,096 records on one GPU node
plus controlled cancellation in **104.720 seconds**. The final complex sequence
processed three sets of 16,384 records across two GPU nodes plus cancellation in
**156.965 seconds**. All edited vectors passed the change threshold; restored
vectors matched baseline exactly. Both sessions verified durable artifacts,
actor recovery, scoped process cleanup and live/durable workflow success.
See the [medium](../architecture/ray-fast-development-audit.md#measured-medium-result)
and [complex](../architecture/ray-fast-development-audit.md#measured-complex-result)
timing boundaries, and the retained
[earlier failures and repairs](../architecture/ray-fast-development-audit.md#earlier-attempts-and-repairs).
The failed LanceDB bootstrap is a
preparation result, not an application Ray or GPU workload result. Record image digest, exact versions,
model snapshot hash, per-worker source/module hashes, real node/GPU counts,
input/output hashes, successful record counts, recovery/cancellation receipts,
and cleanup evidence. Separate preparation, SDK packaging/upload/submission,
job startup, model load, synchronized GPU work, aggregation, artifact upload,
and total source-iteration elapsed time. Compare worker durations on each
worker's own clock; concurrent-wave observations must use one coordinator clock.
Other cancelled startup/preparation attempts also remain preparation diagnostics,
not source-iteration observations. Successful checks establish the canonical
Workbench CLIP UDF plus LanceDB library path, not the full published LanceDB
service image or its HTTP backfill API.

This path demonstrates the measured source-only application changes; it
does not prove arbitrary native dependency edits, persistent models across
source jobs, untrusted multi-tenancy, or performance against native SkyPilot
`exec`. Native `exec` already syncs workdir without provisioning/setup and is a
useful direct-script option, but it has not been measured in this experiment.
[SkyPilot 0.12.2 exec](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/execution.py#L772)

For compatible Python dependency, packaging-metadata or console-entrypoint
changes, reinstall pinned packages into the application environment or repeat its
preparation, validate the workload, and record a new dependency freeze and hash
evidence. Source copying alone does not update installed metadata or regenerate
console-script wrappers; these changes do not inherently require an image build.
[pip interpreter targeting](https://pip.pypa.io/en/stable/topics/python-option/),
[PyPA entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/)

Rebuild or select a compatible base image when changed image-owned Python,
Torch/CUDA, native libraries, system packages or baked vendor components require
it. Source copying cannot repair ABI mismatches. Retain the normal immutable-image
release validation even after a source iteration succeeds.

## Historical local container reproduction: 2026-09-04

This is the earlier measured CPU example for the [Ray and fast development audit](../architecture/ray-fast-development-audit.md).
It calls the real NPA dataset implementation to ingest, validate, curate and
query 100,000 synthetic sensor metadata records. The query uses the supported
manifest backend. It does not decode sensor payloads or invoke LanceDB,
FiftyOne, a GPU, Ray, or a remote scheduler.

Use an isolated checkout with an editable `npa/.venv` installed from
`npa[dev,adapter]`, Docker on the same host, and a clean
`npa/src/npa/workbench/dataset/curation.py`. Run from the repository root.
The experiment reads the committed module with `git show`, writes a private
copy, and changes its quality comparison from `<` to `<=`. A nested read-only
mount selects that copy inside the container. The checkout is never edited;
the runner removes the private override in `finally`. Keep the source unchanged
while comparing the runs so each version has a stable input tree.

### Prepare the existing image and dependency directory

The image below is an existing published release artifact, pinned by digest.
It supplies Python 3.11 and the control runtime, but its intentionally narrow
package set lacks Pydantic. Install that dependency **once**, inside the same
image/interpreter, into an isolated directory. The benchmark keeps those bytes
fixed across iterations; no image is built. For a different workload use its
compatible workbench image and dependency set. A successful source mount does
not validate a new native dependency or GPU ABI.

```bash
set -euo pipefail
umask 077
export NPA_DEV_ROOT="$(mktemp -d)"
export NPA_DEV_IMAGE='ghcr.io/nebius/nebius-physical-ai/npa-sim2real-control@sha256:87fe8530710eea43364a21ad76dbe4b4c2d60e4b49705824fcdb62dc7d185af7'
mkdir -p "$NPA_DEV_ROOT/deps" "$NPA_DEV_ROOT/output"
git diff --exit-code -- npa/src/npa/workbench/dataset/curation.py
git diff --cached --exit-code -- npa/src/npa/workbench/dataset/curation.py

docker pull "$NPA_DEV_IMAGE"
docker run --rm --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --mount "type=bind,src=$NPA_DEV_ROOT/deps,dst=/deps" \
  --entrypoint python3 "$NPA_DEV_IMAGE" \
  -m pip install --disable-pip-version-check --no-cache-dir \
  --target /deps pydantic==2.12.5 > "$NPA_DEV_ROOT/dependency-setup.log" 2>&1
```

Image pull and dependency installation are preparation, excluded from the
iteration timings. The dependency log records the resolved transitive versions;
archive it with the results. Promotion still needs the normal locked image build
and validation process. These containers receive no cloud credentials, no Docker
socket, no host home directory, and no network during workload execution. Source
and dependencies are read-only; only the output directory is writable.

### Write the workload and iteration runner

The workload verifies the imported file hash, all selected record IDs against an
independent predicate, the validation result, and child lineage. Artifact hashes
cover actual serialized files; they are distinct from NPA's manifest identity.

```bash
set -euo pipefail
cat > "$NPA_DEV_ROOT/workload.py" <<'PY_WORKLOAD'
import hashlib
import json
import sys
import time
from pathlib import Path

from npa.workbench.dataset import curation
from npa.workbench.dataset.ingestion import ingest_dataset
from npa.workbench.dataset.schemas import CurateRequest, IngestRequest, QueryRequest, ValidateRequest
from npa.workbench.dataset.validation import validate_manifest

started = time.perf_counter()
label, expected_hash = sys.argv[1:]
module = Path(curation.__file__)
module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
assert module_hash == expected_hash, (module, module_hash, expected_hash)
root = Path('/output') / label
root.mkdir()
raw = [dict(record_id=f'frame-{i:06d}', modality='camera',
            uri=f'/synthetic/frame-{i:06d}.png',
            event='cut_in' if i % 2 == 0 else 'cruise',
            location='synthetic_west' if (i // 2) % 2 == 0 else 'synthetic_east',
            timestamp=f'{i / 30:.6f}', quality={'confidence': ((i // 4) % 10) / 10})
       for i in range(100000)]
input_path = root / 'raw.json'
input_path.write_text(json.dumps({'records': raw}))
ingested = ingest_dataset(IngestRequest(input_uri=str(input_path), output_uri=str(root / 'dataset'),
                                       dataset_id='synthetic-sensors', source='synthetic', version='v1'))
assert ingested.record_count == len(raw)
validated = validate_manifest(ValidateRequest(input_uri=ingested.manifest_uri,
                                             output_uri=str(root / 'validation'), completeness_min=0.7))
assert validated.passed and validated.record_count == len(raw)
curated = curation.curate_dataset(CurateRequest(input_uri=ingested.manifest_uri,
    output_uri=str(root / 'curated'), event='cut_in', location='synthetic_west',
    quality_metric='confidence', min_quality=0.5))
queried = curation.query_dataset(QueryRequest(input_uri=curated.manifest_uri, limit=len(raw)))
strict = label == 'changed'
expected = [r['record_id'] for r in raw if r['event'] == 'cut_in'
            and r['location'] == 'synthetic_west'
            and (r['quality']['confidence'] > 0.5 if strict else r['quality']['confidence'] >= 0.5)]
assert [r['record_id'] for r in queried.records] == expected
assert curated.record_count == len(expected) == (10000 if strict else 12500)
child = json.loads(Path(curated.manifest_uri).read_text())
assert child['lineage']['parent_dataset_id'] == 'synthetic-sensors'
assert child['lineage']['parent_version'] == 'v1'
artifacts = {str(p.relative_to(root)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                     'bytes': p.stat().st_size} for p in sorted(root.rglob('*.json'))}
result = dict(label=label, input_records=len(raw), selected_records=len(expected),
              module_path=str(module), module_sha256=module_hash,
              workload_seconds=round(time.perf_counter() - started, 6),
              validation_passed=validated.passed, query_backend=queried.backend,
              selected_ids_sha256=hashlib.sha256('\n'.join(expected).encode()).hexdigest(),
              artifacts=artifacts)
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
PY_WORKLOAD
```

The runner starts a fresh container for each source version, passes the expected
source hash, measures container wall time, and removes the private override even when a
workload assertion fails. Its result distinguishes wall time from the timed
workload, which includes synthetic input generation and artifact serialization.

```bash
set -euo pipefail
cat > "$NPA_DEV_ROOT/run.py" <<'PY_RUNNER'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

repo = Path.cwd()
root = Path(__file__).resolve().parent
image = 'ghcr.io/nebius/nebius-physical-ai/npa-sim2real-control@sha256:87fe8530710eea43364a21ad76dbe4b4c2d60e4b49705824fcdb62dc7d185af7'
source = repo / 'npa/src/npa/workbench/dataset/curation.py'
original = subprocess.check_output(['git', 'show', 'HEAD:npa/src/npa/workbench/dataset/curation.py'])
assert source.read_bytes() == original, 'Use a clean target source file'
override = root / 'curation-override.py'
needle = b'_record_metric(record, quality_metric) < min_quality:'
assert original.count(needle) == 1
changed = original.replace(needle, b'_record_metric(record, quality_metric) <= min_quality:')
results = []
try:
    for label, content in [('baseline', original), ('changed', changed), ('restored', original)]:
        override.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        cmd = ['docker', 'run', '--rm', '--network', 'none', '--read-only',
               '--user', f'{os.getuid()}:{os.getgid()}', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges',
               '--mount', f'type=bind,src={repo / "npa/src"},dst=/src,readonly',
               '--mount', f'type=bind,src={override},dst=/src/npa/workbench/dataset/curation.py,readonly',
               '--mount', f'type=bind,src={root / "deps"},dst=/deps,readonly',
               '--mount', f'type=bind,src={root / "output"},dst=/output',
               '--mount', f'type=bind,src={root / "workload.py"},dst=/workload.py,readonly',
               '--env', 'PYTHONPATH=/src:/deps', '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--entrypoint', 'python3', image, '/workload.py', label, expected]
        start = time.perf_counter()
        run = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - start
        (root / f'{label}.log').write_text(run.stdout + run.stderr)
        results.append({'label': label, 'exit_code': run.returncode, 'container_wall_seconds': round(wall, 6)})
        if run.returncode:
            raise RuntimeError(f'{label} failed; see private log')
        results[-1].update(json.loads(run.stdout))
        print(json.dumps({k: v for k, v in results[-1].items() if k != 'artifacts'}), flush=True)
finally:
    override.unlink(missing_ok=True)
    assert source.read_bytes() == original, 'Host source changed during run'
    (root / 'workload-evidence.json').write_text(json.dumps({'image': image, 'runs': results,
        'host_source_unchanged': source.read_bytes() == original, 'container_builds': 0,
        'data': '100000 synthetic sensor metadata records; sensor bytes are not read',
        'execution': 'local Docker on operator VM; manifest backend; no GPU or remote scheduler'}, indent=2) + '\n')
PY_RUNNER
npa/.venv/bin/python "$NPA_DEV_ROOT/run.py"
git diff --exit-code -- npa/src/npa/workbench/dataset/curation.py
```

Expected counts are **12,500 → 10,000 → 12,500**, with identical baseline and
restored module/selected-ID hashes. Inspect `workload-evidence.json`, each
`output/*/result.json`, the dataset manifests, and validation reports beneath
`NPA_DEV_ROOT`. Containers remove themselves on completion. Keep the evidence
until reviewed, then remove only this experiment's temporary directory.

### Observed execution, 2026-09-04

| Source | Input records | Selected records | Workload seconds | Container wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 100,000 | 12,500 | 5.011 | 5.653 |
| Changed comparison | 100,000 | 10,000 | 5.918 | 6.893 |
| Restored | 100,000 | 12,500 | 5.942 | 6.905 |

The release manifest was independently fetched anonymously from GHCR (HTTP
200), its SHA-256 matched the requested digest, and Docker pulled it successfully
before the measured runs. Runtime versions were Python 3.11.15, Pydantic 2.12.5,
pydantic-core 2.41.5, httpx 0.28.1, and boto3 1.43.62.

All three containers exited **0**, all validation reports passed, and the
host source stayed unchanged byte-for-byte. Each container imported
`/src/npa/workbench/dataset/curation.py`. Baseline/restored module SHA-256:
`77de310fc9af3fa415f001c2e622e69a7bbf5f155b631b2ba6a7a33e46d5362d`;
changed module SHA-256:
`870dd61f4d7ed842f224151f92c416d480852b8a2e010c74711595fb4236d3dd`.
The selected-ID digest also returned to its original value after restoration.

These are three observations on one operator VM with the image already cached,
not a throughput study or a measured speedup over rebuilding. There were **zero
container builds**. The experiment proves changed NPA Python executed against
unchanged container and dependency bytes; it does not measure S3 transfer,
SkyPilot scheduling, Ray performance, cold image pulls, or model loading.
