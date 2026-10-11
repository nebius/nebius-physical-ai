# HY-World 2.0 image-to-world candidate

[Workflow](../../workflows/testing/byof-hy-world.yaml) · [solution catalog](oss-solution-catalog.md#tencent-hy-world-20-terms-gated-image-to-world-candidate) · [BYOF onboarding](../../skills/workflows/byof-onboard/SKILL.md) · [redistribution record](../../npa/docker/workbench/hy-world/REDISTRIBUTION.md)

This is a Tier-1 BYOF **candidate**, operated only through the normal
`npa workbench workflow` / SkyPilot control plane. It adds no service, tool
namespace, or custom controller. The supported route, when its prerequisites
are met, is one private PNG → Qwen-image HY-Pano → WorldNav → WorldStereo →
GS-data preparation → `world_gs_trainer`. It then validates the generated
WorldStereo video, cameras, PLY/SPZ scene, rendered camera video, provenance,
and a Rerun `.rrd` created from those generated assets.

It is not an accepted image or a claim of live GPU compatibility. Text-to-world
or text-to-panorama, reconstruction-only WorldMirror results, calibrated metric
scale, collision validity, and robot-policy simulation are outside this
candidate. Coordinates are the upstream scene frame and scale is uncalibrated.

## Status and exact identities

No public candidate image digest exists. A privately delivered neutral bootstrap
has passed its governed private delivery scans and is retained only in restricted
operator evidence; it remains quarantined and is not a public release. The
eight-B200 profile still needs real GPU qualification, so neither the private
bootstrap nor this guide is a B200 execution claim. Any future governed build
uses a full-40-character `dev-<sha>` tag, scans the exact image, and uses its
resulting immutable digest in the workflow; it must not be publicly pushed or
promoted from this candidate.

| Component | Exact identity | Delivery |
| --- | --- | --- |
| HY-World source | `Tencent-Hunyuan/HY-World-2.0@df9988efb87bfc0f4947eb3889411cf957478b06` | operator runtime fetch |
| HY-World weights | `tencent/HY-World-2.0@d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0` | operator runtime fetch |
| WorldStereo | `hanshanxue/WorldStereo@ac2ad97ecb043fe80c2f19cd1898006becb9d66e` | operator runtime fetch |
| Image panorama | `Qwen/Qwen-Image-Edit-2509@d3968ef930e841f4c73640fb8afa3b306a78167e` | operator runtime fetch |
| Trajectory VLM | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | operator-private vLLM, started with the exact recipe below |
| Trajectory components | ZIM `667e2d7…`; Grounding DINO `a2bb814…`; SAM 3 `3c879f3…`; MoGe `cb0e8bb…`; Uni3C `fca895f…` | operator runtime fetch |
| Native runtime dependencies | PyTorch3D `88e182f989c80836f4bd744e0d9cb1852762ce01`; `flash-attn==2.8.3` | compiled in operator cache |

The image contains none of Tencent source, weights, runtime source tree,
extension binary, input, cache, or generated output. A run records the selected
image digest, source/component revisions, resolved package inventory, camera
and generated-asset hashes in `hy_world_image_to_world.json`.

## Prerequisites before a build or submission

Tencent's HY-WORLD 2.0 Community License Agreement dated 2026-04-15 covers the
source, algorithms, inference/training code, and weights. It excludes the EU,
UK, and South Korea; an entity over one million MAU at release needs a Tencent
licence; and Section 5(b) prohibits using HY works or outputs to train or
improve another AI model. Use is the documented Tencent acceptance mechanism.
NPA therefore has no `ACCEPT_TERMS` flag and does not collect a duplicate
territory/MAU attestation or claim that a workflow run creates a Tencent grant.
The operator remains responsible for using the components and outputs under the
Agreement; this workflow has no trainer or policy consumer for its outputs.

`facebook/sam3` is a separate gated Meta/Hugging Face access decision: the
operator must complete its provider-side terms/contact flow and verify access to
the exact pinned revision. NPA's current `health access` catalog has no
`hy-world` capability, so it cannot stand in for exact runtime component access
or the SAM 3 approval. Do not invent an acceptance flag or treat an unrelated
HF probe as proof of either fact.

After exact provider access is available, select the actual run project *before*
testing credentials. This makes the S3 probe use that
project's configured bucket and credentials rather than an ambient default:

```bash
PROJECT="<configured-project-alias>"
npa workbench health preflight --project "$PROJECT" \
  --checks nebius,hf,s3 --json
npa workbench workflow validate-spec workflows/testing/byof-hy-world.yaml --json
```

The first command verifies the selected control-plane identity, Hugging Face
credential, and selected-project S3 access. The second is offline declaration
validation. Neither proves exact runtime component access, builds an image,
reserves a GPU, or proves model execution.

### Build and privately deliver the neutral bootstrap

After the legal/provider gates above, the checked-in helper can create the
workflow's private digest without exposing it publicly. It archives exactly the
committed source SHA (so dirty checkout files never enter the Docker context),
builds the neutral bootstrap, runs payload/history, Trivy vulnerability/license
and all-severity secret scans, writes an SPDX SBOM, pushes only to the explicitly
asserted operator-private registry, resolves its digest, and rescans that exact
remote digest. It refuses the official GHCR namespace and never promotes a
release.

The receipt directory is private operator evidence: it contains registry
coordinates and scanner results. Keep it outside the checkout and give it mode
`0700`. This transaction does not fetch Tencent bytes; the build environment
strips model and endpoint credentials.

Authenticate the build host to the exact private registry through its ordinary
Docker credential flow before starting. The helper's `crane` operations reuse
that host credential only to push and pull the immutable digest; scanner
containers receive an archive of the resolved digest, never a credential mount
or registry secret.

```bash
SOURCE_SHA="$(git rev-parse HEAD)"
TAG="dev-${SOURCE_SHA}"
PRIVATE_RECEIPTS="/absolute/private/hy-world-image-receipts"
mkdir -p "$PRIVATE_RECEIPTS"
chmod 0700 "$PRIVATE_RECEIPTS"

npa/docker/workbench/hy-world/build.sh \
  --push --operator-private \
  --registry "<operator-private-registry>/<namespace>" \
  --receipt-dir "$PRIVATE_RECEIPTS" --tag "$TAG"

PRIVATE_RECEIPT="$PRIVATE_RECEIPTS/hy-world-${TAG}.json"
CANDIDATE_IMAGE="$(npa/.venv/bin/python - "$PRIVATE_RECEIPT" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["immutable_image"])
PY
)"
```

`CANDIDATE_IMAGE` is now the immutable private digest for planning and image
preflight. Do not derive it from a mutable tag or scrape build output. Retain the
source-SHA receipt, both payload reports, Trivy reports, and SBOM for the later
exact-digest GPU qualification. A failed pre-push gate pushes nothing; a failed
post-push scan leaves an operator-private validation candidate that must not be
submitted until it is investigated. This path is not public publication, does
not change the candidate's quarantine, and does not bypass the Tencent/SAM 3
prerequisites.

### Private vLLM dependency

HY-World's [released trajectory code](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/worldgen/traj_generate.py)
constructs `OpenAI(api_key="EMPTY", base_url="http://ADDR:PORT/v1")` and makes
multimodal chat-completions requests. Use the ordinary upstream `vllm serve`
surface in an operator-owned CUDA serving workload; do **not** create a
HY-specific controller or ask the HY workflow to manage a persistent service.
This deliberately follows the existing [self-hosted vLLM workflow
pattern](../../workflows/testing/vlm-eval-single.yaml): `vllm serve`, an exact
served model identity, a readiness check, and owner-managed teardown. That
standard pattern starts a server local to its own evaluation run, so it cannot
be borrowed as a cross-run service. For HY, the endpoint owner starts or reuses
the private server below and records its own lifecycle/cleanup receipt.

On that private serving workload, use the pinned model revision and the exact
served identity below. It needs Hugging Face access to the pinned Qwen model;
that is an endpoint-owner prerequisite, separate from Tencent and Meta
approval. The upstream command uses the released HY-World serving parameters:

```bash
MODEL="Qwen/Qwen3-VL-8B-Instruct"
REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
vllm serve "$MODEL" --revision "$REVISION" \
  --served-model-name "$MODEL" --host 0.0.0.0 --port 8000 \
  --api-key EMPTY --max-model-len 32768 --gpu-memory-utilization 0.80
```

For a fresh endpoint, the verified portable route is an ordinary long-running
NPA managed job submitted from a raw SkyPilot YAML. It is not SkyServe: the
YAML has no `service:` section, replica policy, public port, or new controller.
The selected Kubernetes context must be private and the job's Pod IP is used
only inside that cluster. Use a distinct task-owned NPA/SkyPilot state directory
for this job and preserve it for status, logs, cancellation, and cleanup.

Save this file outside source control, after substituting only the selected
private Kubernetes context through `--var`. `HF_TOKEN` travels through the
standard secret channel; it is not in the YAML, image, or command history.

```bash
PROJECT="<configured-project-alias>"
K8S_CONTEXT="<verified-private-kubernetes-context>"
VLLM_RUN_ID="hy-world-qwen3-vl-$(date -u +%Y%m%dT%H%M%SZ)"
umask 077
SERVICE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hy-world-qwen3-vl.XXXXXX")"
chmod 700 "$SERVICE_DIR"
SERVICE_SPEC="$SERVICE_DIR/service.yaml"

cat >"$SERVICE_SPEC" <<'YAML'
name: hy-world-qwen3-vl
resources:
  cloud: kubernetes
  region: ${K8S_CONTEXT}
  accelerators: H200:1
  cpus: 8
  memory: 64
  image_id: vllm/vllm-openai@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c
envs:
  VLLM_API_KEY: EMPTY
  VLLM_GPU_MEMORY_UTILIZATION: "0.80"
  VLLM_MAX_MODEL_LEN: "32768"
  VLLM_MODEL: Qwen/Qwen3-VL-8B-Instruct
  VLLM_MODEL_REVISION: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
run: |
  exec vllm serve "$VLLM_MODEL" --revision "$VLLM_MODEL_REVISION" \
    --served-model-name "$VLLM_MODEL" --host 0.0.0.0 --port 8000 \
    --api-key "$VLLM_API_KEY" --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
YAML

npa workbench health preflight --project "$PROJECT" --checks nebius,hf --json
npa workbench workflow submit "$SERVICE_SPEC" --run-id "$VLLM_RUN_ID" \
  --project "$PROJECT" --infra "k8s/$K8S_CONTEXT" \
  --var "K8S_CONTEXT=$K8S_CONTEXT" --secret-env HF_TOKEN --output-format json
npa workbench workflow status "$VLLM_RUN_ID" --project "$PROJECT" --json
npa workbench workflow logs "$VLLM_RUN_ID" --project "$PROJECT" --follow
```

Wait for the managed-job logs to show the exact model revision loaded and the
OpenAI server listening. Then use the exact task-owned Pod named by that job,
not a label search or another workload's Pod. A transient local port-forward is
private to the operator process and is useful for the required endpoint probes;
it is not a public service exposure. In one terminal:

```bash
KUBECONFIG="<task-owned-kubeconfig>"
POD_NAME="<exact-vllm-pod-from-managed-job-status>"
kubectl --kubeconfig "$KUBECONFIG" --context "$K8S_CONTEXT" \
  port-forward "pod/$POD_NAME" 8000:8000
```

In a second terminal, first verify the exact model identity, then make a small
image-bearing request. The request exercises the same multimodal OpenAI shape
that HY uses. The final HY workflow independently checks `/v1/models` from its
own worker network before generation; a local port-forward is not a substitute
for that runtime reachability check.

```bash
SMOKE_IMAGE="<small-private-png-path>"
SERVICE_PROBE_JSON="$SERVICE_DIR/probe.json"

curl --fail --silent --show-error \
  --header 'Authorization: Bearer EMPTY' \
  http://127.0.0.1:8000/v1/models

npa/.venv/bin/python - "$SMOKE_IMAGE" >"$SERVICE_PROBE_JSON" <<'PY'
import base64
import json
import sys

encoded = base64.b64encode(open(sys.argv[1], "rb").read()).decode("ascii")
payload = {
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
        {"type": "text", "text": "Describe this image in one sentence."},
    ]}],
    "max_tokens": 32,
}
print(json.dumps(payload))
PY

curl --fail --silent --show-error \
  --header 'Authorization: Bearer EMPTY' \
  --header 'Content-Type: application/json' \
  --data-binary "@$SERVICE_PROBE_JSON" \
  http://127.0.0.1:8000/v1/chat/completions
```

Retain the NPA submission/status/log receipts and the redacted model/probe
responses in private evidence. Keep the endpoint alive through the HY workflow,
then cancel the exact managed job before deleting the local temporary files. Do
not run `cluster down`, remove a shared node group, or stop another task's
endpoint:

```bash
npa workbench workflow cancel "$VLLM_RUN_ID" --project "$PROJECT" --json
rm -f "$SERVICE_SPEC" "$SERVICE_PROBE_JSON"
rmdir "$SERVICE_DIR"
```

The endpoint must be private and reachable from the HY worker. Its `/v1/models`
response must list exactly `Qwen/Qwen3-VL-8B-Instruct`; its chat-completions
surface must accept the upstream `Bearer EMPTY` credential shape. Record the
server's model revision and owning lifecycle/cleanup record in private operator
evidence. The workflow now checks `/v1/models` with that exact identity before
it downloads the PNG; it bypasses ambient proxy variables and reports no
endpoint value.

Set only the hostname or address in the submitting environment, not a URL or a
credential, and pass it through the standard SkyPilot secret channel. The port
is the `llm_port` workflow value (default `8000`); the model name is fixed by the
pinned capability and cannot be overridden by a workflow variable:

```bash
export NPA_HY_WORLD_LLM_ADDR="<worker-reachable-vllm-host-or-address>"
```

### Task-owned B200:8 worker group on a verified Kubernetes target

The HY generation workflow needs the exact one-node `B200:8` envelope; an
H200 used for the private Qwen endpoint is a separate dependency and must not
be repurposed. Before creating capacity, select an already verified NPA
Kubernetes target in the selected project, inspect the provider compatibility
matrix for that target's Kubernetes version, and select its existing fabric
GPU-cluster attachment. Quota or a running node is not evidence that an
additional B200 group can be placed.

The ordinary NPA node-group lifecycle creates a task-owned additive group. It
does not create a cluster, modify the shared control plane, network, GPU
cluster, or any existing node group. The supported B200 profile is one
`gpu-b200-sxm` node with the `8gpu-160vcpu-1792gb` preset, managed `cuda13.0`
drivers, the verified fabric attachment, and a 1280-GiB boot disk. A provider
boot-disk size is not Kubernetes allocatable ephemeral storage. Before launch,
the selected B200 node must report at least `1000000000000` allocatable and
currently free ephemeral-storage bytes for the workflow's 1000G request; the
normal Workbench capacity preflight re-reads that fit at submission time. This
is not a claim that the model has already run.

Run this through the same task-owned NPA configuration/context that owns the
workflow. Do not use another image's controller, journal, kubeconfig, or
cluster alias. The values below are placeholders, not infrastructure values to
place in source control:

```bash
PROJECT_ID="<selected-project-id>"
PROJECT="<configured-project-alias>"
SPEC="workflows/testing/byof-hy-world.yaml"
HY_CLUSTER_TARGET="<verified-npa-kubernetes-target>"
GPU_CLUSTER_ID="<verified-project-gpu-cluster-id>"
B200_NODE_GROUP="hy-world-b200-8"

npa cluster node-group add \
  --cluster-name "$HY_CLUSTER_TARGET" --name "$B200_NODE_GROUP" \
  --project-id "$PROJECT_ID" --gpu-type b200 --node-count 1 \
  --node-preset 8gpu-160vcpu-1792gb --boot-disk-size-gib 1280 \
  --gpu-cluster-id "$GPU_CLUSTER_ID" \
  --gpu-driver-mode managed-image --managed-driver-preset cuda13.0 --wait

npa cluster node-group status --cluster-name "$HY_CLUSTER_TARGET" \
  --name "$B200_NODE_GROUP" --project-id "$PROJECT_ID" --format json
npa workbench workflow gpus --project "$PROJECT" --cluster "$HY_CLUSTER_TARGET" \
  --spec "$SPEC" --json
```

The command refuses an unattached B200 group, another preset/count, a disk
below 1280 GiB, `--no-wait`, and the unsafe operator-driver route. It also
verifies that the provider reports the requested GPU-cluster attachment and
disk before recording the group as ready. If either readiness or GPU discovery
fails, retain the returned provider reason; do not change the cluster, network,
or an unrelated node group to make the workflow fit.

Cancel a run before removing only the group named above. This removes the
task-owned node group and its matching local NPA state; it never removes the
shared cluster, its control-plane network, the fabric GPU cluster, the H200
endpoint, or another operator's groups:

```bash
npa workbench workflow cancel "<run-id>" --project "$PROJECT" --json
npa cluster node-group remove --cluster-name "$HY_CLUSTER_TARGET" \
  --name "$B200_NODE_GROUP" --project-id "$PROJECT_ID" --force
```

Keep the node group while a run, artifact collection, or factual output review
is still active. Record the standard cancellation and removal receipts before
declaring the task-owned B200 capacity cleaned up.

## Standard Workbench workflow

Choose a project alias, a writable private bucket, a nonempty prefix, one
private PNG object, and an immutable **private** candidate digest. Do not put a
registry endpoint, provider token, or vLLM hostname into the YAML or source
control. This example uses placeholders deliberately:

```bash
SPEC="workflows/testing/byof-hy-world.yaml"
PROJECT="<configured-project-alias>"
BUCKET="<private-bucket>"
PREFIX=operator-runs/hy-world
INPUT_IMAGE_URI="s3://${BUCKET}/inputs/scene.png"

# CANDIDATE_IMAGE was read from the structured private-build receipt above.
# This value is retained only in the SkyPilot secret channel.
export NPA_HY_WORLD_LLM_ADDR="<worker-reachable-vllm-host-or-address>"
```

Plan exactly the run configuration, without allocating infrastructure:

```bash
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" --run-id hy-world-plan \
  --var bucket="$BUCKET" --var prefix="$PREFIX" \
  --var input_image_uri="$INPUT_IMAGE_URI" --var base_image="$CANDIDATE_IMAGE" \
  --json
```

The specification declares `metadata.executionMode: runtime`. Thus the generic
Workbench submit surface selects the durable runtime even when an agent omits
`--runtime`, and it rejects `--no-runtime`; the explicit `--runtime` below is
only documentary redundancy, not a hidden required flag.

The resolved `--output-root` is `s3://$BUCKET/$PREFIX`. The standard BYOF
runner appends the run ID once, so the declared and uploaded artifacts are both
under `s3://$BUCKET/$PREFIX/<run-id>/`. Overriding `prefix` therefore changes
the real upload location and every output declaration together.

Prepare a durable run ID, then prove the selected immutable image is pullable
with that project before submitting. Record the `run_id` returned by
`prepare-run --json`; do not manufacture one by parsing human output.

```bash
npa workbench workflow prepare-run "$SPEC" --project "$PROJECT" --json
npa workbench workflow preflight-images "$SPEC" --project "$PROJECT" \
  --var bucket="$BUCKET" --var prefix="$PREFIX" \
  --var input_image_uri="$INPUT_IMAGE_URI" --var base_image="$CANDIDATE_IMAGE" \
  --json
```

Only after all legal/access/image gates pass, submit through the ordinary
runtime. `HF_TOKEN` is included only when the operator needs it for approved
runtime component fetches; it is never written into the plan or image.

```bash
npa workbench workflow submit "$SPEC" --project "$PROJECT" --run-id "<run-id>" \
  --runtime --var bucket="$BUCKET" --var prefix="$PREFIX" \
  --var input_image_uri="$INPUT_IMAGE_URI" --var base_image="$CANDIDATE_IMAGE" \
  --secret-env NPA_HY_WORLD_LLM_ADDR --secret-env HF_TOKEN \
  --output-format json
```

The stage first checks the neutral bootstrap, then downloads and decodes the
declared PNG into disposable stage-local storage, then executes the actual
upstream sequence. The OCI byte/layer scanner is separate from mutable inputs
and model caches. A warmed compatible runtime cache is reused; its component
closure gets a private, revision-keyed Hugging Face cache, avoiding shared
`refs/main` writes. A legacy complete cache is revalidated before reuse.

## Monitor, inspect, resume, and clean up

All runtime control remains with the standard Workbench surface:

```bash
npa workbench workflow status "<run-id>" --project "$PROJECT" --watch --json
npa workbench workflow logs "<run-id>" --stage generate --project "$PROJECT" --follow
npa workbench workflow artifacts "<run-id>" --project "$PROJECT" --json
```

If normal discovery cannot locate a run, use the exact durable workflow URI
returned by the submission receipt with `--workflow-s3-uri`; do not guess a
controller, bucket, or prefix. The artifacts command lists the evidence JSON,
generated scene assets, and the Rerun recording. Open the returned
`.rrd` in an authorized Rerun viewer; it is a factual visualization of the
validated run assets, not a collision or robotics simulation viewer.

For a transient controller interruption, resume the recorded run with the same
project, config values, image digest, and secrets:

```bash
npa workbench workflow submit "$SPEC" --project "$PROJECT" \
  --resume-run "<run-id>" --runtime --resume \
  --var bucket="$BUCKET" --var prefix="$PREFIX" \
  --var input_image_uri="$INPUT_IMAGE_URI" --var base_image="$CANDIDATE_IMAGE" \
  --secret-env NPA_HY_WORLD_LLM_ADDR --secret-env HF_TOKEN
```

Use `--retry-absent-in-flight` only when the runtime has established that the
previous managed job and every declared output are absent. To stop a live run,
use the run-scoped, repeat-safe cancellation path:

```bash
npa workbench workflow cancel "<run-id>" --project "$PROJECT" --json
```

This terminal workflow creates no persistent service. Its owned durable outputs
remain at the selected S3 prefix for inspection; delete or retain them under the
operator's storage-retention policy only after cancellation is confirmed. Never
prune shared caches, Docker resources, controllers, or another run's artifacts.

For programmatic read-only monitoring, use the existing SDK rather than a HY
specific client:

```python
from npa.sdk.workbench.workflow import artifacts, logs, status

run = status("<run-id>", project="<configured-project-alias>")
evidence = artifacts("<run-id>", project="<configured-project-alias>")
stage_log = logs("<run-id>", stage="generate", project="<configured-project-alias>")
```

## Outputs and limits

The primary evidence is `hy_world_image_to_world.json`. It binds the source and
model revisions, selected immutable image, resolved packages/submodules and
hashes for the panorama, WorldStereo/rendered videos, cameras, PLY/SPZ and
Rerun report. The report is emitted only after the decoder, finite-camera and
scene-asset validators succeed. A placeholder, echoed manifest, generic mesh,
or multi-view reconstruction cannot satisfy this contract.

Generated assets can be handed to consumers that accept the declared camera,
scene and Rerun artifact schemas, but the coordinate system remains upstream
world space and uncalibrated. The workflow does not generate collision meshes
for robotic use and must not be used to claim physical validity, policy safety,
or model quality. These remain separate work with separate evidence.
