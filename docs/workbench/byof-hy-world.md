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

No candidate image digest exists yet. The Dockerfile is a neutral CUDA bootstrap
and the planned eight-B200 profile is a future qualification target, not a
B200 compatibility statement. A governed build must use a full-40-character
`dev-<sha>` tag, scan the exact image, and use its resulting immutable digest in
the workflow; it must not be publicly pushed or promoted from this candidate.

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
improve another AI model. Before Tencent bytes are fetched, a private,
run-scoped operator record must resolve the territory/MAU condition or cite a
Tencent grant. There is intentionally no `ACCEPT_TERMS` flag.

`facebook/sam3` is a separate gated Meta/Hugging Face access decision: the
operator must complete its provider-side terms/contact flow and verify access to
the exact pinned revision. NPA's current `health access` catalog has no
`hy-world` capability, so it cannot stand in for either Tencent eligibility or
the SAM 3 approval. Do not invent an acceptance flag or treat an unrelated HF
probe as proof of either fact.

After the private record and provider approvals exist, select the actual run
project *before* testing credentials. This makes the S3 probe use that
project's configured bucket and credentials rather than an ambient default:

```bash
PROJECT="<configured-project-alias>"
npa workbench health preflight --project "$PROJECT" \
  --checks nebius,hf,s3 --json
npa workbench workflow validate-spec workflows/testing/byof-hy-world.yaml --json
```

The first command verifies the selected control-plane identity, Hugging Face
credential, and selected-project S3 access. The second is offline declaration
validation. Neither checks Tencent eligibility, approves SAM 3, builds an
image, reserves a GPU, or proves model execution.

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
served identity below. This is an upstream vLLM command, not an NPA publication
or a claim that NPA has provisioned the host. It requires an operator-owned CUDA
environment with the upstream `vllm` CLI installed and Hugging Face access to
the pinned Qwen model; those are the endpoint owner's normal model-serving
prerequisites, separate from Tencent and Meta approval:

```bash
MODEL="Qwen/Qwen3-VL-8B-Instruct"
REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
vllm serve "$MODEL" --revision "$REVISION" \
  --served-model-name "$MODEL" --host 0.0.0.0 --port 8000 \
  --trust-remote-code
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
