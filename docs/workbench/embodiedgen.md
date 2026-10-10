# EmbodiedGen V2: image to rigid object

EmbodiedGen is a Tier-1, operator-private BYOF workflow, not a persistent
service or a dedicated Workbench subcommand. The reusable control-plane surface
is [`byof-embodiedgen.yaml`](../../workflows/testing/byof-embodiedgen.yaml):
the generic `npa workbench workflow` CLI plans, renders, submits, monitors, and
cancels it through SkyPilot; the generic workflow SDK reads its durable state.
This avoids a parallel orchestration path while keeping the workflow usable by a
human or an agent.

The only candidate capability is
`img3d-cli_trellis_image_to_urdf_pybullet`: upstream EmbodiedGen V2.1.0's
`img3d-cli` runs the TRELLIS backend on one image, exports a mesh/URDF and
collision meshes, converts the URDF to MJCF, then loads that *exact generated
URDF* in PyBullet to prove gravity contact, settling, and a decoded view. It
does not claim articulated-object generation, calibrated physical properties,
simulator validation beyond PyBullet, or policy improvement. The VLM-derived
mass, height, and friction fields are estimates, not ground truth.

See the [testing-workflow catalog](../../workflows/testing/README.md#bring-your-own-framework),
[OSS capability catalog](oss-solution-catalog.md#embodiedgen-v2), and the
[EmbodiedGen operator skill](../../skills/tools/embodiedgen/SKILL.md) for the
same capability boundary.

## Delivery, identities, and access boundary

| Boundary | Exact identity / handling |
| --- | --- |
| EmbodiedGen source | `HorizonRobotics/EmbodiedGen@f0124197888c2b733e4eaa65acd81ad9cfda3b79`, Apache-2.0, runtime fetched |
| TRELLIS source | upstream gitlink `55a8e8164b195bbf927e0978f00e76c835e6011f`, MIT, runtime fetched |
| TRELLIS model | `microsoft/TRELLIS-image-large@25e0d31ffbebe4b5a97464dd851910efc3002d96`, MIT, runtime fetched and receipt-verified |
| Baked image | digest-pinned CUDA/OS bootstrap plus NPA fetch/validation code only; no source, model, task input, Python application dependencies, cache, output, or credential bytes |
| Input and cache | worker-readable HTTPS or S3 image fetched into a run-local staging directory; fetched runtime cache is outside image layers and checked against receipts |
| Outputs | run-scoped S3 objects; see [declared artifacts](#declared-input-and-output-contract) |

The CUDA development derivative is operator-private. Build and pull it only
from the operator-controlled registry selected by supported private NPA
configuration; it must never be pushed to a public namespace or GHCR. The
recipe's exact redistribution rationale and third-party notices are in
[`REDISTRIBUTION.md`](../../npa/docker/workbench/embodiedgen/REDISTRIBUTION.md)
and [`THIRD_PARTY_NOTICES.md`](../../npa/docker/workbench/embodiedgen/THIRD_PARTY_NOTICES.md).
Runtime fetch does not grant source, model, input, output, or service rights.

TRELLIS is the selected public backend. SAM3D and the Tencent cloud backend are
not used; do not invent a terms-acceptance environment variable. The configured
Token Factory credential is required solely by EmbodiedGen's upstream VLM
property-estimation adapter. Supply it through `--secret-env`; never put a
credential in YAML, a URI, a shell variable committed to a file, or an output
manifest.

## Prerequisites and values

Run these on the Linux operator host that owns the intended run, after resolving
the project alias, private registry, S3 storage, and Kubernetes context through
normal NPA configuration. An operator host name is not a project alias.

```bash
npa workbench health preflight --project <configured-project-alias> \
  --checks nebius,token_factory,s3 --json
npa workbench health access --capability token_factory --json
```

The second command checks Token Factory capability access; there is no gated
TRELLIS acceptance step for this selected public model. The runtime bootstrap
itself verifies the exact source/gitlink/model revisions and cache receipts
before it invokes upstream code. Use a configured, compatible RTX PRO 6000
Blackwell target; do not switch capacity class, GPU type, or controller just to
run this workflow.

Choose these non-secret values before planning. The input must be readable by
the **worker**, not merely by the machine issuing the command. A local path is
not an input contract. HTTPS inputs must be ordinary credential-free HTTPS URIs;
S3 inputs must be readable by the selected project's worker credentials. The
checked-in default is a pinned upstream JPEG for qualification only; use an
operator-controlled S3 or HTTPS image for a real task.

```bash
PROJECT='<configured-project-alias>'
OUTPUT_BUCKET='<configured-output-bucket>'
KUBERNETES_CONTEXT='<configured-kubernetes-context>'
RUN_ID='<new-unique-run-id>'
INPUT_URI='s3://<worker-readable-bucket>/inputs/<object>.jpg'
WORKFLOW_STATE_URI="s3://${OUTPUT_BUCKET}/workflow-runs/${RUN_ID}"
```

The workflow base64-transports `input_uri` through the rendered shell so URI
characters remain data rather than shell syntax. That transport does not make a
URI reachable or authorize its contents.

## Build the private candidate

This is an operational prerequisite, not a claim that the candidate is already
built or qualified. Do it only after the required secure-image-build checks and
on a host with enough private build space. The tag binds the full current NPA
source SHA; resolve it to a digest before using it anywhere else.

```bash
NPA_SOURCE_SHA=$(git rev-parse HEAD)
IMAGE_TAG="${NPA_REGISTRY}/npa-embodiedgen:dev-${NPA_SOURCE_SHA}"
bash npa/docker/workbench/embodiedgen/build.sh "${IMAGE_TAG}"
IMAGE_DIGEST=$(docker buildx imagetools inspect "${IMAGE_TAG}" \
  --format '{{.Manifest.Digest}}')
IMAGE_REF="${NPA_REGISTRY}/npa-embodiedgen@${IMAGE_DIGEST}"
```

`NPA_REGISTRY` must name a fully-qualified private registry host. `build.sh`
rejects Docker Hub shorthand, public registries, a non-40-character source SHA,
and a tag other than `dev-<full-source-sha>`. Before a live run, scan and
inspect the exact private digest under the secure-image-build procedure. No
public-development tag, release promotion, or anonymous pull is permitted for
this candidate.

## Validate, render, preflight, and submit

Use the same immutable `IMAGE_REF`, input, output bucket, run ID, project, and
Kubernetes context at every phase. The exact image reference must appear both as
`base_image` and as the `workbench.byof.repo` image override; the renderer
rejects mismatch, mutability, and public registries.

```bash
npa workbench workflow validate-spec workflows/testing/byof-embodiedgen.yaml --json

npa workbench workflow plan-spec workflows/testing/byof-embodiedgen.yaml \
  --run-id "${RUN_ID}" \
  --var "bucket=${OUTPUT_BUCKET}" \
  --var "input_uri=${INPUT_URI}" \
  --var "base_image=${IMAGE_REF}" \
  --check-render --json

npa workbench workflow submit workflows/testing/byof-embodiedgen.yaml \
  --run-id "${RUN_ID}" --project "${PROJECT}" \
  --workflow-s3-uri "${WORKFLOW_STATE_URI}" --durable-s3 --plan-only \
  --var "bucket=${OUTPUT_BUCKET}" \
  --var "input_uri=${INPUT_URI}" \
  --var "base_image=${IMAGE_REF}" \
  --image-override "workbench.byof.repo=${IMAGE_REF}" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY
```

The first two commands are local schema/render checks. `submit --plan-only`
renders the same SkyPilot task without launching it. Just before a real submit,
prove the target cluster can pull that exact private digest; this creates only a
bounded, owned pull-probe pod and verifies its cleanup:

```bash
npa workbench workflow preflight-images workflows/testing/byof-embodiedgen.yaml \
  --project "${PROJECT}" --infra "k8s/${KUBERNETES_CONTEXT}" \
  --var "bucket=${OUTPUT_BUCKET}" \
  --var "input_uri=${INPUT_URI}" \
  --var "base_image=${IMAGE_REF}" \
  --image-override "workbench.byof.repo=${IMAGE_REF}" --json
```

When the preflight is successful and the operator is authorized to use the
exact private image and runtime, remove `--plan-only` from the same `submit`
command. NPA then stages required source, uses SkyPilot's standard run-scoped
controller path, and records durable state at `WORKFLOW_STATE_URI`; do not call
raw `sky jobs launch` or create a separate service.

## Declared input and output contract

The state declares one `image/*` input at `input_uri` and these required output
objects below `s3://<OUTPUT_BUCKET>/oss-solutions/embodiedgen/<RUN_ID>/`:

| Object | Meaning and downstream use |
| --- | --- |
| `embodiedgen_image_to_rigid_object.json` | Primary structured provenance. Its exact top-level fields include `input`, `source_revision`, `trellis_revision`, `trellis_model_revision`, `image_reference`, `runtime_receipt_sha256`, `gpu`, `urdf`, `mjcf_conversion`, `generated_asset_bundle`, `collision_geometry`, `physics`, and `artifacts`. |
| `generated_asset.tar.gz` | Exact generated `generated/` tree containing the URDF and mesh assets in their relative layout; unpack before loading it into a downstream simulator |
| `mjcf_asset.tar.gz` | EmbodiedGen-produced `mjcf/` handoff tree and its mesh assets for MuJoCo/Genesis import; conversion is proven, but it is not a MuJoCo or Genesis physics result |
| `pybullet_view.png` | Final decoded PyBullet render |
| `pybullet_settle.mp4` | Fully decoded PyBullet settling view |
| `npa_byof_summary.json` | BYOF worker result, exact image/source metadata, exit code, and byte/hash inventory of uploaded artifacts |

Download the primary JSON first, verify its image/source/model/input hashes
against the selected run, then use its `urdf.path`, `collision_geometry`, and
bundle member inventory to unpack the exact assets. The BYOF profile requires a
successful smoke exit and readbacks of its primary report and summary. Separately,
the standard workflow runtime validates every declared S3 output before it marks
the workflow wave complete. A summary JSON alone is not the generation claim.

## Monitor, inspect, retry, and clean up

All lifecycle commands use the original project, run ID, and durable workflow
prefix. They are generic Workbench control-plane commands, so agents can use
the same contract without a special EmbodiedGen API.

```bash
npa workbench workflow status "${RUN_ID}" --project "${PROJECT}" \
  --workflow-s3-uri "${WORKFLOW_STATE_URI}" --watch --json
npa workbench workflow logs "${RUN_ID}" byof-run --project "${PROJECT}" \
  --workflow-s3-uri "${WORKFLOW_STATE_URI}" --json
npa workbench workflow artifacts "${RUN_ID}" --project "${PROJECT}" \
  --workflow-s3-uri "${WORKFLOW_STATE_URI}" --json
npa studio search --project "${PROJECT}" --bucket "${OUTPUT_BUCKET}" \
  --prefix "oss-solutions/embodiedgen/${RUN_ID}/" --read-metadata
```

For an interrupted submission, inspect `status`, `logs`, and the declared
outputs first. Resume only with the same immutable image/configuration and
durable identity; `--resume-run` reconciles the existing run rather than
blindly launching a duplicate. This single stage has no supported TRELLIS
mid-generation checkpoint, so a verified retry re-runs the stage; it does not
claim model-level resume.

```bash
npa workbench workflow submit workflows/testing/byof-embodiedgen.yaml \
  --resume-run "${RUN_ID}" --project "${PROJECT}" \
  --workflow-s3-uri "${WORKFLOW_STATE_URI}" --durable-s3 \
  --var "bucket=${OUTPUT_BUCKET}" \
  --var "input_uri=${INPUT_URI}" \
  --var "base_image=${IMAGE_REF}" \
  --image-override "workbench.byof.repo=${IMAGE_REF}" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY
```

To stop only this run, cancel by its exact identity and wait for the resulting
terminal state. Do not tear down a shared controller, cluster, registry, cache,
or another owner's resources. `npa cleanup --json` is a report-only local
residue audit; it does not delete cloud resources.

```bash
npa workbench workflow cancel "${RUN_ID}" --project "${PROJECT}" \
  --workflow-s3-uri "${WORKFLOW_STATE_URI}" --json
npa cleanup --json
```

The generic SDK offers the equivalent read-side agent integration after
submission; it does not add a second submitter:

```python
from npa.sdk.workbench import workflow

state = workflow.status(RUN_ID, project=PROJECT, workflow_s3_uri=WORKFLOW_STATE_URI)
artifacts = workflow.artifacts(RUN_ID, project=PROJECT, workflow_s3_uri=WORKFLOW_STATE_URI)
```

## Qualification status

Implementation, local workflow validation, and render checks are complete only
when their recorded tests pass. This candidate has **no** built private digest,
image-byte scan, actual RTX PRO 6000 run, or live artifact in the public
repository. Those are separate operational gates. Until they are recorded for
one exact private digest, this is an implementation-complete, live-qualification-pending
workflow—not a supported release or a claim of generated model quality.
