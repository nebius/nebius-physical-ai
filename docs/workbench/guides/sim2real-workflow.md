# Run Sim2Real manually from a fresh checkout

[Guides](README.md) · [Workbench documentation](../README.md)

Run the canonical [14-stage workflow](../../../workflows/main/sim2real.yaml)
from a terminal on an always-on Linux operator machine. This guide uses the public
Franka seed and stock robot; no agent deployment or private operator scripts
are required. Stage 12 records the external physical-robot validation seam.
Completing the simulated workflow does not establish real-robot success.

Run the commands from the checkout root in **one Bash shell** on Linux with
`/proc` mounted. Replace quoted placeholders with
your private values. Keep the run settings outside Git so the same arguments
can be restored after a shell or machine restart.

| Order | Gate before continuing |
| --- | --- |
| Install and validate | CLI loads and the canonical spec validates without credentials |
| Select images | Five compatible immutable images, a common source SHA, and qualification evidence are available |
| Configure access | Selected project, writable storage, Transfer access, and hosted evaluator access pass |
| Prepare the cluster | RTX rendering drivers, CPU capacity, shared filesystem, exact context, and controller ownership are ready |
| Warm and preflight | The selected Isaac cache is ready and every actual image path passes |
| Stage and plan | Real seed objects are staged; the same arguments render the complete graph |
| Submit and inspect | Durable runtime completes and the report, checkpoint, RRD, and MCAP are verified |
| Clean up | Cancel the exact run before retiring resources you own |

## 1. Install and validate locally

Install [host prerequisites](../../install.md): Git, Python 3.12, Nebius CLI,
Terraform, and `kubectl`. A local GPU, Isaac installation, Docker daemon, and
AWS CLI are unnecessary for this operator path. Engines run in cluster images.
macOS can install, validate, and plan. Perform execution, monitoring, recovery,
and cleanup on Linux: the isolated SkyPilot API verifies process ownership
through `/proc`. WSL2 must remain running throughout the submission.

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
```

For the measured development reproduction, select the checkout in the
[execution receipt](../validation/sim2real-manual-development-20261010.md)
before creating the virtualenv. It records the execution and cleanup revisions.
For another image set, use its maintainer's compatible checkout.

```bash
python3.12 -m venv npa/.venv
npa/.venv/bin/python -m pip install -e npa
npa/.venv/bin/npa --version
npa/.venv/bin/npa workbench workflow --help

SPEC=workflows/main/sim2real.yaml
npa/.venv/bin/npa workbench workflow validate-spec "$SPEC" \
  --preset public-franka-lift --json
```

Expected: CLI checks exit zero and validation reports `status: valid`.
This guide creates `npa/.venv`; the shared quickstart creates `.venv`. Use the
explicit executable paths shown here throughout, including in new shells.
Validation proves the declaration, not image availability or a GPU run.

## 2. Select a qualified image set before provisioning

The canonical YAML intentionally leaves all five image references and
`source_sha` empty. Obtain an image-set receipt from your image maintainer or
use the measured development set below.

The [October 10 development execution receipt](../validation/sim2real-manual-development-20261010.md)
provides five exact public development pins and their tested operator revisions.
It records full execution and zero strict placement successes; it does not
promote the quarantined release defaults.

| Setting | Image and required capability |
| --- | --- |
| `CONTROLLER_IMAGE` | `npa-sim2real-control`, including current hosted Stage 8/9 support |
| `COSMOS3_IMAGE` | `npa-cosmos3`, with real guarded Cosmos3-Nano edge-controlled `video2video` transfer |
| `ENVGEN_IMAGE` | `npa-envgen`, with real scenario generation |
| `ISAAC_IMAGE` | `npa-isaac-lab`, with rollout, PPO, held-out evaluation, and RTX rendering |
| `VIEWER_IMAGE` | `npa-rerun-viewer`, with RRD and MCAP finalization |
| `SOURCE_SHA` | Common full 40-hex commit baked into all five images as `NPA_IMAGE_SOURCE_SHA` |

Every image must be a complete `repository@sha256:<64-hex-digest>` reference.
The canonical workflow binds all five role values as immutable execution inputs:
after setting them with `--var`, do not use a global `--image` or a
`--image-override` that selects different bytes for a Sim2Real stage. Planning
and image preflight reject that mismatch before a pull or launch. To replace a
role image, change its corresponding `--var` and rerun the complete five-role
preflight with the same argument set.

When an agent backend generates a canonical Sim2Real draft, its selected
profile must provide all five raw role fields: `controller_image`,
`cosmos3_image`, `envgen_image`, `isaac_image`, and `viewer_image`. A missing
or partial set remains non-runnable. NPA does not infer those roles from legacy
`policy_image` or `vlm_image` fields, and it never combines selections from
different profiles. This selection check is separate from qualification: each
chosen immutable image still has to pass the complete five-role preflight.

The receipt must name the tested CLI/workflow checkout, record any later
operator cleanup fixes, and establish
compatibility with its hosted evaluator contract. Use that checkout when
reproducing the result; its revision can differ from the common image source
SHA. Pullability and matching labels alone do not prove those capabilities.
Retain both revisions with your run settings.

**Release gate:** the checked-in [readiness record](../../../workflows/main/sim2real.readiness.json)
currently marks `source_image` as `unverified`; this checkout does not supply a
qualified five-image default. Stop before provisioning if you have no qualified
set. Image maintainers use [container packaging and publication](../container-packaging.md)
to close that prerequisite. The historical September 4 coherent set predates
the current MiniMax evaluator contract, and affected historical public releases
remain [quarantined](../validation/public-default-quarantine-impact-20261005.md#other-quarantined-images-and-sim2real).
Neither substitutes for a compatible set. Do not bypass preflight, inject source
into pods, or patch archived artifacts to make an old set pass.

```bash
SOURCE_SHA='<common-full-40-hex-source-sha>'
CONTROLLER_IMAGE='<qualified-controller-repository>@sha256:<64-hex-digest>'
COSMOS3_IMAGE='<qualified-cosmos3-repository>@sha256:<64-hex-digest>'
ENVGEN_IMAGE='<qualified-envgen-repository>@sha256:<64-hex-digest>'
ISAAC_IMAGE='<qualified-isaac-repository>@sha256:<64-hex-digest>'
VIEWER_IMAGE='<qualified-viewer-repository>@sha256:<64-hex-digest>'
```

<a id="5-buildpush-once-and-prove-the-exact-image-pulls"></a>

For private images, prepare exact-host registry authorization and an existing
Kubernetes Docker config pull secret using the [registry setup](../troubleshooting/known-footguns.md#private-registry-credentials-expire).
NPA's manifest checks and Kubernetes node pulls need separate authorization.
Reference the secret in `kubernetes.pod_config.spec.imagePullSecrets` in your
private SkyPilot config. The canonical Stage 00 cache-warm state uses the
selected Isaac image through the standard SkyPilot runtime, so it receives that
configuration and its pull secrets. Do not add a direct `kubectl` cache-warming
Job. Anonymous public GHCR images need neither credential.

## 3. Configure the project, storage, and model access

Stage 8 uses a single hosted `MiniMaxAI/MiniMax-M3` evaluator through Nebius
Token Factory. Review the [MiniMax-M3 model license](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE)
and the terms applicable to your hosted use. NPA does not distribute or cache
these weights. The [August 2026 notice](https://docs.tokenfactory.nebius.com/august-2026-deprecation-notice)
announced removal of the former public Cosmos3 model on August 31.

The canonical `token_factory_model` config key records the actual hosted model;
its fresh Stage 8 state and envelope are respectively
`stage-08-token-factory` and `token-factory.json`
(`npa.sim2real.token_factory_evaluator.v1`). MiniMax results are not attributed
to NVIDIA Cosmos. Historical `cosmos3_*` names in already-published schemas or
outcomes are not reinterpreted or rewritten. A dedicated or self-hosted Cosmos3
Reasoner is a separate optional backend: it is not enabled by a Token Factory
model rename and requires real access plus a validated evaluation contract.
Start a new run when changing models or upgrading this evaluator contract;
Stage 9 verifies model identity, family, endpoint, request accounting, and
exact Stage 7 rollout coverage before PPO.
The hosted endpoint must support the ordered JSON Schema response contract:
each generated event has a fixed action index and recorded camera reference,
including explicit neutral values for unsampled actions. Validate this with
real rollout frames before a full run; invalid responses remain rejected.

Stage 8 defaults to eight concurrent requests and all declared primary frames:
`evaluation_concurrency=8`, `evaluation_max_frames=0`. The former eight-frame
sample left many action times without visual evidence. A positive frame count
still selects a deterministic sample and preserves neutral labels for actions
without evidence. Keep capture visibility checks even with full frame coverage.

Each completed rollout evaluation is published, read back, and validated in
an iteration-scoped S3 receipt before the Stage 8 barrier is published. Receipts
bind the image source, endpoint, model, threshold, actions, metadata, and frame
bytes. An interrupted Stage 8 resumes verified results and requests only missing
evaluations. Storage denial and corrupt evidence fail closed. The envelope
reports `reused_rollouts` and `new_requests` separately from the original
evaluation usage, so replay does not appear as new inference.


Choose a local project alias and a cluster context name. The alias is not the
provider project ID; the context may differ from the provider cluster name.
`NPA_CLUSTER` consistently means the NPA context/profile name below.

```bash
export NPA_PROJECT='<local-project-alias>'
export NPA_CLUSTER='<npa-cluster-context>'
export NPA_CONFIG_DIR="$HOME/.npa/sim2real-operator"
mkdir -p "$NPA_CONFIG_DIR"
chmod 700 "$NPA_CONFIG_DIR"
npa/.venv/bin/npa configure --project-alias "$NPA_PROJECT"
npa/.venv/bin/npa configure --show
```

Select the intended tenant, project, and region, save model credentials in the
private credential store, and create or deliberately reuse writable S3 storage.
Interactive storage creation needs project admin permissions. For existing
operator-managed storage, follow [configuration](../../configuration.md)
instead of creating a second bucket. Keep secrets under `~/.npa/` or in a
private process environment, never in workflow YAML or shell history.

| Credential name | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Seed reads, checkpoints, reports, and durable run state |
| `HF_TOKEN` | Exact Cosmos3-Nano and Guardrail runtime-access preflight |
| `NEBIUS_TOKEN_FACTORY_KEY` | Hosted Stage 8 evaluation; an IAM token cannot replace it |

Before a GPU submission, validate the exact Stage 3 runtime payloads with the
same `HF_TOKEN` that the job will receive:

- [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano), for the structural `video2video` weights.
- [Cosmos Guardrail](https://huggingface.co/nvidia/Cosmos-Guardrail1), for runtime safety checks.

The preflight probes pinned payloads for both; it does not accept a legacy
Transfer2.5/Predict2.5 approval as a substitute. `401` means invalid
authentication; `403` means missing approval or token scope. See [Hugging Face
setup](../huggingface-token.md). `NGC_API_KEY` is not a runtime prerequisite
when all five images already exist; separate builds may need it.

Stage 8 needs image input and strict ordered JSON Schema output. This checkout
uses hosted MiniMax-M3; review its [model license](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE)
and hosted-service terms. `token_factory_model` records the actual model and
results record its family and exact endpoint. Do not select a text-only model.
For a custom endpoint, set the credential-free `token_factory_base_url` and
validate its model discovery and inference contract before submit; a Cosmos3
Reasoner needs a separately supported backend and is not claimed here. Start a
new run when changing evaluator contracts.

Review the [Omniverse terms](https://docs.omniverse.nvidia.com/usd/latest/common/NVIDIA_Omniverse_License_Agreement.html),
[Isaac additional licenses](https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html),
and [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
before warming or running Isaac. Non-interactive NPA submissions default
`ACCEPT_EULA=Y`; `--no-accept-eula` opts out and blocks Isaac execution.
Privacy and telemetry consent remain disabled independently.

Set `NPA_BUCKET` to the **bare bucket name** shown for that project:

```bash
export NPA_BUCKET='<selected-bucket-name>'
npa/.venv/bin/npa workbench health preflight \
  --project "$NPA_PROJECT" --checks nebius,s3,token_factory --json
npa/.venv/bin/npa workbench health access --capability sim2real --json
npa/.venv/bin/npa workbench token-factory models
```

Expected: authentication and storage checks pass, Transfer payload access passes,
and the key-scoped model list includes `MiniMaxAI/MiniMax-M3`. A `SKIP` is not
proof of access. Save the selected project's credentials first; the access and
model commands use the saved configuration.

## 4. Prepare the exact execution target

Choose **one** cluster path below. Bootstrap pinned SkyPilot before provisioning,
because cluster readiness includes its GPU smoke task:

```bash
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$NPA_CONFIG_DIR/sim2real-runtime"
npa/.venv/bin/npa skypilot bootstrap --path "$NPA_CONFIG_DIR/skypilot-venv"
export NPA_SKYPILOT_BIN="$(npa/.venv/bin/npa skypilot status --bin-path)"
```

### New NPA-managed cluster

Use `rtx-rendering`. The default managed-driver compute profile lacks the
mounted graphics stack required by Isaac cameras. One `16vcpu-64gb` CPU node
has headroom for the 8-vCPU/32-GiB CPU states and SkyPilot controller;
`8vcpu-32gb` is too small after Kubernetes reservations.

The cache needs a shared filesystem attached to consuming node groups and the
`csi-mounted-fs-path-sc` StorageClass. The S3 bucket does **not** provide that
filesystem. On a fresh checkout without private Terraform overrides, use the
supported Terraform setting:

```bash
export TF_VAR_enable_filestore=true
export TF_VAR_filesystem_csi_chart_repository='<operator-approved-filesystem-csi-helm-repository>'
CLUSTER_ARGS=(
  --project "$NPA_PROJECT" --cluster-name "$NPA_CLUSTER"
  --gpu-workload-profile rtx-rendering
  --gpu-graphics-smoke-image "$ISAAC_IMAGE"
  --cpu-nodes 1 --cpu-platform cpu-e2 --cpu-preset 16vcpu-64gb
)
npa/.venv/bin/npa provision-if-absent "${CLUSTER_ARGS[@]}" \
  --dry-run --output-format json
npa/.venv/bin/npa provision-if-absent "${CLUSTER_ARGS[@]}"
```

Review the selected project and intended actions before the real command.
The profile defaults to one RTX PRO 6000 GPU node. For a deliberately larger
pool, add the same GPU shape/count options to `CLUSTER_ARGS` before both calls.
The CSI Helm repository is a required operator input; NPA supplies no default.
Obtain it before provisioning. The filesystem needs its own quota;
`TF_VAR_existing_filestore` can attach an
authorized existing filesystem instead. Private `terraform.tfvars` values take
precedence over `TF_VAR_*`. See the [cluster configuration example](../../../deploy/cluster/terraform.tfvars.example)
and [driver contract](../mk8s-gpu-driver-strategy.md#rtx-rendering-workload-profile).

Expected: provisioning completes node stability, CUDA, and graphics checks and
writes the kubeconfig. The SkyPilot GPU task runs after private namespace
selection below. Complete both readiness steps.
The graphics probe uses the selected immutable Isaac image. The governed SONIC
default may be quarantined; select this verified workflow image explicitly
instead of bypassing the image gate.

### Using a cluster `provision-if-absent` did not create

Reuse an externally managed cluster's capacity and filesystem.
`provision-if-absent` does not adopt it and can plan a second cluster.
Obtain its provider project ID and provider cluster name, then register it:

```bash
NPA_PROJECT_ID='<selected-provider-project-id>'
NPA_CLUSTER_NAME='<existing-provider-cluster-name>'
npa/.venv/bin/npa cluster kubeconfig \
  --cluster-name "$NPA_CLUSTER_NAME" --project-id "$NPA_PROJECT_ID" \
  --project "$NPA_PROJECT" --context "$NPA_CLUSTER"
```

`npa cluster kubeconfig` writes the kubeconfig and local cluster-identity
record that `npa cluster status` and `workflow submit --infra k8s/<context>`
read; without it, submit refuses with "No NPA cluster identity exists for
context ...; refusing controller adoption." `npa skypilot bind-controller`
records which project/context owns the shared SkyPilot jobs controller;
without it, submit refuses with "No shared controller owner is bound."

`npa workbench health sim2real --project "$NPA_PROJECT"` defaults to the
image-independent coherence, project-scoped S3, token, and cluster checks. It
checks pod creation and Job patch permissions using the selected kubeconfig
identity; it does not impersonate the retired Sim2Real sibling-Job controller's
`agent-sa`. The default deliberately does not resolve retired loop images or
certify canonical role images: run the complete five-role `workflow
preflight-images` command below for source-attested pullability. Keep the same
project, kubeconfig, and context for health checks and canonical `workflow
submit --runtime`. Use `--checks config,registry` or `--checks all` only when
diagnosing the retained legacy loop configuration. When selecting only
image-independent health checks, omit image, threshold, and S3-prefix
overrides: the CLI rejects inputs those checks cannot use.

## 3. Add a schedulable CPU pool before GPU work

SkyPilot's Kubernetes jobs controller requests 2 vCPU/8 GiB. Sim2Real CPU states
request 8 vCPU/32 GiB and deliberately use the small
`npa-sim2real-control` image. Give them a Ready, schedulable, appropriately
untainted node with enough allocatable CPU and memory. It may also advertise GPUs;
the CPU profile has no GPU exclusion:

Require validated RTX graphics drivers, a working shared filesystem on the
consuming node groups, and sufficient CPU capacity. If a CPU pool is absent,
its owner can add one:

```bash
npa/.venv/bin/npa cluster node-group add-cpu \
  --cluster-name "$NPA_CLUSTER" --name sim2real-cpu \
  --platform cpu-d3 --preset 16vcpu-64gb --node-count 1 --wait
```

An 8-vCPU/32-GiB node cannot fit the canonical CPU pod after Kubernetes reserves
node resources. Discover the available CPU platforms and presets in the selected
project before creating this 16-vCPU/64-GiB pool; the `cpu-d3` example is region
dependent.

For an adopted cluster, retain its owner's filesystem configuration when
repeating `provision-if-absent` readiness. If its default StorageClass is
`csi-mounted-fs-path-sc`, set `TF_VAR_enable_filestore=true` in the private
shell settings. Otherwise the validator expects `compute-csi-default-sc` and
waits for a configuration that does not describe this cluster. Use the exact
existing CPU/GPU node counts and shapes, its adopted context and kubeconfig,
in this argument array for the readiness command below:

```bash
CLUSTER_ARGS=(
  --project "$NPA_PROJECT" --cluster-name "$NPA_CLUSTER"
  --gpu-workload-profile rtx-rendering --gpu-graphics-smoke-image "$ISAAC_IMAGE"
  --cpu-nodes '<existing-cpu-node-count>'
  --cpu-platform '<existing-cpu-platform>' --cpu-preset '<existing-cpu-preset>'
  --gpu-nodes '<existing-gpu-node-count>'
  --gpu-platform '<existing-gpu-platform>' --gpu-preset '<existing-gpu-preset>'
)
```

The cached kubeconfig is reused; these checks do not replace adoption or create
a missing filesystem attachment.

### Isaac placement and mixed GPU pools

The check resolves the three Isaac states' GPU, CPU, and memory requirements,
including resource overrides, and merges global, selected-context, and task pod
placement with the same rules used by SkyPilot. Cordoned or unready nodes,
untolerated taints, node allowlists, selectors, and required node affinity exclude
pools that cannot host these tasks. A managed compute pool excluded from Isaac
placement does not block a separate operator-mounted render pool. Missing node
readiness/resource evidence or unsupported placement constraints block validation
instead of establishing compatibility.

This label check detects a known unsupported driver configuration. It does not
replace the real graphics-readiness gate or prove that unlabelled drivers work.

Expected for the Isaac pool: the RTX PRO 6000 rows report
`gpu.deploy.operands=true`. If they report `driverful=true` instead, reprovision
that pool with `--gpu-workload-profile rtx-rendering`.

Transfer and EnvGen default to `RTXPRO6000:1` so one reserved RTX pool can run
the entire graph. On a mixed cluster, pass `--var cosmos3_accelerator=B200:1`
and `--var envgen_accelerator=B200:1` to use reserved B200 compute capacity.
The Isaac resource remains `RTXPRO6000:1` because it requires RT cores and the
graphics driver stack. The canonical `gpu_concurrency` default is deliberately
`1`: static YAML cannot know the target's live capacity. Before planning,
submitting, or resuming, a task-scoped driver must discover the selected
EnvGen-compatible GPUs that are actually schedulable and set
`gpu_concurrency=min(shard_count, measured_capacity)`. Do not use nominal,
enumerated, or total pool capacity. For example, a measured two-GPU target uses
`2`, rendering the eight shards in four bounded waves. If reserved placement
fails, provision an operator-authorized preemptible fallback; application or
model-access failures require fixing their cause before retrying.

### Select a private workload namespace and verify the target

For either path, use the kubeconfig printed by NPA. The default location is:

```bash
export KUBECONFIG="$NPA_CONFIG_DIR/clusters/$NPA_CLUSTER/kubeconfig"
kubectl config use-context "$NPA_CLUSTER"
kubectl config current-context
kubectl get nodes -L nvidia.com/gpu.product,nebius.com/driverful,nvidia.com/gpu.deploy.operands
kubectl get storageclass csi-mounted-fs-path-sc
```

Create a namespace for this operator run and prepare its authenticated context
through the supported [namespace command](../namespaces.md). Choose new names
and a new private destination; `namespace context` refuses existing destinations.
Retain the private runtime directory selected before provisioning.

```bash
export NPA_NAMESPACE=sim2real-manual
export NPA_CLIENT_DIR="$NPA_CONFIG_DIR/$NPA_NAMESPACE-client"
npa/.venv/bin/npa workbench namespace apply "$NPA_NAMESPACE" --context "$NPA_CLUSTER"
npa/.venv/bin/npa workbench namespace context "$NPA_NAMESPACE" \
  --context "$NPA_CLUSTER" --output-dir "$NPA_CLIENT_DIR"
export KUBECONFIG="$NPA_CLIENT_DIR/kubeconfig"
export SKYPILOT_GLOBAL_CONFIG="$NPA_CLIENT_DIR/sky.yaml"
npa/.venv/bin/npa skypilot verify \
  --cluster "$NPA_CLUSTER" --kubeconfig "$KUBECONFIG" --output-format json
npa/.venv/bin/npa provision-if-absent "${CLUSTER_ARGS[@]}" \
  --kubeconfig "$KUBECONFIG" --skip-s3 --sky-smoke \
  --dry-run --output-format json
npa/.venv/bin/npa provision-if-absent "${CLUSTER_ARGS[@]}" \
  --kubeconfig "$KUBECONFIG" --skip-s3 --sky-smoke
npa/.venv/bin/npa workbench workflow gpus \
  --cluster "$NPA_CLUSTER" --project "$NPA_PROJECT" --json
```

Expected: the intended context is selected, CPU and RTX nodes are Ready,
the renderer-facing graphics gate passed, Kubernetes is enabled, and GPU
discovery reports requestable capacity. Require `provider_mutation=false` in
the cached readiness preview before its real command. The actual GPU smoke
bootstraps SkyPilot's service account in this selected namespace; `verify`
alone checks authentication and cannot prepare it for image pull probes.
Newly provisioned operator nodes report
`nvidia.com/gpu.deploy.operands=true`; labels on an older external cluster do not
replace the actual GLX/EGL/Vulkan check. An excluded managed
compute pool can coexist with the render pool. A StorageClass name alone does
not prove its filesystem attachment; see [shared-cache diagnostics](../model-weight-cache.md).
The isolated state derives its own stable controller identity; do not call
`bind-controller` or use another operator's runtime directory. Keep all four
configuration exports (`KUBECONFIG`, `SKYPILOT_GLOBAL_CONFIG`,
`NPA_SKYPILOT_ISOLATED_CONFIG_DIR`, and `NPA_SKYPILOT_BIN`) on this Linux machine and restore them in every
shell used for submit, status, logs, resume, or cleanup. NPA cluster identity
must still match the selected project/context. See [SkyPilot setup](../../orchestration/skypilot-setup.md).

## 5. Provision the cache claim; let the canonical workflow warm it

The only operator-managed object is the durable RWX claim. Create or verify it
through the shared CLI against the exact context and namespace that submission
will use. This command never creates a Job and never replaces an existing
operator-owned claim.

```bash
ISAAC_CACHE_PVC="npa-isaac-cache"
npa/.venv/bin/npa workbench sim2real cache provision \
  --name "$ISAAC_CACHE_PVC" \
  --kubeconfig "$KUBECONFIG" \
  --context "$NPA_CLUSTER" \
  --namespace "$NPA_NAMESPACE" \
  --storage-class csi-mounted-fs-path-sc \
  --storage 64Gi \
  --json
```

Expected: the receipt reports `status: ready`, `phase: Bound`, and
`ReadWriteMany`. The command waits without a synthetic job-duration deadline;
if storage provisioning is pending, inspect that exact claim and its events,
then rerun the same command. It refuses to alter a foreign or incompatible
claim, so use a new claim name instead of replacing somebody else's volume.

Do not apply a raw warming Job. The canonical `sim2real.yaml` begins with its
CPU-only `stage-00-isaac-cache-warm` state. That state runs the selected
immutable Isaac image's real `isaac-bootstrap warm`, validates `ready=yes`, and
publishes `cache/isaac-warm.json` with the image source attestation and
content-addressed cache-manifest hash. Only later Isaac GPU states receive the
same claim as read-only and offline. A new Isaac image/bootstrap produces a new
versioned cache tree; begin a new source-qualified workflow run so stage 00
records it without changing an existing run's authority. Keep prior trees and
the claim for existing runs. See
[runtime-fetch packaging](../container-packaging.md#runtime-fetched-isaac-sim-why-the-isaac-images-are-publishable).

### Verify native camera evidence

Before evaluating rollouts, inspect the first and last frames from primary,
side, and overhead cameras. Confirm that all views render the actual scene, then
inspect the primary frames selected for hosted evaluation: the manipulation
object and end effector must be visible enough to assess the task. Stage 8 sees
only primary images, so a clear secondary view cannot compensate for an
occluded primary view. The default primary pose looks across the table from -Y;
the orthogonal rear pose remains the secondary `side` stream. A valid image or
matching action timestamp alone does not establish task visibility. The primary
camera aims 25 degrees downward to retain the observed front-edge manipulation
area; verify this framing again for each robot and scenario set.

The serialized camera poses
use WXYZ quaternions. Isaac Lab 3 changed sensor offsets to XYZW, so the rollout
and held-out evaluation adapters convert the ordering at the sensor boundary.
See the [Isaac Lab 3 migration guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/migration/migrating_to_isaaclab_3-0.html).
If capture or primary task visibility is invalid, fix and rebuild the image and
regenerate those rollouts before hosted evaluation or PPO.

Native camera metadata and sampled action rows record physical
`timestamp_seconds` using Isaac's actual `env.step_dt`, including physics
substeps. `timestamp_timebase` is `simulation_elapsed_since_rollout_start`:
zero is immediately before the first rollout `env.step`, and each post-action
sample records `completed_simulation_steps * simulation_step_seconds`. The
clock continues across environment auto-resets. Existing zero-based `sim_step`
values still identify the action being observed. The final context image uses
the horizon as its `sim_step` sentinel but does not advance physical time; it can
share a timestamp with the last sampled action image.

Physical episode IDs also continue independently for each parallel environment.
Every reset is retained even when it occurs between sampled actions. The first
sampled interval after a reset is archived with zero training credit, and an
immediate autoreset cannot supply the preceding action's outcome. Hosted v5
evaluation checks this boundary evidence again before PPO. See the
[episode and temporal-credit contract](sim2real-data-contracts.md#inner-loop-stages-79).
Changing to this contract requires a new run with rebuilt images.

For example, a 0.02-second environment step produces times 0.02 and 0.42 seconds
at action steps 0 and 20, regardless of capture FPS. Sparse decision samples do
not establish continuous coverage between them. Rerun and MCAP currently use a
separate presentation clock, `frame_index / capture.fps`; optional MP4 exports
play at 2 FPS. These playback durations are not physical simulation durations
and must not be used to establish a sustained grasp or placement.

## 6. Keep one argument set for planning, submit, and resume

Create a fresh ID. The canonical default is `1`; only replace it with the
measured compatible, schedulable EnvGen capacity, bounded by eight shards. The
single-GPU cluster above uses `1`; a measured two-GPU target uses `2`. Keep the
same value for planning, submit, and every same-run resume. Concurrency batches
the eight EnvGen shards without reducing training quality or iteration counts.
Discovery and submit preflight must establish capacity; a wave plan does not
reserve it.

```bash
RUN_ID="sim2real-$(date -u +%Y%m%d-%H%M%S)"
NPA_GPU_CONCURRENCY=1
CONFIG_ARGS=(
  --var bucket="$NPA_BUCKET"
  --var source_sha="$SOURCE_SHA"
  --var controller_image="$CONTROLLER_IMAGE"
  --var cosmos3_image="$COSMOS3_IMAGE"
  --var envgen_image="$ENVGEN_IMAGE"
  --var isaac_image="$ISAAC_IMAGE"
  --var viewer_image="$VIEWER_IMAGE"
  --var isaac_cache_pvc="$ISAAC_CACHE_PVC"
  --var gpu_concurrency="$NPA_GPU_CONCURRENCY"
  --var token_factory_model=MiniMaxAI/MiniMax-M3
)
WORKFLOW_ARGS=(--preset public-franka-lift "${CONFIG_ARGS[@]}")
TARGET_ARGS=(--project "$NPA_PROJECT" --infra "k8s/$NPA_CLUSTER")
SECRET_ARGS=(
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
  --secret-env HF_TOKEN --secret-env NEBIUS_TOKEN_FACTORY_KEY
)
npa/.venv/bin/npa workbench workflow preflight-images "$SPEC" \
  "${TARGET_ARGS[@]}" "${CONFIG_ARGS[@]}" \
  --assume-decision promote_checkpoint --json
```

Expected: every image pull and bootstrap path passes under the actual context,
ServiceAccount, placement, and pull secrets. Large cold images can use
`--image-bootstrap-timeout-seconds 0` to remove the observation deadline;
provenance, capability checks, and verified probe cleanup remain required.
`preflight-images` accepts configuration overrides, but has no `--preset`
option. `WORKFLOW_ARGS` adds the seed preset for planning, submit, and resume.
For a custom hosted endpoint, append its credential name to `SECRET_ARGS` and
keep the evaluator setting consistent. Retain all arrays and values privately
for exact reuse after a shell restart.

### Stage the public Franka seed

The opt-in preset fetches the anonymous public [Franka lift dataset](https://huggingface.co/datasets/huyyyyan/pi05-Isaac-sim_Franka_lift_cube)
at revision `42c181e40a43afb1702c29d6f24d5de25219aff8`. Upstream metadata
declares Apache-2.0; staging checks access, revision, and license before fetching.
It uploads real Parquet actions, both MP4s, every decoded primary-camera frame in
source order (never a representative subset), and hashed manifests. The canonical
Cosmos3 Stage 3 adapter rejects a sequence with fewer than six frames or incomplete
frame-index provenance before GPU inference begins.

```bash
npa/.venv/bin/npa workbench workflow trigger stage-preset \
  --preset public-franka-lift --project "$NPA_PROJECT" \
  --bucket "$NPA_BUCKET" --run-id "$RUN_ID" --output-format json
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" \
  "${WORKFLOW_ARGS[@]}" --run-id "$RUN_ID" \
  --waves --check-render --assume-decision promote_checkpoint --json
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  "${TARGET_ARGS[@]}" "${WORKFLOW_ARGS[@]}" "${SECRET_ARGS[@]}" \
  --run-id "$RUN_ID" --runtime --plan-only --assume-decision promote_checkpoint
```

Expected: staging reports the readable manifest; planning includes all 14 stages,
the shard batches, and Stage 7 → hosted Stage 8 → Stage 9. `--assume-decision`
selects a branch for planning only; omit it from execution. `--plan-only`
launches no tasks and does not establish live readiness. The preset binds the
seed prefix to this ID; preserve both on resume.

The source's 7D IK-relative actions and two cameras are conditioning evidence.
The policy uses 8D joint-delta-plus-gripper actions and three evaluation cameras;
source actions are not silently reused as PPO inputs. See [data contracts](sim2real-data-contracts.md).

### Use your own seed instead

Before image preflight/staging/planning, append verified `dataset_id`,
`trigger_uri`, and `seed_manifest_uri` overrides to `CONFIG_ARGS`, then set
`WORKFLOW_ARGS=("${CONFIG_ARGS[@]}")` to omit the public preset. Your manifest and objects must already be
readable in S3. Follow [customer assets](sim2real-customer-assets.md) and
[RobotSpec](sim2real-robot-spec.md) for robot/scene inputs. Use those same
arguments for all later commands; do not change inputs during resume.

### Save the settings before submitting

Save the variables and Bash arrays from this shell in a private file. The
credential-name array contains names only; credential values stay in NPA's
private store or the process environment.

```bash
(
  umask 077
  declare -p NPA_CONFIG_DIR NPA_PROJECT NPA_CLUSTER NPA_BUCKET \
    NPA_NAMESPACE NPA_CLIENT_DIR KUBECONFIG SKYPILOT_GLOBAL_CONFIG \
    NPA_SKYPILOT_ISOLATED_CONFIG_DIR NPA_SKYPILOT_BIN \
    SPEC RUN_ID SOURCE_SHA CONTROLLER_IMAGE COSMOS3_IMAGE ENVGEN_IMAGE \
    ISAAC_IMAGE VIEWER_IMAGE NPA_GPU_CONCURRENCY \
    CONFIG_ARGS WORKFLOW_ARGS TARGET_ARGS SECRET_ARGS \
    > "$NPA_CONFIG_DIR/$RUN_ID-settings.sh"
)
```

In another Bash shell on the same operator machine, return to the same checkout
root and load that exact file before status, logs, resume, or cleanup:

```bash
source "$HOME/.npa/sim2real-operator/<original-run-id>-settings.sh"
```

Use your actual private configuration directory if it differs from the example.
Restore any credentials supplied only through the environment from their secure
source. Keep this file outside Git, including after the run finishes.

## 7. Submit and follow the run

Keep the driver on the always-on operator machine. Laptop sleep or closing the
terminal can interrupt it while cluster jobs continue.

```bash
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  "${TARGET_ARGS[@]}" "${WORKFLOW_ARGS[@]}" "${SECRET_ARGS[@]}" \
  --run-id "$RUN_ID" --runtime --durable-s3 --max-wait-seconds 0
```

Submit enforces storage writes, secret propagation, model access, cluster/cache
readiness, immutable source-attested images, pulls, and scheduler capacity before
GPU work. Resolve reported prerequisites through setup; `--skip-preflight` is
not part of this path.

Production defaults retain 10,000 scenarios, 64 rollouts, 32 action samples,
1,024 PPO environments, 24 steps per environment per update, 2,000 PPO updates
per inner pass, and three inner/outer iterations. Validation
selects the checkpoint used for gold evaluation; gold remains held out.
Later training rollouts and PPO continuation load the latest training checkpoint,
including its optimizer state for PPO. This can differ from the checkpoint
selected for gold. Promotion uses strict 5 cm stable
placement and threshold `0.50`. Reduced plumbing runs must declare overrides
and cannot claim production efficacy. `--max-wait-seconds 0` removes the wave
wait deadline without weakening success predicates.

Allow hours for the full run. Measured 2,000-update PPO passes on one RTX PRO
6000 Blackwell took about 44 minutes each, before validation and uploads;
this is an observation, not a completion deadline. Cold cache preparation and
renderer startup add separate time. The [ten-minute demo](sim2real-demo-script-10min.md)
presents an already completed result.

From a second shell, restore the project/run settings and inspect:

```bash
npa/.venv/bin/npa workbench workflow status "$RUN_ID" --project "$NPA_PROJECT" --watch
npa/.venv/bin/npa workbench workflow logs "$RUN_ID" --project "$NPA_PROJECT" --follow
npa/.venv/bin/npa workbench workflow artifacts "$RUN_ID" --project "$NPA_PROJECT"
```

Model loading and Transfer inference can stay quiet between log messages.
A cold Isaac renderer startup can also compile ray-tracing shaders on the CPU
while GPU utilization stays low. Its log names the Kit log file, where shader
compilation progress is recorded. PPO retains the full RSL-RL training log and
prints its console tail when a pass finishes, so workflow logs can also stay
quiet during optimizer updates. Check live status and pod events while waiting.
`BYO_ROLLOUT_DONE` finishes native rollout capture; Stage 7 still materializes
and uploads the canonical rollout artifacts afterward. `BYO_EVAL_DONE` can
similarly precede final report and render publication.
A running pod or GPU activity does not prove stage completion;
require the declared output artifacts and successful scheduler status.

Inspect primary, side, and overhead footage for the robot, object, and task.
Stage 8 sees primary frames; secondary views cannot compensate for occlusion.
Invalid capture needs corrected images and new rollouts, not relabelled metadata.
The [temporal-credit contract](sim2real-data-contracts.md#inner-loop-stages-79)
enforces action/frame and simulator-episode bindings; playback FPS is not
physical simulation time. Use measured PPO losses and each pass's own validation
result; compatibility proxy fields are not optimizer measurements. The selected
checkpoint's score does not replace earlier candidate measurements.

## Policy training and held-out evaluation

The production training default is 2,000 PPO updates per inner pass. After the
complete pass, validation ranks every saved 100-update checkpoint and the final
checkpoint using the same sealed split. The trainer reads each candidate object
to record its actual SHA-256 and byte size, which validation must corroborate
before ranking. Each candidate records its native zero-based index and completed
updates separately from the full pass count. A stronger earlier actor may win;
selection never reads Gold. Subsequent train rollouts
and PPO resume the best candidate's exact checkpoint, which can differ from the
newest training checkpoint; a three-pass loop performs 6,000 new updates while
retaining stronger validation candidates. Strict success and placement rank
first, followed by stable grasp and lift achieved in the same episode. Extra
lifts without stable grasp cannot displace a stronger paired-skill candidate.
Before a resumed rollout or PPO pass starts, the candidate and its validation
report must agree on the checkpoint URI, SHA-256, byte size, and generator
digest. An incomplete or contradictory byte identity fails closed; it cannot
select a resume checkpoint.

Fresh canonical stock-Franka training initializes the actor with
`franka-cartesian-gru-distillation-v2` before PPO. The actor commands the native
absolute TCP pose and binary gripper through Isaac Lab's differential-IK action
term. Its GRU learns history from measured proprioception, object and goal state,
TCP geometry, and the actual reset clock. The clock saturates at five seconds;
changing that feature does not change the simulation clock or episode duration.
Training, camera rollouts, validation, and Gold share this policy configuration
and reset its recurrent state at every real episode boundary.

Two demonstration rounds use pure teacher actions and then a 50% actor mixture.
A third collection uses a 75% actor mixture and retains earlier demonstrations.
Each fitting round performs 32 passes over real reset-bound sequences, with
training-only expert phase labels balancing the loss. Those labels are separate
from the actor's observations. Fitting initializes native observation
normalization in training mode. Both initial rounds must establish strict
placements on at least half their training scenarios. Before PPO starts, a
separate actor-only rollout must also pass that threshold across the complete
consumed training split. If it fails, another 75% actor collection and retained
fit precede a fresh complete actor-only assessment. The audit preserves every
assessment and corrective fit. The actor then receives all 2,000 configured PPO
updates; supervised updates remain separate from PPO telemetry.

The stock-Franka scenario task uses Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG`
arm gains and robot gravity compensation. Physics runs at 200 Hz with four
substeps per 50 Hz policy action; episode duration and PPO update counts retain
their configured values. This finer timestep reduces numerical motion during
object holding. The object retains gravity and its sealed scenario's friction
and mass. Training and held-out execution share this configuration,
and their `robot_control` audit records the actual gains and clock as
`stock-franka-high-pd-200hz-v1`.

The training teacher commands a downward XYZW `(1, 0, 0, 0)` hand orientation,
limits each TCP correction to 10 cm, and uses the native Cartesian action
scaling. Position actions use a 0.1 m scale, so the default initial action-noise
standard deviation of 0.005 corresponds to 0.5 mm per translational coordinate.
Collection prints measured
phase counts and strict training-placement coverage after each episode cycle;
failed teacher-only coverage reports its actual count and stops before fitting.
Both accepted warmup rounds must establish stable placement in at least half
of the complete training split. If the first mixed collection falls below this
guard, the actor receives another 32-epoch fit from a newly collected teacher-only
round that passes the same guard, then retries the mixture. `warmup_recovery`
retains every failed mixed collection and teacher recovery fit, including their
native replay and normalization counts. Failed mixtures never stand in for a
successful demonstration round or actor-only handoff.

`NPA_BYO_ISAAC_BOOTSTRAP=auto` is the default for fresh stock-Franka PPO with a
consumed training URI. `none` disables initialization; an explicit
`franka-cartesian-gru-distillation-v2` rejects other tasks, custom robots, legacy
physics, or resumed checkpoints. Existing entropy, learning-rate, and action-noise
settings retain their operator overrides. The native job receives
`ROBOT_BOOTSTRAP_PROFILE`, and its `bootstrap-audit.json` binds the profile,
training-scenario SHA-256, demonstration successes, initialized normalization,
recurrent sequences, supervised fitting losses, actor-only handoff, and zero
PPO updates at initialization. After the successful actor-only handoff, the
native normalizer freezes its trained sample count so PPO cannot silently change
the learned actor's input scale. The default demonstration and resumed stock
transport/convergence passes use a fixed PPO learning rate of `0.00001`, zero
entropy, and initial/convergence action-noise standard deviation `0.005`.
Existing explicit entropy, learning-rate, and noise overrides still apply;
custom robots and legacy tasks retain their task configuration. `none` disables
both demonstration initialization and its automatic normalization freeze.
Every native periodic and final checkpoint
records the policy's action and model semantics; loading a stock checkpoint
with missing or changed semantics fails. Checkpoints also bind the actual
normalization count and update bound; loading restores the saved bound only
after it agrees with the loaded statistic tensors. The adapter rejects missing or
contradictory audit evidence. Validation and Gold execute the saved learned
actor alone; they do not run the demonstration teacher or a post-actor
controller. Training demonstration success does not establish held-out quality.
After the complete configured PPO pass, validation includes the first saved
checkpoint as well as every configured periodic checkpoint and the final
checkpoint. Native checkpoint index zero contains one completed PPO update;
discarding it can lose the strongest learned warm start. The selected
checkpoint's update count remains separate from the completed pass count.

A checkpoint file alone does not prove that the actor learned to grasp. Until
at least half of its validation episodes achieved both stable grasp and lift,
resumed PPO retains the exploration schedule and repeats the training-only goal
curriculum. Once validation proves that milestone, the next pass learns transport
toward the exact goals. The stock recurrent profile preserves its small fixed
updates and trained normalization across transport and placement consolidation.
The placement-consolidation milestone requires at least half of validation episodes to bring
the grasped, lifted object within 8 cm of the goal. That 8 cm milestone selects a
training phase; strict success still requires the unchanged 5 cm distance,
0.03 m/s speed, and three consecutive stable steps. Missing distance evidence
keeps the transport phase. `resume_curriculum` records the decision, paired-skill
and approach counts, and exact checkpoint identity. Stage 9 sets
`NPA_BYO_ISAAC_RESUME_PHASE=exploration|transport|convergence` and disables unvalidated
automatic resume; direct compatibility callers retain `convergence` as their
default. Gold is excluded from this decision, and its goals and strict predicates
remain fixed. Reduced plumbing proofs may override the update count explicitly;
effectiveness runs should retain the production default.

Native Isaac rollout and evaluation jobs upload camera and point-cloud files
with 16 file workers, then publish completion metadata. Typed storage transport
and service failures retry without a run deadline; a rejected upload fails the
job and stops sibling retries. Evaluation also fails if a declared camera file
cannot be synchronized locally. Stage 7 downloads its declared raw camera frames
concurrently and fails before manifest publication when any transfer fails.
Stage 14 reads every earlier
outer-loop evidence document to retain all completed PPO curves, while limiting
camera downloads to the final outer loop and its selected Gold footage.
## Resume and verify

Restore the **same** spec, ID, images/source SHA, target, input/preset arguments,
and secret-name arrays. Inspect status first and ensure the original driver has
stopped. Never run two runtime drivers for one run.

```bash
npa/.venv/bin/npa workbench workflow status "$RUN_ID" --project "$NPA_PROJECT"
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  "${TARGET_ARGS[@]}" "${WORKFLOW_ARGS[@]}" "${SECRET_ARGS[@]}" \
  --runtime --resume-run "$RUN_ID" --durable-s3 --max-wait-seconds 0
```

Resume reconciles recorded jobs and completed waves. Target, credential, image,
and accelerator checks still run; adopted jobs do not need a second free
allocation. A genuinely terminal failed wave remains failed. After fixing its
cause, add `--retries 1` to this command to request one new attempt. Queued
capacity waits do not themselves consume recovery attempts. See
[durable runtime behavior](sim2real-durable-controller.md).

Use the artifact listing's exact S3 locations to download selected objects with
the Nebius storage console or your S3 client, using the selected project's
storage endpoint. The optional [AWS CLI transfer examples](scoped-storage-transfers.md)
show explicit endpoint and credential-profile selection. Inspect:

| Artifact | Required evidence |
| --- | --- |
| `reports/sim2real-report.json` | Exactly 14 canonical ComponentRecords; Stage 12 is `SEAM`, all others are `WORKS` |
| Selected checkpoint and validation/gold reports | Matching SHA/size, disjoint splits, strict success, and gold render lineage |
| `reports/sim2real.rrd` | Non-empty, independently decoded rollout, critique, training, validation, and gold timeline |
| `reports/sim2real.mcap` | Non-empty, independently decoded camera and evaluation evidence |
| `npa-workflow/runtime.json` | Completed graph, not merely some successful jobs |

The measured development run emitted a 15.4 GB RRD and 10.5 GB MCAP. Inspect
the artifact sizes and available disk space before downloading both; leave room
for the selected files and their readers. Open downloaded recordings with the
[viewer guides](../rerun-sharing.md).
Completion proves orchestration; report the measured policy result even when it
misses the threshold. Missing or corrupt finalization outputs mean completion
is unverified. The [architecture audit](../../architecture/sim2real-compositional-workflow.md)
defines full provenance and restart evidence.

## Clean up

Retain needed outputs/settings first. Stop this run's driver and cancel through
NPA with its original project/run identity:

```bash
npa/.venv/bin/npa workbench workflow cancel "$RUN_ID" --project "$NPA_PROJECT" --json
npa/.venv/bin/npa workbench workflow status "$RUN_ID" --project "$NPA_PROJECT"
```

Require verified terminal jobs and no active workers before retiring their
controller or cluster. If cancellation cannot verify the original controller,
use [controller recovery](../controller-recovery.md); a new environment's empty
queue does not prove cleanup.

For the isolated controller configured in this guide, retain the same private
client settings and retire it after cancellation verifies no active jobs:

```bash
npa/.venv/bin/npa skypilot cleanup-controller \
  --project "$NPA_PROJECT" --context "$NPA_CLUSTER" --yes --json
```

This command checks controller ownership and refuses active jobs. It leaves the
GPU pool and shared filesystem available to their owner. Follow the
[owned local workflow API cleanup](../../teardown.md#owned-local-workflow-api)
instructions after all clients using this run's isolated API have stopped.

Idle GPU clusters keep billing. For a **dedicated project created for this
run**, preview ordered teardown and review its exact resource inventory:

```bash
npa/.venv/bin/npa destroy --project "$NPA_PROJECT" --all
```

Only add `--yes` after reviewing the inventory and saving required artifacts.
The plan can include buckets, controllers, clusters, and resources beyond this
run. On shared infrastructure, cancel your run and coordinate pool retirement
with its owner. Follow [teardown](../../teardown.md) for controller → cluster →
owned storage ordering. Local `npa cleanup` does not stop cloud spend. Keep
recovery identity until remote deletion is verified.
