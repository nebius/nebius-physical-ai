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
`source_sha` empty. Obtain an image-set receipt from your image maintainer:

| Setting | Image and required capability |
| --- | --- |
| `CONTROLLER_IMAGE` | `npa-sim2real-control`, including current hosted Stage 8/9 support |
| `TRANSFER_IMAGE` | `npa-cosmos2-transfer`, with real Transfer inference |
| `ENVGEN_IMAGE` | `npa-envgen`, with real scenario generation |
| `ISAAC_IMAGE` | `npa-isaac-lab`, with rollout, PPO, held-out evaluation, and RTX rendering |
| `VIEWER_IMAGE` | `npa-rerun-viewer`, with RRD and MCAP finalization |
| `SOURCE_SHA` | Common full 40-hex commit baked into all five images as `NPA_IMAGE_SOURCE_SHA` |

Every image must be a complete `repository@sha256:<64-hex-digest>` reference.
The receipt must name the tested CLI/workflow checkout and establish
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
TRANSFER_IMAGE='<qualified-transfer-repository>@sha256:<64-hex-digest>'
ENVGEN_IMAGE='<qualified-envgen-repository>@sha256:<64-hex-digest>'
ISAAC_IMAGE='<qualified-isaac-repository>@sha256:<64-hex-digest>'
VIEWER_IMAGE='<qualified-viewer-repository>@sha256:<64-hex-digest>'
```

<a id="5-buildpush-once-and-prove-the-exact-image-pulls"></a>

For private images, prepare exact-host registry authorization and an existing
Kubernetes Docker config pull secret using the [registry setup](../troubleshooting/known-footguns.md#private-registry-credentials-expire).
NPA's manifest checks and Kubernetes node pulls need separate authorization.
Reference the secret in `kubernetes.pod_config.spec.imagePullSecrets` in your
private SkyPilot config. The cache-warming Job below is created directly by
`kubectl`, so its template also needs that secret; SkyPilot config does not
apply to it. Anonymous public GHCR images need neither credential.

## 3. Configure the project, storage, and model access

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
| `HF_TOKEN` | Gated Transfer checkpoint access |
| `NEBIUS_TOKEN_FACTORY_KEY` | Hosted Stage 8 evaluation; an IAM token cannot replace it |

Review and obtain access to all three gated Transfer dependencies under your
Hugging Face account:

- [Cosmos Transfer 2.5](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B), for inference weights.
- [Cosmos Predict 2.5](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B), for the pinned tokenizer.
- [Cosmos Guardrail](https://huggingface.co/nvidia/Cosmos-Guardrail1), for runtime safety weights.

Your read token must cover all three repositories. The access command probes
their pinned payloads; accepting Transfer alone leaves Stage 3 unusable.
`401` means invalid authentication; `403` means missing approval or token scope.
See [Hugging Face setup](../huggingface-token.md). `NGC_API_KEY` is not a runtime
prerequisite when all five images already exist; separate builds may need it.

Stage 8 needs image input and strict ordered JSON Schema output. This checkout
uses hosted MiniMax-M3; review its [model license](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE)
and hosted-service terms. The `cosmos3_model` setting and artifact names remain
for compatibility; results record the actual model and family. Do not select a
text-only model. For an authorized Cosmos 3 endpoint, select the exact supported
model ID and propagate `NEBIUS_TOKEN_FACTORY_BASE_URL` as a secret name so local
checks and remote evaluation use that same endpoint. Start a new run when
changing evaluator contracts.

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

Require validated RTX graphics drivers, a working shared filesystem on the
consuming node groups, and sufficient CPU capacity. If a CPU pool is absent,
its owner can add one:

```bash
npa/.venv/bin/npa cluster node-group add-cpu \
  --cluster-name "$NPA_CLUSTER" --name sim2real-cpu \
  --platform cpu-e2 --preset 16vcpu-64gb --node-count 1 --wait
```

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

## 5. Warm the selected Isaac image

Use the [shipped warming template](../../../npa/docker/workbench/common/warm-isaac-cache.yaml)
with the exact Isaac digest. It creates the RWX PVC and runs bootstrap as an
unprivileged user. Substitute only the image input:

```bash
CACHE_SETUP_DIR="$HOME/.npa/sim2real-setup/$NPA_CLUSTER"
mkdir -p "$CACHE_SETUP_DIR"
chmod 700 "$CACHE_SETUP_DIR"
sed "s|image: ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab@sha256:<64-hex-digest>|image: $ISAAC_IMAGE|" \
  npa/docker/workbench/common/warm-isaac-cache.yaml \
  > "$CACHE_SETUP_DIR/warm-isaac-cache.yaml"
kubectl --namespace "$NPA_NAMESPACE" apply -f "$CACHE_SETUP_DIR/warm-isaac-cache.yaml"
kubectl --namespace "$NPA_NAMESPACE" wait --for=condition=complete job/npa-warm-isaac-cache --timeout=-1s
kubectl --namespace "$NPA_NAMESPACE" logs job/npa-warm-isaac-cache
kubectl --namespace "$NPA_NAMESPACE" get pvc npa-isaac-cache
```

Private-registry users must include the prepared pull secret in the generated
Job's `spec.template.spec.imagePullSecrets` before applying it.
Expected: the Job logs show `ready=yes` for the selected image's `expected_tree`,
the Job succeeds, and the PVC is `Bound` with `ReadWriteMany`. Cold installation downloads several
gigabytes and can stay quiet while extracting wheels into the shared filesystem.
While waiting, inspect the Job and
pod events from another terminal; a failed Job never reaches `complete`.
Keep the rendered manifest outside Git because it can contain private registry
locations. The checked-in template remains unchanged.

After changing images, verify the old warming Job is terminal, delete only that
Job (`kubectl --namespace "$NPA_NAMESPACE" delete job npa-warm-isaac-cache`), and apply the newly rendered
template. Keep the PVC and older versioned trees. The selected image's
`isaac-bootstrap status` must report `ready=yes` for its own `expected_tree`;
matching wheel pins alone do not prove compatibility. See
[runtime-fetch packaging](../container-packaging.md#runtime-fetched-isaac-sim-why-the-isaac-images-are-publishable).

## 6. Keep one argument set for planning, submit, and resume

Create a fresh ID and choose compatible GPU concurrency (1–8). The single-GPU
cluster above uses `1`. Concurrency batches the eight EnvGen shards without
reducing training quality or iteration counts. Discovery and submit preflight
must establish capacity; a wave plan does not reserve it.

```bash
RUN_ID="sim2real-$(date -u +%Y%m%d-%H%M%S)"
NPA_GPU_CONCURRENCY=1
CONFIG_ARGS=(
  --var bucket="$NPA_BUCKET"
  --var source_sha="$SOURCE_SHA"
  --var controller_image="$CONTROLLER_IMAGE"
  --var transfer_image="$TRANSFER_IMAGE"
  --var envgen_image="$ENVGEN_IMAGE"
  --var isaac_image="$ISAAC_IMAGE"
  --var viewer_image="$VIEWER_IMAGE"
  --var isaac_cache_pvc=npa-isaac-cache
  --var gpu_concurrency="$NPA_GPU_CONCURRENCY"
  --var cosmos3_model=MiniMaxAI/MiniMax-M3
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
It uploads real Parquet actions, both MP4s, decoded frames, and hashed manifests.

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

## 7. Submit and follow the run

Keep the driver on the always-on operator machine. Laptop sleep or closing the
terminal can interrupt it while cluster jobs continue.

```bash
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  "${TARGET_ARGS[@]}" "${WORKFLOW_ARGS[@]}" "${SECRET_ARGS[@]}" \
  --run-id "$RUN_ID" --runtime --resume --durable-s3 --max-wait-seconds 0
```

Submit enforces storage writes, secret propagation, model access, cluster/cache
readiness, immutable source-attested images, pulls, and scheduler capacity before
GPU work. Resolve reported prerequisites through setup; `--skip-preflight` is
not part of this path.

Production defaults retain 10,000 scenarios, 64 rollouts, 32 action samples,
2,000 PPO updates per inner pass, and three inner/outer iterations. Validation
selects checkpoints; gold remains held out. Promotion uses strict 5 cm stable
placement and threshold `0.50`. Reduced plumbing runs must declare overrides
and cannot claim production efficacy. `--max-wait-seconds 0` removes the wave
wait deadline without weakening success predicates.

From a second shell, restore the project/run settings and inspect:

```bash
npa/.venv/bin/npa workbench workflow status "$RUN_ID" --project "$NPA_PROJECT" --watch
npa/.venv/bin/npa workbench workflow logs "$RUN_ID" --project "$NPA_PROJECT" --follow
npa/.venv/bin/npa workbench workflow artifacts "$RUN_ID" --project "$NPA_PROJECT"
```

Model loading and Transfer inference can stay quiet between log messages.
The first Isaac startup can also compile ray-tracing shaders on the CPU while
GPU utilization stays low. Its log names the Kit log file, where shader
compilation progress is recorded. PPO retains the full RSL-RL training log and
prints its console tail when a pass finishes, so workflow logs can also stay
quiet during optimizer updates. Check live status and pod events while waiting.
A running pod or GPU activity does not prove stage completion;
require the declared output artifacts and successful scheduler status.

Inspect primary, side, and overhead footage for the robot, object, and task.
Stage 8 sees primary frames; secondary views cannot compensate for occlusion.
Invalid capture needs corrected images and new rollouts, not relabelled metadata.
The [temporal-credit contract](sim2real-data-contracts.md#inner-loop-stages-79)
enforces action/frame and simulator-episode bindings; playback FPS is not
physical simulation time. Use measured PPO losses and each pass's own validation
result; compatibility proxy fields are not optimizer measurements. Gold
selection does not replace earlier candidate measurements.

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

Open downloaded recordings with the [viewer guides](../rerun-sharing.md).
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
