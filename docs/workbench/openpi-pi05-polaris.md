# OpenPI pi0.5 Polaris: direct, serve, train, and evaluate

The connected OpenPI workflow family has two deliberately separate surfaces:

- `byof-openpi.yaml` packages and byte-scans the immutable upstream
  `Physical-Intelligence/openpi@15a9616a00943ada6c20a0f158e3adb39df2ccac`
  source, pushes it to the operator's private registry, and resolves the image
  to a digest. Its original direct plus same-pod WebSocket smoke remains a
  regression gate.
- `openpi-pi05-four-mode.yaml` consumes only that digest and runs the live
  negative terms probe, direct inference, private cross-pod serving, real
  optimizer/checkpoint work, and disjoint held-out evaluation.

The builder uses CUDA 12.8, compiles an `sm_100` runtime probe, and retains
upstream's pinned JAX CUDA 12 stack. A CUDA 13.0 managed-driver MK8s node is
backward compatible with that userspace stack. GPU artifacts record the actual
driver, XLA platform, JAX/JAXlib, GPU product, allocated GPU count, compute
capability, compiled-kernel proof, timings, and available peak-memory counters.

## Policy and checkpoint

- Config: `pi05_droid_jointpos_polaris`
- Checkpoint: `gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris`
- Embodiment: Franka Panda, seven arm joints plus one parallel-jaw gripper
- Output: a 15-step action chunk with eight dimensions. Dimensions 0–6 are
  joint-position targets in radians; dimension 7 is the gripper target. They
  are not joint velocities.

The checkpoint and PaliGemma tokenizer are fetched only after the workload
starts, into operator/project-controlled runtime storage. They are never copied
into the distributable image or uploaded with inference evidence. Provenance
includes a SHA-256 over the checkpoint's canonical manifest of GCS object names,
generations, sizes, MD5 values, and CRC32C values, plus the tokenizer's exact GCS
generation, size, MD5, and CRC32C. The public GCS sources are opened
anonymously; the workload does not probe or require Google application
credentials.

OpenPI source is Apache-2.0. The CUDA base/runtime retains NVIDIA's upstream
license terms. This BYOF result stays in the operator's private project registry
and is not classified for public redistribution. Checkpoint weights, input
frames, and robot state are not baked into it. The checkpoint contains
Gemma-derived material: before any image build or checkpoint fetch, the
operator must review the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy),
then provide the exact run-scoped gate:

```bash
export NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES
```

NPA forwards that value through SkyPilot's secret channel. The four-mode
server places it in an ephemeral Kubernetes Secret referenced by the server
pod, then deletes and independently verifies that Secret absent. Acceptance is
not a workflow default, image environment variable, build argument, repository
file, or project credential. No other agreement is inferred from it.

## Request, response, and service contract

| Field | Type and shape |
| --- | --- |
| `observation/exterior_image_1_left` | `uint8[224,224,3]` |
| `observation/wrist_image_left` | `uint8[224,224,3]` |
| `observation/joint_position` | `float32[7]` |
| `observation/gripper_position` | `float32[1]` |
| `prompt` | string, for example `pick up the fork` |

Direct and service requests use deterministic synthetic frames and a valid
neutral Franka state. Both direct `policy.infer` and each of two upstream
WebSocket client requests must return finite `float64[T,8]` with `T >= 5`.

The four-mode server is a digest-pinned Kubernetes Deployment behind a private
ClusterIP Service with readiness and liveness probes. A distinct CPU-only
client Job connects through the Service DNS name; server and client pod UIDs
must differ. There is no Ingress or public load balancer. The older builder
smoke remains explicitly labeled same-pod loopback.

All service waits are bounded by monotonic, configurable failure-recovery
deadlines. The Deployment progress deadline matches server readiness, and the
client Job has `activeDeadlineSeconds` in addition to `backoffLimit: 0`.
Pending/Unschedulable placement, image-pull failures, failed probes, uncertain
API responses, and stuck deletion/finalizers fail closed into exact-identity
cleanup. These are service recovery controls, not workflow, job-count, cost, or
operator runtime caps.

The controller Role name-scopes Secret, Service, Deployment, and Job
`get`/`delete` permissions. It cannot read a foreign Secret, read pod logs, or
delete pods. Kubernetes cannot name-scope `create`, and Deployment/Job pod names
are controller-generated, so `create` on the four resource kinds and pod
`list` remain namespace-wide residuals. The controller always applies a run
label selector and verifies immutable ownership before acting. Client evidence
comes from the Job termination message, and server hardware evidence comes
from a private in-cluster diagnostic endpoint—not pod-log access.

A robot consumer should execute about five position targets at 15 Hz, observe
again, and re-query the policy.

The production consumer for that contract is the
[Antioch / Isaac Sim Franka bridge](antioch-openpi-franka.md). It places the
renderer on RTX PRO 6000, retains this B200 service and wire protocol, and
fails to no-action on timeout, reconnect exhaustion, malformed responses, or
unsafe absolute targets.

## Tiny training and held-out evaluation contract

The workflow generates a deterministic NPZ at run time with two uint8
`224x224` cameras, seven float32 Franka joints, one float32 gripper value,
prompts, and 15x8 absolute joint/gripper targets. Training and held-out sample
IDs and full-content hashes are distinct. Normalization statistics are computed
from the training split only.

Training uses upstream's supported pi0.5 PaliGemma/action-expert LoRA variants,
`CheckpointWeightLoader` for the Polaris base, upstream `init_train_state`, and
upstream JIT-compiled `train_step`. It records finite loss/gradient metrics,
real forward/backward/AdamW work, different before/after trainable state hashes,
a positive finite update norm, and an Orbax checkpoint that is saved and
reloaded. The full reloadable checkpoint is stored under the run's private S3
prefix; weights never enter Git or the distributable image.

Evaluation independently downloads and hashes that exact checkpoint, rejects
any train/held-out ID overlap, runs upstream `compute_loss(train=False)`, and
reports held-out model loss plus trajectory MAE/MSE against the absolute
targets. It also reloads the policy surface and requires a valid finite
trajectory.

## Validate and run

Validate and render both connected workflows:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-openpi.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-openpi.yaml \
  --run-id openpi-builder-plan

npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/openpi-pi05-four-mode.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/openpi-pi05-four-mode.yaml \
  --run-id openpi-four-mode-plan \
  --var runtime_image=cr.<region>.nebius.cloud/<registry>/openpi@sha256:<digest>
```

On a fresh isolated B200 MK8s context, the canonical E2E builds the source,
executes the declared editable install and `nvcc -arch=sm_100` build, pushes and
resolves the image, scans the built bytes, preserves the historical negative
and inference regression, and submits the connected graph through
`npa workbench workflow submit`:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_BYOF_OPENPI_LIVE_B200=1 \
NPA_E2E_PROJECT=<project-alias> \
NPA_E2E_S3_BUCKET=<existing-project-bucket> \
NPA_BYOF_S3_ENDPOINT=https://storage.<bucket-region>.nebius.cloud \
NPA_BYOF_OPENPI_PROJECT_REGISTRY=cr.<region>.nebius.cloud/<project-registry> \
NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES \
npa/.venv/bin/python -m pytest -q -s \
  npa/tests/e2e/test_byof_openpi_polaris_live_e2e.py
```

For a manual four-mode submission, use the immutable image value returned by
the builder. Source staging is automatic; `--stage-src` makes the branch-code
dependency explicit. `--max-wait-seconds 0` waits indefinitely.

```bash
export OPENPI_IMAGE='cr.<region>.nebius.cloud/<registry>/openpi@sha256:<digest>'
export OPENPI_RUN_ID="openpi-four-mode-$(date -u +%Y%m%dT%H%M%SZ)"
export OPENPI_NAMESPACE=default
export OPENPI_KUBECONFIG='<task-owned-kubeconfig>'
export OPENPI_CONTEXT='<isolated-context>'
export OPENPI_SERVICE_ACCOUNT
OPENPI_SERVICE_ACCOUNT="$(npa/.venv/bin/python -c \
  'import os; from npa.workflows.byof.openpi_service import controller_service_account_name; print(controller_service_account_name(os.environ["OPENPI_RUN_ID"]))')"

# Fail closed if this exact name is foreign. Name-scoped get/delete rules cover
# the exact Secret, Service, Deployment, and Job. Namespace-wide scope is limited
# to Kubernetes' unavoidable create rules and pod list; pod logs and pod delete
# are not granted.
npa/.venv/bin/python -m npa.workflows.byof.openpi_service_rbac apply \
  --run-id "$OPENPI_RUN_ID" \
  --namespace "$OPENPI_NAMESPACE" \
  --service-account "$OPENPI_SERVICE_ACCOUNT" \
  --kubeconfig "$OPENPI_KUBECONFIG" \
  --context "$OPENPI_CONTEXT" \
  --delete-timeout-seconds 120 \
  --poll-interval-seconds 5 \
  --api-timeout-seconds 30

npa/.venv/bin/npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/openpi-pi05-four-mode.yaml \
  --run-id "$OPENPI_RUN_ID" \
  --runtime \
  --max-wait-seconds 0 \
  --poll-seconds 30 \
  --stage-src \
  --infra "k8s/${OPENPI_CONTEXT}" \
  --project '<project-alias>' \
  --registry 'cr.<region>.nebius.cloud/<registry>' \
  --config-path '<task-owned-skypilot-config.yaml>' \
  --var 'bucket=<operator-project-bucket>' \
  --var "prefix=oss-solutions/openpi/${OPENPI_RUN_ID}" \
  --var "runtime_image=${OPENPI_IMAGE}" \
  --var 'gpu_type=B200' \
  --var 'gpu_count=1' \
  --var 'expected_gpu_type=B200' \
  --var 'expected_compute_capability=10.0' \
  --var "service_namespace=${OPENPI_NAMESPACE}" \
  --var "service_account=${OPENPI_SERVICE_ACCOUNT}" \
  --var 'service_gpu_node_selector_value=B200' \
  --var 'service_server_ready_timeout_seconds=1200' \
  --var 'service_client_timeout_seconds=600' \
  --var 'service_cleanup_timeout_seconds=180' \
  --var 'service_api_timeout_seconds=30' \
  --var 'service_http_timeout_seconds=30' \
  --secret-env NPA_OPENPI_ACCEPT_GEMMA_TERMS \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY

# Run after the controller is terminal, including after a failed submission.
# Deletion first verifies the complete immutable run owner on every object.
npa/.venv/bin/python -m npa.workflows.byof.openpi_service_rbac delete \
  --run-id "$OPENPI_RUN_ID" \
  --namespace "$OPENPI_NAMESPACE" \
  --service-account "$OPENPI_SERVICE_ACCOUNT" \
  --kubeconfig "$OPENPI_KUBECONFIG" \
  --context "$OPENPI_CONTEXT" \
  --delete-timeout-seconds 120 \
  --poll-interval-seconds 5 \
  --api-timeout-seconds 30
```

The canonical live E2E wraps submission with those exact apply/delete calls and
fails if controller RBAC is foreign, broader than the declared contract, or not
independently absent afterward. The workflow itself also deletes and verifies
the Deployment, ClusterIP Service, client Job, and ephemeral terms Secret.
The live wrapper also proves that the controller ServiceAccount can read only
the deterministic terms Secret name and is forbidden from reading an otherwise
valid foreign Secret.

All resource, dataset, checkpoint, service, and output values are configurable.
When retargeting, change accelerator/count, expected hardware checks, service
node selector, CPU/memory/scratch values, dataset counts/location, checkpoint,
namespace, and output prefix together. No live cluster, project, registry, or
bucket identity is committed.

Terms refusal evidence uses an attempt-scoped key under
`diagnostics/terms-refusals`; it is never written to a state's declared success
URI. The live negative gate retains that diagnostic at its separate durable key
and references it from the accepted retry written to that same logical success
URI. Serving declares a single `serve_artifact_root_uri`; both `service.json`
and `cleanup.json` derive
from that root, so an output override cannot desynchronize code from the
workflow's declared artifacts.

`NPA_BYOF_OPENPI_REUSE_IMAGE` is rejected by the canonical release gate. A
previous image may be used for diagnosis, but does not replace fresh build and
byte-scan evidence.

## Live acceptance and limitations

The historical builder baseline uses one B200 (`sm_100`) on isolated reserved
MK8s capacity. It includes private-registry build/push/digest verification,
built-byte inspection, an exit-64 negative terms workload, and positive direct
plus same-pod inference. It fetched 27 checkpoint objects (12,434,530,837
bytes) at runtime and returned finite `float64[15,8]` trajectories.

The four-mode gate adds cross-pod ClusterIP topology, a real LoRA optimizer
update and checkpoint reload, and held-out offline metrics. Acceptance requires
all four artifacts, independent S3 readback, exact service cleanup, and final
provider teardown. It does **not** claim physical Franka task success, external
Ingress, model convergence, or robot success from offline evaluation.

The connected live qualification on 2026-08-16 used a fresh isolated MK8s
cluster with one physical B200 and allocated one B200 per GPU stage. Driver
580.159.04 exposed 183,359 MiB and compute capability 10.0; the runtime executed
the compiled `sm_100` probe. Direct inference returned finite
`float64[15,8]`. The private ClusterIP server and distinct CPU client completed
two finite `float64[15,8]` requests in 39.350 seconds cold and 50.2 milliseconds
warm, then exact Deployment, Service, Job, pod, and Secret cleanup passed.

The one-step upstream pi0.5 LoRA optimizer smoke reported loss 0.145676,
gradient norm 1.46636, and trainable update L2 0.0957375 with different
before/after hashes. Orbax saved and reloaded step 1; its private 29-file
checkpoint manifest covered 9,152,431,613 bytes. Evaluation independently
reloaded that exact manifest and used two samples excluded from the four-sample
training split. Mean upstream held-out loss was 0.182892, trajectory MAE was
0.0111408, and trajectory MSE was 0.000200538; the reloaded policy also emitted
a finite `float64[15,8]` trajectory. This is an optimizer/checkpoint and offline
evaluation smoke on tiny deterministic data, not convergence or physical robot
success.
