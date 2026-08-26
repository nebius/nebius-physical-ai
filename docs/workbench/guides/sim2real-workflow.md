# Compositional Sim2Real operator runbook

This is the onboarding source of truth for the canonical 14-stage workflow:
[`sim2real.yaml`](../../../npa/workflows/workbench/npa-workflows/sim2real.yaml).
Complete the gates in order. A production submit repeats the decisive S3,
model-access, cluster-object, immutable-image, and image-pull checks before it
creates a run or launches work.

## 1. Accept the exact third-party terms

The canonical runtime downloads one gated checkpoint under the operator's
Hugging Face account. Sign in, review the applicable model terms, and
request/accept access:

- [`nvidia/Cosmos-Transfer2.5-2B`](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)

Stage 8's second evaluator is the hosted `nvidia/Cosmos3-Super-Reasoner` through
Nebius Token Factory. Its model classification is OpenMDW-1.1; retain NVIDIA
Cosmos origin and attribution notices when distributing model materials. NPA
does not distribute or cache those hosted model weights and does not add a
second EULA boolean.

Isaac runtime warming and execution additionally require the operator to review
the [NVIDIA Omniverse terms](https://docs.omniverse.nvidia.com/usd/latest/common/NVIDIA_Omniverse_License_Agreement.html),
[Isaac Sim additional licenses](https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html),
and [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/).
NPA's non-interactive Isaac policy defaults the public `ACCEPT_EULA` value to
`Y` and derives Kit's internal value only inside the launcher. Use
`--no-accept-eula` (or a recognized negative `ACCEPT_EULA` value) to opt out;
privacy and telemetry consent remain independent and disabled by default.

Create a read token and verify it can access Cosmos Transfer.
Configure `NEBIUS_TOKEN_FACTORY_KEY` only in the private credential store or
runtime environment, then verify the key-specific hosted model list and a
minimal inference (which also fails closed for an unusable balance):

```bash
export HF_TOKEN='<hugging-face-read-token>'
npa/.venv/bin/npa configure --no-interactive --save-env-credentials
npa/.venv/bin/npa workbench health access --capability sim2real
npa/.venv/bin/npa workbench token-factory models
```

Expected: one Sim2Real `HF access ok` line, the exact
`nvidia/Cosmos3-Super-Reasoner` model ID, and zero exit statuses. A `401` means the
token is invalid or did not reach the check; a `403` means the account has not
accepted access or a fine-grained token omits that repository. See
[Hugging Face setup](../huggingface-token.md). `NGC_API_KEY` is not required by
the submitted runtime when all five images are already in the selected registry;
it may be required by a separate image-build/source-fetch path.

## 2. Configure Nebius, storage, Kubernetes, and SkyPilot

Start with the shared [quickstart](../../quickstart.md) and
[Kubernetes setup](../kubernetes.md). Resolve the project once, ensure S3 and
the cluster, select the exact kube context, and bootstrap the pinned SkyPilot:

```bash
export NPA_PROJECT='<configured-project-alias>'
export NPA_CLUSTER='<cluster-name>'

npa/.venv/bin/npa configure --show
npa/.venv/bin/npa provision-if-absent --project "${NPA_PROJECT}" --dry-run --output-format json
npa/.venv/bin/npa provision-if-absent --project "${NPA_PROJECT}"
export KUBECONFIG="${HOME}/.npa/clusters/${NPA_CLUSTER}/kubeconfig"
npa/.venv/bin/npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa/.venv/bin/npa skypilot status --bin-path)"
"${NPA_SKYPILOT_BIN}" check kubernetes
```

Expected: the dry run names the intended S3/Kubernetes actions without changing
them; the real command converges them; `sky check` reports Kubernetes enabled.
If the kubeconfig path differs, use the path printed by `npa cluster status`.
Clear stale ambient bearer tokens if authentication disagrees with the Nebius
CLI: `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN`.

The workflow runtime needs its storage credentials inside every wave and the
Token Factory key in the hosted Stage 8 leaf. Request propagation by secret
name even when values come from the selected project's private NPA credential
store; never put values in YAML, receipts, logs, or reports.

## 3. Add a schedulable CPU pool before GPU work

SkyPilot's Kubernetes jobs controller requests 2 vCPU/8 GiB. Sim2Real CPU states
request 8 vCPU/32 GiB and deliberately use the small
`npa-sim2real-control` image. Give them a Ready, schedulable, appropriately
untainted node with enough allocatable CPU and memory. It may also advertise GPUs;
the CPU profile has no GPU exclusion:

```bash
npa/.venv/bin/npa cluster node-group add-cpu \
  --cluster-name "${NPA_CLUSTER}" \
  --name sim2real-cpu \
  --platform cpu-e2 \
  --preset 16vcpu-64gb \
  --node-count 1 \
  --wait

kubectl get nodes -o custom-columns=NAME:.metadata.name,READY:.status.conditions[-1].status,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory,GPU:.status.allocatable.nvidia\.com/gpu
```

Expected: at least one Ready row with roughly 16 CPU and 64 GiB memory.
The larger preset is intentional: Kubernetes reserves part of nominal node
capacity, so an `8vcpu-32gb` node cannot fit a pod that requests the full
8 vCPU/32 GiB profile. A separate `8vcpu-32gb` pool is sufficient for the
controller alone, but not for the canonical Sim2Real CPU states.
If the preflight reports no fitting CPU node, remove `NoSchedule`/`NoExecute`
taints that the tasks do not tolerate or add/resize this pool.

## 4. Create Kueue admission objects and warm Isaac once

The canonical defaults name the `sim2real-gpu` LocalQueue and
`sim2real-production` PriorityClass. Create the queue objects with quotas that
cover the cluster's actual concurrent GPU, CPU, and memory requests; the helper
generates the exact repository-owned schemas:

```bash
export NPA_GPU_PRODUCT='<exact nvidia.com/gpu.product label from kubectl get nodes>'
export NPA_GPU_QUOTA='<concurrent GPU count>'
export NPA_CPU_QUOTA='<aggregate CPU quota, for example 64>'
export NPA_MEMORY_QUOTA='<aggregate memory quota, for example 512Gi>'

npa/.venv/bin/python - <<'PY' | kubectl apply -f -
import os
import yaml
from npa.workflows.sim2real.job_scheduling import kueue_queue_manifests

docs = kueue_queue_manifests(
    namespace="default",
    gpu_product=os.environ["NPA_GPU_PRODUCT"],
    gpu_quota=int(os.environ["NPA_GPU_QUOTA"]),
    cpu_quota=os.environ["NPA_CPU_QUOTA"],
    memory_quota=os.environ["NPA_MEMORY_QUOTA"],
)
print(yaml.safe_dump_all(docs, sort_keys=False))
PY

kubectl get localqueue.kueue.x-k8s.io sim2real-gpu -n default
kubectl get priorityclass sim2real-production
```

Expected: both `get` commands return their named object. Missing Kueue CRDs mean
Kueue must be installed first; a queue with insufficient CPU or memory quota can
leave a GPU Job suspended even when a GPU is free.

Choose the digest-pinned Isaac image now, then warm a shared RWX cache. The
template is the authoritative PVC/security/bootstrap contract:

```bash
export NPA_ISAAC_IMAGE='<registry>/npa-isaac-lab@sha256:<64-hex>'
sed "s|image: ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab@sha256:<64-hex-digest>|image: ${NPA_ISAAC_IMAGE}|" \
  npa/docker/workbench/common/warm-isaac-cache.yaml | kubectl apply -f -
kubectl wait --for=condition=complete job/npa-warm-isaac-cache --timeout=-1s
kubectl logs job/npa-warm-isaac-cache
kubectl get pvc npa-isaac-cache -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,MODES:.status.accessModes
```

Expected: the Job completes, its log ends with a successful bootstrap, and the
PVC is `Bound` with `RWX`. Exit 78 means acceptance was explicitly disabled;
image pull failures are handled in the next gate. See
[runtime-fetch packaging](../container-packaging.md#nvidia-isaac--omniverse-runtime-fetch-images).

## 5. Build/push once and prove the exact image pulls

The workflow does not copy images between registries. Use the public GHCR
defaults, or build/push modified runtime images to an explicitly selected private
registry. Never submit tags: resolve and retain immutable `@sha256:` references
for these five config keys:

| Config key | Required image |
| --- | --- |
| `controller_image` | `npa-sim2real-control` |
| `transfer_image` | `npa-cosmos2-transfer` |
| `envgen_image` | `npa-envgen` |
| `isaac_image` | `npa-isaac-lab` (same bytes used to warm the cache) |
| `viewer_image` | `npa-rerun-viewer` |

Relevant build entrypoints are
`npa/docker/workbench/sim2real-build.sh`,
`npa/docker/workbench/cosmos2-transfer/build.sh`, and
`npa/docker/workbench/isaac-lab/build.sh`. Follow
[build and push](../container-packaging.md) when images are absent.

Put the six references in shell variables, then reproduce the actual manifest
pulls with the same config used by submit:

```bash
export CONTROLLER_IMAGE='<registry>/npa-sim2real-control@sha256:<64-hex>'
export TRANSFER_IMAGE='<registry>/npa-cosmos2-transfer@sha256:<64-hex>'
export ENVGEN_IMAGE='<registry>/npa-envgen@sha256:<64-hex>'
export ISAAC_IMAGE="${NPA_ISAAC_IMAGE}"
export VIEWER_IMAGE='<registry>/npa-rerun-viewer@sha256:<64-hex>'
export SPEC=npa/workflows/workbench/npa-workflows/sim2real.yaml

npa/.venv/bin/npa workbench workflow preflight-images "${SPEC}" \
  --project "${NPA_PROJECT}" \
  --infra "k8s/${NPA_CLUSTER}" \
  --assume-decision promote_checkpoint \
  --var controller_image="${CONTROLLER_IMAGE}" \
  --var transfer_image="${TRANSFER_IMAGE}" \
  --var envgen_image="${ENVGEN_IMAGE}" \
  --var isaac_image="${ISAAC_IMAGE}" \
  --var viewer_image="${VIEWER_IMAGE}"
```

Expected: every image is pullable and bootstrap-compatible. `not_found` means
build/push the printed image; `forbidden` means fix the exact-host registry
credential. Public GHCR releases need no credential. Private images require an
explicit credential or operator-managed Kubernetes Docker config secret; NPA
does not mint or refresh either. See
[registry troubleshooting](../troubleshooting/known-footguns.md#private-registry-credentials-expire).

## 6. Validate, plan, and submit

Set a real bucket and task-aligned seed prefix. The trigger prefix and its
`dataset-manifest.json` must already be readable; customer data contracts are in
[Sim2Real customer assets](sim2real-customer-assets.md).

### Explicit public Franka-lift seed

`public-franka-lift` is an opt-in seed preset; it is never the silent production
default. It runtime-fetches the anonymous public Hugging Face dataset
[`huyyyyan/pi05-Isaac-sim_Franka_lift_cube`](https://huggingface.co/datasets/huyyyyan/pi05-Isaac-sim_Franka_lift_cube)
at immutable revision `42c181e40a43afb1702c29d6f24d5de25219aff8`. Upstream
metadata declares Apache-2.0. NPA does not vendor or bake its bytes and does not
invent a dataset-acceptance flag: the stager verifies anonymous access, the exact
revision, and the declared license before downloading.

Stage the minimal real subset first, using the same bucket and run ID as the
workflow. The command uploads one source episode's Parquet actions, both source
MP4s, four decoded `camera-*.png` frames, `actions.json`, a sample-rollout
manifest, and `dataset-manifest.json`. Counts, byte sizes, and SHA-256 hashes are
derived from those objects; `dataset-manifest.json` is uploaded last.

```bash
export RUN_ID="sim2real-$(date -u +%Y%m%dT%H%M%SZ)"
export NPA_BUCKET='<bucket-name>'

npa/.venv/bin/npa workbench workflow trigger stage-preset \
  --preset public-franka-lift \
  --project "${NPA_PROJECT}" \
  --bucket "${NPA_BUCKET}" \
  --run-id "${RUN_ID}" \
  --output-format json

npa/.venv/bin/npa workbench workflow validate-spec "${SPEC}" \
  --preset public-franka-lift --json
npa/.venv/bin/npa workbench workflow plan-spec "${SPEC}" \
  --preset public-franka-lift --run-id "${RUN_ID}" \
  --var bucket="${NPA_BUCKET}" --waves \
  --assume-decision promote_checkpoint

npa/.venv/bin/npa workbench workflow submit "${SPEC}" \
  --preset public-franka-lift --project "${NPA_PROJECT}" \
  --infra "k8s/${NPA_CLUSTER}" --runtime --run-id "${RUN_ID}" \
  --var bucket="${NPA_BUCKET}" \
  --var controller_image="${CONTROLLER_IMAGE}" \
  --var transfer_image="${TRANSFER_IMAGE}" \
  --var envgen_image="${ENVGEN_IMAGE}" \
  --var isaac_image="${ISAAC_IMAGE}" \
  --var viewer_image="${VIEWER_IMAGE}" \
  --var isaac_cache_pvc=npa-isaac-cache
```

The source contract remains `Isaac-Lift-Cube-Franka-IK-Rel-v0`, Franka, two
cameras (`image`, `wrist_image`), and 7D IK-relative actions. These actions and
cameras are seed/conditioning evidence only. The manifest separately records the
canonical workflow boundary: 8D joint-delta-plus-gripper policy actions and
three evaluation cameras (`primary`, `side`, `overhead`). No source action is
silently reused as PPO input, and the strict stable-placement metric remains 5
cm. For a custom/private dataset, omit `--preset` and continue to use the
existing `--var dataset_id=...`, `trigger_uri=...`, and `seed_manifest_uri=...`
path.

```bash
export RUN_ID="sim2real-$(date -u +%Y%m%dT%H%M%SZ)"
export NPA_BUCKET='<bucket-name>'

npa/.venv/bin/npa workbench workflow validate-spec "${SPEC}" --json
npa/.venv/bin/npa workbench workflow plan-spec "${SPEC}" \
  --run-id "${RUN_ID}" --waves --assume-decision promote_checkpoint \
  --var bucket="${NPA_BUCKET}" \
  --var controller_image="${CONTROLLER_IMAGE}" \
  --var transfer_image="${TRANSFER_IMAGE}" \
  --var envgen_image="${ENVGEN_IMAGE}" \
  --var isaac_image="${ISAAC_IMAGE}" \
  --var viewer_image="${VIEWER_IMAGE}" \
  --var isaac_cache_pvc=npa-isaac-cache
```

Expected: validation reports valid, and the wave plan shows the 14-stage graph
with the Stage 4 parallel wave and direct Stage 7 → hosted Stage 8 → Stage 9
sequence. Then submit through the durable runtime:

```bash
npa/.venv/bin/npa workbench workflow submit "${SPEC}" \
  --project "${NPA_PROJECT}" \
  --infra "k8s/${NPA_CLUSTER}" \
  --runtime --resume --max-wait-seconds 0 \
  --run-id "${RUN_ID}" \
  --var bucket="${NPA_BUCKET}" \
  --var trigger_uri="s3://${NPA_BUCKET}/sim2real-triggers/${RUN_ID}/" \
  --var seed_manifest_uri="s3://${NPA_BUCKET}/sim2real-triggers/${RUN_ID}/dataset-manifest.json" \
  --var controller_image="${CONTROLLER_IMAGE}" \
  --var transfer_image="${TRANSFER_IMAGE}" \
  --var envgen_image="${ENVGEN_IMAGE}" \
  --var isaac_image="${ISAAC_IMAGE}" \
  --var viewer_image="${VIEWER_IMAGE}" \
  --var isaac_cache_pvc=npa-isaac-cache \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY
```

Before any launch, submit now fails with one consolidated prerequisite report if
the required secret propagation, three gated model probes, CPU node, cache PVC,
Kueue queue, PriorityClass, S3 write probe, immutable images, or image pulls are
not ready. `--skip-preflight` is an expert escape hatch and is not part of this
runbook.

For a reduced plumbing proof, add `--var outer_iterations=1 --var
inner_iterations=1` and deliberately chosen smaller scenario/PPO values. Do not
change the strict 5 cm metric or sealed gold contract. Do not add arbitrary run
deadlines; `--max-wait-seconds 0` keeps durable status in
`s3://<bucket>/sim2real/<run-id>/npa-workflow/runtime.json`.

## Resume and verify

```bash
npa/.venv/bin/npa workbench workflow status "${RUN_ID}" --project "${NPA_PROJECT}" --watch
# after an operator/controller restart:
npa/.venv/bin/npa workbench workflow submit "${SPEC}" \
  --project "${NPA_PROJECT}" --infra "k8s/${NPA_CLUSTER}" \
  --runtime --resume-run "${RUN_ID}" --max-wait-seconds 0 \
  <the same --var and --secret-env arguments>
```

Completion must include `reports/sim2real-report.json`, non-empty
`reports/sim2real.rrd` and `reports/sim2real.mcap`, the selected checkpoint, and
exact validation/gold lineage. Pipeline completion proves orchestration, not
policy efficacy; report the measured strict success without weakening it. The
[architecture/resume contract](../../architecture/sim2real-compositional-workflow.md)
defines the 14 ComponentRecords and restart audit.
