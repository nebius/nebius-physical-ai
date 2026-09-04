# Sim2Real live findings: evaluation and RTX remediation (2026-09-02)

This document began as an evaluation of five findings reported after the Living
Lab run against `origin/main` at `e5ddb7d25ef2af5485bb43409579657497220a77`.
The follow-up on the same branch now implements the repository and provisioning
fix for finding 5. Findings 1–4 remain audit results rather than changes made by
this pull request. The sections below keep the original finding, the implemented
driver contract, and the sanitized live-validation outcome distinct.

The referenced `sim2real-repo-fixes.md` was not present in the clone or found in
the repository's GitHub code search. The available originating context was the
operator-supplied summary and [PR #367](https://github.com/nebius/nebius-physical-ai/pull/367),
which merged after the original evaluation. PR #367's code changes
were inspected separately from the original evaluation baseline; its “Not included” section
independently reports the publication, `source_sha`, Kueue, and graphics gaps.

## Dispositions

| # | Disposition | Maintainer consequence |
|---:|---|---|
| 1 | **Valid in substance; public-set wording corrected** | A coherent five-image release is required. Current public state is worse than “five inconsistent images”: only four of the five roles are in the accepted public plan, and three of those four lack required attestations. |
| 2 | **Partially valid** | Add `source_sha` to rendered-plan and submit examples. Plain `plan-spec --waves` does not require it; `submit --plan-only` and execution do. |
| 3 | **Valid repository/operator gap; fresh-cluster generalization bounded** | The runbook needs an explicit, pinned Kueue installation prerequisite or an equivalent provisioned contract. |
| 4 | **Valid** | The access catalog omits a runtime-fetched, gated Predict2.5 tokenizer. The related loss of vendor diagnostics is also valid. |
| 5 | **Partially valid; actionable cluster gap fixed here** | NPA now has an explicit RTX rendering profile that selects RTX PRO 6000 plus the supported GPU Operator mounted-driver path and fails closed on GLX, EGL, and Vulkan device readiness. Existing defaults and the NVSwitch operator rejection remain unchanged. |

## 1. Exact-source and bootstrap-attested images

**Disposition: valid in substance.** The claim should say that one global
`config.require_baked_npa: "1"` applies to every rendered step, rather than that
each stage repeats the field.

Repository facts:

- [`sim2real.yaml`](../../npa/workflows/workbench/npa-workflows/sim2real.yaml)
  sets `require_baked_npa` globally and maps all CPU/GPU roles to five mandatory
  operator-provided images.
- The renderer rejects a step without a registry-qualified digest and rejects
  any non-40-hex `config.source_sha`. Its generated setup imports the baked
  stage module and compares image-local `NPA_IMAGE_SOURCE_SHA` byte-for-byte
  with workflow `NPA_SIM2REAL_SOURCE_SHA`; a mismatch exits before the stage.
  See
  [`skypilot_render.py`](../../npa/src/npa/orchestration/npa_workflow/skypilot_render.py).
- Image preflight requires the exact digest to satisfy
  `org.nebius.npa.skypilot-bootstrap-contract=skypilot-0.12.2-v1` for a trusted
  first-party image; missing first-party metadata is not replaced with a
  best-effort probe. See
  [`image_bootstrap_contract.py`](../../npa/src/npa/orchestration/skypilot/image_bootstrap_contract.py)
  and the preflight path in
  [`workflow/__init__.py`](../../npa/src/npa/cli/workbench/workflow/__init__.py).

Registry/publication facts, verified anonymously on 2026-09-02:

| Workflow role | Accepted public release | Source env | Bootstrap label |
|---|---|---|---|
| controller | **No public-plan entry; `npa-sim2real-control:latest` unresolved** | n/a | n/a |
| Transfer | `npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z` | absent | absent |
| EnvGen | `npa-envgen:cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | absent | absent |
| Isaac | `npa-isaac-lab:3.0.0b2.post1` | present, source `4a7967ce…` | present |
| viewer | `npa-rerun-viewer:0.31.4` | absent | absent |

The repository's accepted-release verifier resolved all 31 recorded releases
and matched their recorded digests anonymously. The controller is deliberately
outside `publicly_publishable_tools()`, as the current
[container catalog](container-image-catalog.md) already states. Therefore there
is no published five-image default set to reconcile, and the four available
public roles cannot satisfy one exact-source execution. The reported publishing
decision—build and publish all five roles from one release SHA, including the
controller, then document those exact digests—is consistent with the enforced
contract. PR #367 added the controller Dockerfile's bootstrap label; it did not
publish the coherent set.

## 2. Missing `source_sha` in the operator guide

**Disposition: partially valid.** Every `plan-spec` and `submit` example in the
canonical [Sim2Real guide](guides/sim2real-workflow.md) omits
`--var source_sha=...`. The CLI has no git-HEAD inference for this config value.

There is one material correction to the finding:

- `workflow plan-spec ... --waves` is a graph planner and succeeded without
  `source_sha` in this evaluation.
- `workflow submit ... --plan-only` renders SkyPilot tasks. With otherwise valid
  immutable image placeholders it exited 1 at `stage-01-trigger` with
  `requires an exact source SHA`; the same command with a 40-hex value exited 0
  and returned `PLANNED` for five rendered tasks.
- A real submit follows that rendering path. Each resulting task receives
  `NPA_SIM2REAL_SOURCE_SHA`, and the generated setup rechecks it against
  `NPA_IMAGE_SOURCE_SHA` at runtime.

Thus the guide's rendered-plan/submit path is broken as written, although the
plain graph-planning command is not. Documentation should define the release
SHA once and pass it to every rendered-plan, initial-submit, and resume example.

## 3. Kueue installation

**Disposition: valid as a repository/operator gap.** The code pins
`KUEUE_VERSION = "0.17.3"` and uses `kueue.x-k8s.io/v1beta2` in
[`job_scheduling.py`](../../npa/src/npa/workflows/sim2real/job_scheduling.py).
The workflow applies queue labels to every GPU resource. The guide generates
`ResourceFlavor`, `ClusterQueue`, `LocalQueue`, and `PriorityClass` objects, but
only says that missing CRDs mean Kueue “must be installed first.” A repository
search found no Kueue installer or install command.

Primary upstream and live checks agree with the reported command shape:

- The [Kueue v0.17.3 installation source](https://github.com/kubernetes-sigs/kueue/blob/v0.17.3/site/content/en/docs/installation/_index.md)
  documents the OCI Helm chart, namespace creation, and wait flow.
- The official OCI registry resolved chart tag `0.17.3`; tag `v0.17.3` returned
  `MANIFEST_UNKNOWN`.
- The currently configured, reachable NPA Kubernetes environment had zero Kueue
  CRDs and no Kueue controller object. This is sanitized supporting evidence,
  not proof about every new Nebius cluster or the historical Living Lab cluster.

No unrelated cluster was mutated merely to test installation. Repository
provisioning defaults to managed GPU drivers but contains no general Kueue
installation step, so a newly provisioned cluster cannot be assumed to satisfy
this workflow prerequisite. The exact reported Helm invocation is technically
consistent with the pin; an equivalent digest/pinned manifest would satisfy the
same decision.

## 4. Missing Predict2.5 tokenizer access probe

**Disposition: valid, including the related error-reporting claim.** At the
Cosmos Transfer source revision pinned by the image (`67d56b7d…`), NVIDIA's
[`packages/cosmos-oss/cosmos_oss/checkpoints.py`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/67d56b7d550a3911024a32dc23ae0bae5258e633/packages/cosmos-oss/cosmos_oss/checkpoints.py)
registers the Wan2.1 VAE as `tokenizer.pth` from
`nvidia/Cosmos-Predict2.5-2B` at revision
`f176dc95b4a70f53ce01c4b302851595e7322b00`. The same pinned source's
checkpoint downloader invokes an exact-revision Hugging Face file download.
The [official Hugging Face revision](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/tree/f176dc95b4a70f53ce01c4b302851595e7322b00)
reports gated access and contains `tokenizer.pth`; an anonymous one-byte request
returned 401.

Current NPA behavior does not model that dependency:

- [`WORKBENCH_ASSETS`](../../npa/src/npa/workbench/model_access.py) contains four
  exact Transfer2.5 checkpoint probes for each of `cosmos2`, `paidf`, and
  `sim2real`, but no `Cosmos-Predict2.5-2B` entry.
- A live `health access --capability sim2real --json` check exited 0 with all
  four Transfer checkpoint probes passing (and no Predict tokenizer probe).
  This directly demonstrates the premature-green path; it does not imply that
  the caller lacks Predict entitlement.
- The Sim2Real guide names only `nvidia/Cosmos-Transfer2.5-2B` in its terms and
  access section.

The proposed capability membership and exact tokenizer probe are therefore
well-founded. The revision and file above—not a moving branch or metadata
endpoint—are the dependency boundary evidenced by the pinned runtime.

The related `run_cosmos_transfer` finding is also valid. The runner redirects
combined vendor stdout/stderr into an unnamed temporary file, then catches
`CalledProcessError`/`OSError`, raises a fixed generic `RuntimeError` with
`from None`, and destroys the temporary file. The Sim2Real wrapper can only log
that generic message. Sensitive vendor logs should still be sanitized, but the
current behavior removes the actual failure detail needed to distinguish model
access, download, CUDA, and inference failures.

## 5. Graphics/Vulkan userspace

### Original finding

**Disposition: partially valid; the historical host assertion was not
independently verifiable.** NVIDIA's
[container-toolkit contract](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)
states that the `graphics` capability is required for OpenGL, EGL, and Vulkan.
Selecting an NVIDIA runtime and requesting `NVIDIA_DRIVER_CAPABILITIES=all`
asks the runtime to mount those driver components; it cannot mount libraries
that the node does not provide.

Repository and workload boundaries differ:

| Surface | Current repository contract | Disposition of asserted impact |
|---|---|---|
| Sim2Real | The Isaac image requests `NVIDIA_DRIVER_CAPABILITIES=all`, but `sim2real.yaml` does not itself provision a cluster or select its driver mode. | **Valid portability risk, now addressed by an explicit provisioning profile.** Operators must create or verify the target with `gpu_workload_profile: rtx-rendering`; workflow image settings alone cannot establish node-side graphics readiness. |
| LeIsaac | Its generated pod explicitly sets `runtimeClassName: nvidia` and `NVIDIA_DRIVER_CAPABILITIES=all`, while comments assume the runtime supplies matching graphics/Vulkan libraries. | **Valid portability risk.** The request is present, but no node-side graphics availability probe makes the assumption true. |
| Content Agents | Current docs require `--gpu-driver-mode operator`; the image declares `compute,utility,graphics,display`; rendered RTX stages fail early unless `libGLX_nvidia.so.0` is loadable. | **Stale/already addressed for current main.** The general node distinction is real, but this workflow already documents and enforces its chosen remedy. |
| NuRec/NRE | The workflow requires RT cores and R580+ on Blackwell and runs NVIDIA's NRE container. No checked repository contract identified Vulkan/GLX as a prerequisite. | **Not substantiated as the same defect.** RT-core routing alone does not prove a Vulkan dependency; validate the exact NRE container/runtime separately before changing its driver policy. |

The reported workaround—installing host-driver-version-matched GL/Vulkan
packages in an application image—can supply missing userspace, but it couples
the image to the host kernel-module branch. The repository already records that
tradeoff for a separate SONIC compute-only variant and prefers host-mounted,
driver-matched libraries for its operator-mode RTX path. That is evidence that
the workaround is technically plausible, not evidence that it is the correct
portable contract for all four surfaces.

The historical run prefix could not be retrieved from the credentials/project
scope available to this evaluation, and no matching live Kubernetes objects
remained. Consequently, the specific statement that those Living Lab nodes had
compute-only userspace is supported by PR #367's report but was not independently
verified here. Running a graphics probe on a different cluster would not prove
the historical node state, so no substitute GPU smoke was presented as that
evidence.

### Repository fix

The new public `rtx-rendering` GPU workload profile is available from
`npa provision-if-absent --gpu-workload-profile rtx-rendering`, the provisioning
SDK/config surface, fleet specs, and the agent provisioning bridge. It selects
one `gpu-rtx6000` / `1gpu-24vcpu-218gb` worker by default and requires
`gpu_driver_mode=operator`. Explicit conflicting platform, preset, zero-worker,
or managed-driver selections fail before mutation. Omitting the profile keeps
all existing defaults, and operator mode remains rejected for NVSwitch
topologies.

The profile also makes graphics readiness part of normal cluster validation.
After the existing stable-node, driver-component, allocatable-GPU, and per-node
CUDA vectorAdd gates pass, NPA schedules an immutable-digest probe on every
requested GPU worker with `runtimeClassName: nvidia`, one GPU, and
`NVIDIA_DRIVER_CAPABILITIES=all`. The probe dynamically loads
`libGLX_nvidia.so.0` and `libEGL_nvidia.so.0`, creates a Vulkan instance through
`vulkaninfo`, and requires enumeration of an NVIDIA physical device. Missing
libraries, a nonfunctional Vulkan loader, a non-NVIDIA device, or an
unschedulable probe is fatal; there is no profile-specific warn-only or skip
path.

### Live validation outcome

Health, configuration, credential, and gated-access preflight passed before
mutation. NPA then provisioned one control plane, one CPU worker, and one RTX
PRO 6000 Blackwell worker with GPU Operator mounted drivers. The unmodified
strict gates all passed: requested nodes remained Ready for the 120-second
stability window, the GPU worker reported capacity, operator components were
healthy, CUDA vectorAdd passed on the requested GPU worker, the default storage
class existed, SkyPilot resolved the cluster accelerator and completed its real
GPU smoke, and the smoke resources were cleaned up. Provisioning completed with
17 recorded actions and zero warnings.

The profile's graphics gate passed independently in 144.36 seconds. On the GPU
worker it loaded one GLX and one EGL NVIDIA userspace library, created a Vulkan
instance, and enumerated an NVIDIA physical device. The live path exposed and
fixed three integration defects before that result: an incorrect public probe
image path, unsafe graphics-library teardown in the probe process, and failure
to pass SkyPilot's resolved accelerator label into the final smoke.

A minimal asset-free Isaac Sim scene then exercised RayTracedLighting over
Vulkan on the same worker. It generated a cube, ground plane, light, and camera,
captured three distinct 512×512 RGB PNGs through Kit's awaited viewport-capture
contract, and wrote those frames plus a summary to object storage. Read-after-
write verification decoded all three images, measured a minimum channel span of
249 and minimum maximum-channel standard deviation of 62.116, and verified four
objects totaling 743,313 bytes. The aggregate artifact digest is
`f239f2cdb5f0c50d5e1f50266a3e041b4f0783669ad0f063114bb610d75abc15`.

The stock Franka capture was also attempted, but its vendor USD dependency was
not available from the runtime environment. That attempt was not counted as a
render success; the procedural scene removed the external-asset dependency
while preserving the RTX/Vulkan rendering requirement. The validated cluster
is retained, with transient validation Jobs, ConfigMaps, and credential Secret
removed, and is ready for future RTX workloads.

## Evidence and limitations

Evaluation commands included spec validation, graph planning, rendered submit
planning with and without `source_sha`, anonymous verification of all 31
accepted public image releases, OCI config inspection for the four published
workflow roles, official Kueue chart-tag resolution, a live capability-scoped
Hugging Face access check, official Hugging Face revision/file checks, sanitized
read-only Kubernetes discovery, repository tests, and publication guardrails.
The final non-E2E repository suite completed with 12,585 passed, 36 skipped,
12 deselected, and one expected XPASS; all 2330 guardrail tests and the 114-test
smoke target also passed. The first suite invocation exposed a broken
user-level `rerun` launcher; the two affected GR00T tests and then the complete
suite passed with the mandated repository virtualenv first on `PATH`.

No full 14-stage Sim2Real run was launched: findings 1–4 still prevent that
workflow from being a valid public-default proof. The focused live validation
for finding 5 instead provisions the exact supported cluster contract and runs
the smallest representative Isaac/RTX render that can prove the driver-facing
interface. PR #367's historical run remains secondary evidence because its
durable artifacts were not accessible in the available project scope.

## 2026-09-03 coherent public-image follow-up

The publication and exact-source blockers described above are now resolved by
one five-image release built from
`45e1128113abd4c03fe95f17bbcab5da333134b9`. The controller is now in the
canonical public inventory after built-layer and history scans confirmed its
`redistribution: public` contract. Cosmos and Isaac proprietary/model material
is still excluded from public layers and fetched only at runtime with the
operator's authorization.

| Workflow role | Additive supported tag | Published digest |
|---|---|---|
| controller | `npa-sim2real-control:0.1.2-sim2real-coherent-20260903` | `sha256:d1b0b38f3c4eb6d1ba0ce75c0be40b1d80ae32b0cf0b5b0e343502945c5600f5` |
| Transfer | `npa-cosmos2-transfer:2.5.1-sim2real-coherent-20260903` | `sha256:9330a6c10dccee9050f748b6e98d2ba79de9f889695c137ed97b28a7e9ddf658` |
| EnvGen | `npa-envgen:0.1.2-sim2real-coherent-20260903` | `sha256:46e67a8f2e30d83aa8b1c3f75853ebcf428c13693fce90f99bbc95906849b8b3` |
| Isaac | `npa-isaac-lab:3.0.0b2.post1-sim2real-coherent-20260903` | `sha256:645c4405169afa28668a970cca9d0c9f3000cda6f74c7ec3c16d5962b152f804` |
| viewer | `npa-rerun-viewer:0.31.4-sim2real-coherent-20260903` | `sha256:9517575558596e1b132a11fc623d3532014f3761d36d6b07a9b7f2342424b66b` |

For every digest, independent empty-config reads resolved the development ref
twice and the release tag once, then confirmed the exact common
`NPA_IMAGE_SOURCE_SHA`, `skypilot-0.12.2-v1` bootstrap label, and non-root OCI
user. Real digest-pinned capability checks passed: the controller expanded both
canonical decision branches; Transfer ran four guarded diffusion steps and
validated a 93-frame MP4; EnvGen produced 16 rows and advanced Genesis physics
on CUDA; Isaac produced and decoded three distinct RayTracedLighting/Vulkan
frames on RTX PRO 6000; and Rerun converted, reopened, served, and read a real
robotics trace.

This follow-up does not retroactively change the 2026-09-02 observations. The
configured workflow cluster still lacks Kueue, so no complete 14-stage run was
claimed and no unrelated scheduler installation was performed. The release
establishes a coherent public image boundary and a working documented
source-SHA/render path; model access, workflow data, queue admission, and policy
efficacy remain runtime/operator concerns.

## 2026-09-04 reviewer and live-validation follow-up

Reviewer follow-up rebuilt all five roles from the single reviewed source SHA
`c164fd3480f8a9ea8f9df9ccb9509502fd527996` and promoted the resulting bytes,
without rebuilding, to new additive release tags:

| Workflow role | Additive supported tag | Published digest |
|---|---|---|
| controller | `npa-sim2real-control:0.1.2-sim2real-coherent-20260904` | `sha256:87fe8530710eea43364a21ad76dbe4b4c2d60e4b49705824fcdb62dc7d185af7` |
| Transfer | `npa-cosmos2-transfer:2.5.1-sim2real-coherent-20260904` | `sha256:0caddf68ccac1b69bd9fe3fb089bcc325a111059065452d31fdf0576629895d3` |
| EnvGen | `npa-envgen:0.1.2-sim2real-coherent-20260904` | `sha256:08eb75118f5a04194d33a60308212db7706dd9c339d74afc5471a58608bf0422` |
| Isaac | `npa-isaac-lab:3.0.0b2.post1-sim2real-coherent-20260904` | `sha256:e321e8631c7e318b5012dad210d9cd1001b7dc833cbff0369e420c5c12657ab6` |
| viewer | `npa-rerun-viewer:0.31.4-sim2real-coherent-20260904` | `sha256:4c09c9cf3c14606db8e45c8ee564388888cd3261448f7c3f1304bfdfb1b9e5b2` |

The Transfer guardrail tokenizer cache now verifies an exact-revision,
content-addressed materialization before reuse. It rejects symlinks, missing or
empty files, revision/repository drift, and hash or size mismatches; a new cache
is staged completely and atomically installed with its marker written last.
Typed failures distinguish rate limiting, revision removal, access denial, and
network outages. Hermetic tests cover verified offline reuse, corruption,
partial downloads, atomicity, retry behavior, revision-not-found and 429 paths,
and prove that NLTK consumes ordinary materialized files rather than Hugging
Face blob links.

The exact controller digest was also started as a real SkyPilot Kubernetes pod.
The task observed effective UID 1000, generated SSH host keys only at pod
runtime after their absence from image layers was independently established,
waited for `sshd`, exercised `rsync`, and verified shell-argument forwarding,
the exact source SHA, and the `skypilot-0.12.2-v1` bootstrap label. The task
succeeded and its pod was removed.

An NPA-owned validation cluster was provisioned with eight single-GPU RTX PRO
6000 workers plus a CPU worker. All nodes passed readiness, CUDA, graphics, and
stability checks. Kueue 0.17.3 was installed through the pinned repository
contract, its ClusterQueue and LocalQueue were Active, and the Isaac cache PVC
was Bound. A preliminary exact-image run reached Stage 9 and exposed a real
Isaac point-cloud defect: an empty camera view was reduced before its empty
guard. The final source skips matched empty views and fails closed, without
writing an artifact, when point and color row counts differ; hermetic regression
tests cover both cases.

The fresh canonical run was then validated and planned with its unchanged
production defaults, the public seed was staged, all five exact image
preflights returned 200 with matching attestations, and all Hugging Face/NGC/S3
checks passed. Submission nevertheless failed closed before any workflow task
launched: the authenticated Token Factory catalog omitted the required
`nvidia/Cosmos3-Super-Reasoner`, and a minimal request returned HTTP 409 with an
upstream stopped-model response. No substitute model or authorization bypass
was used. Consequently this evidence does **not** claim 14-stage completion,
14 ComponentRecords, final RRD/MCAP/media, or policy efficacy; those validations
remain pending until the required hosted evaluator is served for the configured
key.

After the runtime fixes, the full local non-E2E suite completed with 12,930
passed, 36 skipped, 12 deselected, and one expected XPASS. The final release
metadata and documentation gates are recorded in PR #373's current checks.
