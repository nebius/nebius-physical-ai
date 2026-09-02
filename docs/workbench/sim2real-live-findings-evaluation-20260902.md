# Sim2Real live-findings evaluation (2026-09-02)

This document evaluates five findings reported after the Living Lab run against
`origin/main` at `e5ddb7d25ef2af5485bb43409579657497220a77`. It is an evaluation,
not an implementation plan: no product, image, infrastructure, or runbook fix is
included.

The referenced `sim2real-repo-fixes.md` was not present in the clone or found in
the repository's GitHub code search. The available originating context was the
operator-supplied summary and [PR #367](https://github.com/nebius/nebius-physical-ai/pull/367),
which remains open and unmerged as of this evaluation. PR #367's code changes
were inspected separately from current `main`; its “Not included” section
independently reports the publication, `source_sha`, Kueue, and graphics gaps.

## Dispositions

| # | Disposition | Maintainer consequence |
|---:|---|---|
| 1 | **Valid in substance; public-set wording corrected** | A coherent five-image release is required. Current public state is worse than “five inconsistent images”: only four of the five roles are in the accepted public plan, and three of those four lack required attestations. |
| 2 | **Partially valid** | Add `source_sha` to rendered-plan and submit examples. Plain `plan-spec --waves` does not require it; `submit --plan-only` and execution do. |
| 3 | **Valid repository/operator gap; fresh-cluster generalization bounded** | The runbook needs an explicit, pinned Kueue installation prerequisite or an equivalent provisioned contract. |
| 4 | **Valid** | The access catalog omits a runtime-fetched, gated Predict2.5 tokenizer. The related loss of vendor diagnostics is also valid. |
| 5 | **Partially valid and environment-specific** | Sim2Real and LeIsaac have an unverified graphics-userspace dependency. Content Agents already carries an explicit operator-mode/fail-closed contract. The same Vulkan claim is not established for NuRec. |

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
contract. PR #367 only adds the controller Dockerfile's bootstrap label; it does
not publish the coherent set and is not on current `main`.

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
| Sim2Real | The Isaac image requests `NVIDIA_DRIVER_CAPABILITIES=all`, but `sim2real.yaml` neither selects a GPU driver mode nor declares `runtimeClassName: nvidia`, nor does its preflight probe GL/Vulkan availability. | **Valid portability risk.** RTX rendering can fail on a compute-only node despite GPU allocation. Whether SkyPilot or the cluster injects a runtime class is external state, not a YAML guarantee. |
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
compute-only userspace is supported by PR #367's report but not independently
verified here. Running a graphics probe on a different cluster would not prove
the historical node state, so no substitute GPU smoke was presented as that
evidence.

## Evidence and limitations

Evaluation commands included spec validation, graph planning, rendered submit
planning with and without `source_sha`, anonymous verification of all 31
accepted public image releases, OCI config inspection for the four published
workflow roles, official Kueue chart-tag resolution, a live capability-scoped
Hugging Face access check, official Hugging Face revision/file checks, sanitized
read-only Kubernetes discovery, repository tests, and publication guardrails.
The final non-E2E repository suite completed with 12,558 passed, 36 skipped,
12 deselected, and one expected XPASS; all 2327 guardrail tests and the 114-test
smoke target also passed. The first suite invocation exposed a broken
user-level `rerun` launcher; the two affected GR00T tests and then the complete
suite passed with the mandated repository virtualenv first on `PATH`.

No full Sim2Real GPU run was launched: claims 1–4 fail before a workload could
validly start on current public defaults, and claim 5 concerns the unavailable
historical host. PR #367 supplies a reported real 14-stage run, but because its
durable artifacts were not accessible in the available project scope, this
evaluation treats that run as secondary evidence rather than independently
verified proof.
