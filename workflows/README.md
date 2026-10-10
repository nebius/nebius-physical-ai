# NPA workflow catalog

The [video variant sweep](guides/video-variant-sweep.md) combines timed VLM source descriptions and user hints into structured LLM-enhanced prompts, then pairs each shared prompt with every configured parameter combination. Cosmos3 full-source edge transfer (or the compatible Transfer 2.5 reference), paired visual review and Postgres/MLflow lineage complete the dataset path. Its operator kit previews the prompt-augmentation flow, prompt reuse, every parameter combination and worker assignment before submission. Direct prompts explicitly bypass augmentation. Completed candidates have verified recovery receipts; native runs export actual controls, a matrix of recorded outcomes, an offline HTML comparison viewer and an MP4 walkthrough.

[Docs](../docs/README.md) · [Authoring guide](../docs/workbench/npa-workflow-guide.md)

These `npa.workflow/v0.0.1` YAML files compose Workbench operations into a state
graph. NPA validates the graph, renders SkyPilot tasks, and manages run-scoped
artifacts. Start with a runbook that matches the result you want.

For automatic public sample setup and a common launch/view command, start with
[the four public workflow demos](../docs/workbench/guides/public-workflow-demos.md).

Kubernetes GPU profiles accept a string such as `RTXPRO6000:2` or a single-entry
mapping such as `{RTXPRO6000: 2}`. Both preserve the count when resolving the
cluster's GPU product name. Select one GPU request before submitting a Kubernetes
profile; see the [resource preflight guide](../docs/workbench/npa-workflow-guide.md#durable-run-supervision-and-recovery).

## Choose a starting point

| Goal | Spec and runbook |
| --- | --- |
| Run a complete public sample with one command | [Four workflow demos](../docs/workbench/guides/public-workflow-demos.md) — automatic inputs, standard GPU execution, and offline HTML results |
| Label videos in Encord and export an annotated MP4 | [Encord partner workflows](partners/encord/README.md) — real object tracks, exported-label verification, and media roundtrip |
| Augment a video or LeRobot episode | [PAIDF + Cosmos 3](guides/paidf-cosmos3.md) — public starter, local MP4, and episode/camera inputs; [twelve variants with preserved padding](../docs/workbench/guides/paidf-appearance-12.md#apply-the-recipe) |
| Generate an image or video | [Cosmos 3](../docs/workbench/cosmos3-generate.md) |
| Compare Cosmos3-Super serving topologies | [Benchmark results and workflows](../benchmark/cosmos3-super/README.md) |
| Improve a navigation policy from field failures | [Field failure workflow](testing/field-failure-policy-improvement.yaml) · [native and operator adapter runbook](../docs/workbench/cookbooks/field-failure-policy-improvement.md) — sealed data/runtime required; GPU acceptance pending |
| Reconstruct a captured scene | [NuRec](../docs/workbench/guides/neural-reconstruction.md) |
| Prepare a reconstructed scene for Isaac navigation | [Scan-to-Isaac handoff](../docs/workbench/guides/scan-to-isaac-navigation.md) — supplied collision mesh, portable USDZ, and native PhysX probes |
| Reconstruct metric RGB-D into a collision scene | [RGB-D scan to Isaac](../docs/workbench/guides/rgbd-scan-to-isaac.md) — measured TSDF surface, held-out depth qualification, colored USDZ, native PhysX; [explicit native-training handoff](../docs/workbench/guides/rgbd-scan-to-isaac.md#continue-into-native-navigation-training) requires the companion navigation implementation |
| Compose the 14-stage robot loop | [Sim2Real](../docs/workbench/guides/sim2real-workflow.md) |
| Train a GR00T policy | [GR00T N1.7](../docs/workbench/cookbooks/groot-1-7-training.md) |
| Run the Antioch-authored warehouse | [Warehouse batch](../docs/workbench/antioch-warehouse.md) — contact conveyor, six-carton stacking, measured evidence and readback |
| Post-train on Antioch warehouse data | [Warehouse vision post-training](../docs/workbench/antioch-posttrain.md) — real recorded frames, pretrained ResNet-18, held-out carton-cycle evaluation |
| Evaluate a BEHAVIOR 2026 policy | [Start here](../docs/workbench/challenge-onboarding.md) · [Workflow](testing/behavior-challenge-eval.yaml) · [measured scope and limits](../docs/workbench/behavior-campaign.md#scope-and-validation-status) — operator runtime required |
| Run a live π0.5 robot pickup in Antioch | [OpenPI live pickup](partners/antioch/openpi-live-pickup.md) — pretrained-policy inference, physical success checks, and native recording |
| Collect Antioch trajectories and train ACT | [Antioch ACT workflow](partners/antioch/antioch-offline-policy-train.yaml) — completed dataset → LeRobot training; [runbook](partners/antioch/README.md#dataset-based-act-training) |
| Fine-tune XR1 on Antioch robot demonstrations | [Antioch pipeline](partners/antioch/README.md) — physical demonstrations, Nebius S3, NPA credential storage, attached evaluation with the pinned Antioch SDK, and measured policy results |
| Package your own repository | [BYOF](../docs/workbench/cookbooks/byof-isaac-lab/README.md) |

A catalog entry describes a contract, not a guarantee that every configuration
has run successfully. The `testing/` directory includes executable examples,
integration tests, and explicitly labeled fixtures. Read each guide's required
input, image/GPU compatibility, and validation scope before allocating compute.

## Commands

From the repository root with [NPA installed](../docs/install.md), inspect a
checked-in example locally:

```bash
npa workbench workflow validate-spec workflows/testing/cosmos3-generate.yaml
npa workbench workflow plan-spec workflows/testing/cosmos3-generate.yaml --run-id demo
```

Expect a valid spec and one `generate` stage. Its bucket is a placeholder;
validation and planning do not verify model access, stage data, or reserve GPUs.

For execution, follow the selected runbook in order:

1. Configure the project and prepare its input, storage, credentials, and compute.
2. Validate and plan with the configuration overrides you will submit.
3. Use `prepare-run` to persist a run ID, then check image pullability.
4. Submit through `submit --runtime` using the same project, context, and values.
5. Inspect the exact run's `status`, `logs`, and `artifacts`; verify the outputs.

Successful submission reports `run_id`, including at the top level of JSON
output. Keep it with your private run notes. Runs with durable bucket-backed
state can also be rediscovered:

```bash
npa workbench workflow list \
  --s3-bucket '<your-bucket>' --workflow-s3-prefix '<parent-prefix>' --json
```

See [run lifecycle](../docs/run-lifecycle.md) for launch, monitoring, interrupted
submissions, and safe resume, and [teardown](../docs/teardown.md) for cleanup.
The [workflow guide](../docs/workbench/npa-workflow-guide.md) explains relative
and absolute ledger prefixes. Runtime resume preserves the recorded ledger
location and verifies its storage access before launch.

### Runtime choices

Use the runtime orchestrator for parallel groups and decisions evaluated during
the run. `metadata.executionMode: runtime` selects it automatically and rejects
`--no-runtime`. Runtime-required workflows reject `--assume-decision` during
execution; use assumed decisions only to inspect alternate paths offline.
`plan-spec --waves` previews scheduling waves. `submit --plan-only` previews the
static render and launches no workload.

`config.source_overlay: true` uses the automatically staged checkout's NPA code
with the selected images' installed dependencies. Other workflows default to
false; `NPA_SRC_OVERLAY=1` is an operator override. Keep the checkout available
during submission. The [authoring guide](../docs/workbench/npa-workflow-guide.md)
explains the full source and image contract.

### Isolated controller diagnostics

Pass the run's `--isolated-config-dir` to workflow status, logs, and cancellation.
Status uses that controller's Kubernetes configuration for pod, event, and node
diagnostics, and retains the same controller in suggested log commands. It does
not require the operator's ambient Kubernetes configuration to match the run.

### Failed-attempt diagnostics

Python stages that create useful local evidence before failing can publish an
explicit allowlist with `npa.workflows.attempt_diagnostics.publish_failed_attempt`.
The helper writes immutable, attempt-scoped originals, verifies their full-byte
readback, and writes the `failed` receipt last. It never writes a success or
qualification result.

```python
from npa.workflows.attempt_diagnostics import publish_failed_attempt

publish_failed_attempt(
    "s3://example-bucket/diagnostics",
    run_id="demo",
    stage="evaluate",
    attempt_id="attempt-1",
    exit_code=1,
    files={"worker.log": "/workspace/run/worker.log"},
)
```

Choose and redact every file explicitly. The helper does not discover a
directory or decide whether a log can contain credentials. Use a new
`attempt_id` for each execution; existing diagnostic objects are never
overwritten. The helper includes `run_id` in the object prefix. An interrupted
call may be retried with the same identifiers and exact file bytes; it accepts
existing identical objects and rejects different bytes. An immutable attempt
manifest reserves the full file set before any original is published, so competing
retries cannot add files to the same attempt. A completed receipt is checked
before any writes. Its original files must still pass readback verification.
Publish closed files from caller-controlled directories: final-component symlinks
and nonregular files are rejected, while parent directories are trusted.

## Layout

| Directory | Contents |
| --- | --- |
| [`main/`](main/README.md) | Principal workflows with guides beside each YAML link |
| [`testing/`](testing/README.md) | Reference workflows, component tests and fixtures, each paired with its guide |
| [`partners/encord/`](partners/encord/README.md) · [`partners/antioch/`](partners/antioch/README.md) | Partner integration workflows and their adjacent runbooks |
| [`guides/`](guides/README.md) | Setup and operation runbooks |

The CLI, agent, and live-submit matrix discover main, testing, and partner specs. Raw
SkyPilot tasks are separate tool-specific examples; see the
[reference-assets index](../npa/workflows/workbench/README.md).

The established Cosmos Transfer VDA, the separately named direct NVIDIA VDA,
and the direct DIG, IAA, and EVG translations are testing-tier PAIDF specs. The
direct translations record `reports/upstream.json` and all execute through
SkyPilot rather than OSMO or Airflow. The established Transfer and Cosmos 3
workflow YAMLs remain unchanged from `main`.

## Spec catalog

### Main workflows

See the [main workflow and guide table](main/README.md) beside the three principal YAML specs.

### Partner workflows

| Entry | Notes |
| --- | --- |
| [Encord labeling demo](partners/encord/encord-labeling-demo.yaml) | Upload → create labeling project → import bounding-box tracks → export and verify → annotated MP4; [runbook](partners/encord/README.md). |
| [Encord roundtrip demo](partners/encord/encord-roundtrip-smoke.yaml) | Exact media push → pull → verify, alongside standalone [push](partners/encord/encord-push.yaml) and [pull](partners/encord/encord-pull.yaml). |
| [OpenPI live pickup runbook](partners/antioch/openpi-live-pickup.md) | Operator-managed Antioch simulation ↔ pretrained π0.5 inference on Nebius, with finite physical checks and native recording. Uses the live deployment commands. |
| [`antioch-offline-policy-train.yaml`](partners/antioch/antioch-offline-policy-train.yaml) | Antioch trajectory collection → completed LeRobotDataset v3 → ACT training. Defaults exercise the cartpole data/checkpoint path with one optimizer step. |
| [`xr1-antioch-finetune.yaml`](partners/antioch/xr1-antioch-finetune.yaml) | [Antioch pipeline](partners/antioch/README.md): robot demonstrations → Nebius S3 → native XR1 fine-tuning on eight RTX PRO 6000 GPUs → Antioch held-out evaluation → results and recordings in S3. |

### Testing and reference workflows

See the [testing workflow and guide tables](testing/README.md) beside the reference YAML specs.

#### Generation and reconstruction

[Workflow and guide table](testing/README.md#generation-and-reconstruction).

#### Robot learning and simulation

[Workflow and guide table](testing/README.md#robot-learning-and-simulation).

#### Data, perception, and scenario analysis

[Workflow and guide table](testing/README.md#data-perception-and-scenario-analysis).

#### Bring your own framework

[Workflow and guide table](testing/README.md#bring-your-own-framework).

#### Hosted inference and VLM evaluation

[Workflow and guide table](testing/README.md#hosted-inference-and-vlm-evaluation).

#### Infrastructure and runtime examples

[Workflow and guide table](testing/README.md#infrastructure-and-runtime-examples).

## Live GPU / CPU submit E2E

Skip-by-default. Use a configured operator VM with Nebius credentials,
SkyPilot bootstrapped, and access to the workflow's models and services:

```bash
# CPU-tier workflows
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu \
  ./scripts/npa-workflow-submit-live-e2e.sh

# Full matrix (cpu + gpu + multi)
./scripts/npa-workflow-submit-live-e2e.sh

# Plan-only preflight for the matrix (no job launch)
NPA_E2E_NPA_WORKFLOW_SUBMIT_PLAN_ONLY=1 \
  ./scripts/npa-workflow-submit-live-e2e.sh
```

Supported images use public GHCR by default; `NPA_E2E_REGISTRY` is an optional
explicit override for custom images. Hosted inference stages need
`NEBIUS_TOKEN_FACTORY_KEY`; model and data access depend on the selected spec.
Use `NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=paidf-cosmos3.yaml` to select the Cosmos 3
workflow. Matrix source of truth:
[`submit_matrix.py`](../npa/src/npa/orchestration/npa_workflow/submit_matrix.py).
Set `NPA_E2E_NPA_WORKFLOW_RUNTIME=1` to run its full dynamic pipeline; the
one-shot test verifies that assumed promotion is refused. Runtime validation
downloads and fully decodes the source and generated videos, verifies their
timelines and control hashes, and checks every downstream component report.

## Further reading

- [Workflow runbooks](guides/README.md) and [robot guides](../docs/workbench/guides/README.md).
- [Tool catalog](../docs/workbench/npa-workflow-tool-catalog.md) and [authoring reference](../docs/workbench/npa-workflow-guide.md).
- Agent skills: [author a workflow](../skills/workflows/author-npa-workflow/SKILL.md) or [design a pipeline](../skills/workflows/generate-npa-workflow/SKILL.md).

MJLab workflows require an explicitly built MJLab image while its public release is quarantined; see the [MJLab guide](../docs/workbench/mjlab.md).
