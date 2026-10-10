# Manual workflow operations

[Workbench documentation](../README.md) · [Workflow catalog](../../../workflows/README.md) · [workflow authoring reference](../npa-workflow-guide.md)

Run an existing workflow from saved private configuration, inspect its outputs,
and resume the same run after an interruption. The example uses PAIDF Cosmos 3;
other workflows have their own input, image and model-access requirements.

Before a cloud run, complete [Workbench setup](../getting-started.md), select a
compatible GPU cluster, and read the [PAIDF setup guide](../../../workflows/guides/paidf-cosmos3.md).
Run the examples from the checkout root in the same Bash shell. Replace quoted
placeholders with your private values. For release-specific options, use
`npa workbench workflow <command> --help`.

Use `prepare-run` for a new ID, `preflight-images` for image checks, and
`submit --resume-run` to resume. Set parameters with `--var KEY=VALUE`.

| Task | Go to |
| --- | --- |
| Save configuration and credentials | [1. Configuration](#1-save-private-configuration) |
| Find a workflow | [2. Workflow selection](#2-choose-a-workflow) |
| Reuse YAML with parameter overrides | [3. Parameters](#3-set-workflow-parameters) |
| Prepare, plan, submit, inspect or resume | [4. Single run](#4-run-and-inspect-one-workflow) |
| Submit episodes and choose concurrency | [5. Batches and scale](#5-submit-episode-batches) |
| Select source data and assess quality | [6. Inputs and quality](#6-check-inputs-and-quality) |
| Copy selected GCS/S3 objects | [7. Storage transfers](#7-transfer-selected-objects) |
| Draft a new workflow | [8. Authoring](#8-author-a-new-workflow) |

## 1. Save private configuration

NPA saves operator configuration and credentials outside the repository,
normally under `~/.npa/`. A separate automation environment can set
`NPA_CONFIG_DIR` to an owner-only directory. Record the local project alias and
other non-secret settings once instead of retyping them into a workflow:

```bash
npa configure \
  --project-alias '<local-project-alias>' \
  --tenant-id '<your-tenant-id>' \
  --project-id '<your-project-id>' \
  --region '<your-region>' \
  --no-provision
npa configure --show --env
```

The first command writes the selected local configuration without creating or
adopting storage. It does not put those values in the YAML. Use
`npa configure --save-env-credentials` only when the relevant credentials are
already in the private process environment; it writes an owner-only credentials
file and never prints values.

Keep shell exports, cloud credentials, and object paths in a private
`.env.workflow-private` file only if `git check-ignore -v .env.workflow-private`
shows that it is ignored. This repository ignores `.env` and `.env.*`; do not
force-add an ignored file. Do not place a secret value after `--secret-env`:
that flag takes an environment-variable *name*, not its value.

For a reproducible manual run, retain privately: the source revision, selected
project alias, spec path, run ID, complete `--var` list, and exact input/output
object URIs. Those facts are enough to repeat the configuration without putting
private state into Git.

## 2. Choose a workflow

Discover shipped candidates before choosing one:

```bash
rg --files workflows/main workflows/testing -g '*.yaml'
sed -n '1,220p' workflows/README.md
```

`workflows/main/` contains the small set of principal workflows. It is a useful
starting point, not a blanket claim that every input, cluster, or output use is
qualified. `workflows/testing/` contains references, components, fixtures, and
experiments. A testing workflow may need a specific input contract, image,
credential, or validation step and may not be suitable as an operational
template without review. Read the adjacent guide and the YAML's `config`,
`resources`, `inputs`, and `outputs` before selecting either kind.

## 3. Set workflow parameters

Set the spec and your private values. `--var KEY=VALUE` overlays the YAML's
`config` for that command; use the same overrides for planning, image preflight
and submit. `validate-spec` checks the stored spec and accepts `--preset`, not
`--var`.

```bash
SPEC='workflows/main/paidf-cosmos3.yaml'
PROJECT_ALIAS='<local-project-alias>'
BUCKET='<your-bucket>'
KUBE_CONTEXT='<your-kubernetes-context>'

npa workbench workflow validate-spec "$SPEC" --json
```

Use documented `config` keys instead of copying rendered commands or artifact
paths into YAML. A shipped `--preset <name>` can select a documented recipe;
its dataset, task and trigger keys cannot be replaced by `--var`.

The PAIDF Cosmos 3 workflow's useful first overrides include `bucket`,
`variant_count`, `appearance_profiles_json`, `grade_threshold`, and
`attribute_threshold`. The existing [twelve-profile recipe](paidf-appearance-12.md)
is an opt-in configuration overlay; it supplies twelve profiles through
`appearance_profiles_json`. A profile JSON override must provide `lighting`,
`background`, `color_grade`, and `surface_finish` for every profile. More
variants can cycle through profiles; a variant count is not a count of distinct
profiles.

For editor completion, [export the installed workflow schema](../npa-workflow-guide.md#schema-and-editor-completion).
The schema describes declaration structure; tool-specific `config` values still
need validation and planning.

## 4. Run and inspect one workflow

Run health checks before a provider operation. `preflight` checks selected
credentials; it does not prove a Kubernetes context, image pullability, or GPU
availability. A `SKIP` result is not a successful access check.

```bash
npa workbench health preflight \
  --project "$PROJECT_ALIAS" --checks nebius,s3,token_factory --json
npa workbench health access --capability cosmos3 --json
npa workbench workflow prepare-run "$SPEC" --project "$PROJECT_ALIAS" --json
```

Set `RUN_ID` to the `run_id` returned by `prepare-run` before planning or submitting:

```bash
RUN_ID='<run-id-returned-by-prepare-run>'
```

`prepare-run` creates a fresh ID unless you pass `--resume-run "$RUN_ID"` to
prepare metadata for an existing run.

The access command above checks the selected Cosmos 3 model closure. Choose the
matching capability when operating another workflow; its model or service access
requirements can differ.

Plan the actual submission configuration, then check every selected image path.
For Kubernetes image verification, `--infra` needs the exact `k8s/<context>`
target; the ambient context is not used as a substitute.

```bash
npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket="$BUCKET" --var variant_count=1 \
  --check-render --waves --json
npa workbench workflow preflight-images "$SPEC" \
  --project "$PROJECT_ALIAS" \
  --infra "k8s/$KUBE_CONTEXT" \
  --var bucket="$BUCKET" --var variant_count=1 --json
```

Use `submit --plan-only` to inspect a submission without launching it.
PAIDF selects the runtime driver automatically; the example spells out
`--runtime`. `--assume-decision` is only for planning a branch.

This first submission uses the workflow's starter input. For your own data,
add exactly one [input selection](#6-check-inputs-and-quality).

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" --project "$PROJECT_ALIAS" \
  --infra "k8s/$KUBE_CONTEXT" --runtime --durable-s3 \
  --var bucket="$BUCKET" --var variant_count=1 \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN
```

`--secret-env` forwards only named, separately held credentials. The workflow
has distinct storage, model-access, and hosted-model prerequisites; passing a
credential does not prove access or acceptance. A submitted workflow can
fan out its declared stages, but fanout, variant count, and GPU/node capacity
are different facts. `plan-spec --waves` shows the workflow's declared waves;
it neither reserves nor proves scheduler capacity. A request such as `GPU:2`
also needs two compatible GPUs on one schedulable node for a single task.

Monitor the same run rather than submitting another one merely because a log is
not ready:

```bash
npa workbench workflow status "$RUN_ID" --project "$PROJECT_ALIAS" --watch
npa workbench workflow logs "$RUN_ID" --project "$PROJECT_ALIAS" \
  --stage generate-variants --no-follow
npa workbench workflow artifacts "$RUN_ID" --project "$PROJECT_ALIAS" --json
```

`artifacts` lists durable object URIs; it does not by itself qualify their
contents. Inspect the workflow's declared reports and output media. For PAIDF,
read `grade/cosmos_evaluator.json`, `grade/quality_disposition.json`, each
variant's `metadata.json`, and the generated MP4 before treating a result as
usable.

If the command reports that the original driver must be reconciled or resumed,
use the recorded run ID and same relevant flags. Do not combine `--run-id` and
`--resume-run`:

```bash
npa workbench workflow submit "$SPEC" \
  --resume-run "$RUN_ID" --project "$PROJECT_ALIAS" \
  --infra "k8s/$KUBE_CONTEXT" --runtime --durable-s3 \
  --var bucket="$BUCKET" --var variant_count=1 \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN
```

Follow the command's recovery message. Resume preserves the durable ledger;
a missing status response is not evidence that the workload failed.

Inspect the exact evaluator artifact listed by `artifacts`:

```bash
npa workbench insights report --input-path '<exact-local-or-s3-report>'
```

The [shared report guide](../insights-reports.md) explains the response and
supported formats; [Cosmos evidence](../cosmos-evaluator-report.md) covers
diagnostic roles and incomplete evidence. Inspection reads existing evidence; it does not
run evaluation.

## 5. Submit episode batches

First verify one selected episode through the single-run sequence above. Then
create a private manifest using the [dataset batch guide](paidf-dataset-batches.md).
It defines the workflow, project/context, inputs, explicit episode/camera
selectors, configuration and secret names.

```bash
MAX_CONCURRENT_RUNS='<operator-selected-concurrency>'
npa workbench workflow batch plan ./paidf-batch.yaml
npa workbench workflow batch submit ./paidf-batch.yaml \
  --state-dir ./private-batch-state --max-concurrent-runs "$MAX_CONCURRENT_RUNS"
npa workbench workflow batch status '<batch-id>' \
  --state-dir ./private-batch-state
```

Choose concurrency for [existing cluster capacity](paidf-dataset-batches.md#concurrency-and-existing-capacity).
The foreground driver counts active workflow clients, not GPUs; it does not
provision or resize node groups. For measured scale, record queue/startup and
generation time, GPU occupancy, accepted outputs and failed-run recovery.

Keep the same manifest, checkout and private state directory. Resume with the
same `batch submit` command plus `--resume`: started runs are reconciled and
completed runs rechecked before pending work is admitted. `batch status` is a
local ledger, not a fresh remote observation. If execution is uncertain, inspect
the original run IDs. See [batch recovery](paidf-dataset-batches.md#progress-and-recovery)
for interruptions, retained locks and cancellation.

## 6. Check inputs and quality

PAIDF accepts exactly one input mode per submission. For one local H.264 MP4,
use `--input-video /absolute/path/source.mp4`; for one staged object, use
`--input-uri 's3://<your-bucket>/<run-scoped-prefix>/source.mp4'`. For a
LeRobot dataset, give the dataset prefix plus the exact camera and episode, and
require the explicit selection:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" --project "$PROJECT_ALIAS" \
  --infra "k8s/$KUBE_CONTEXT" --runtime --durable-s3 \
  --lerobot-uri 's3://<your-bucket>/<dataset-prefix>/' \
  --lerobot-camera 'observation.images.front' --lerobot-episode 0 \
  --require-explicit-lerobot-selection \
  --var bucket="$BUCKET" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN
```

The LeRobot path reads the selected episode/camera video rather than treating a
dataset prefix as an instruction to copy every episode. It produces augmented
video and review artifacts for that selection; it does not create a complete
LeRobot dataset with action or state records.

The twelve appearance profiles are prompt/configuration choices, not proof of
physical material properties. Source-relative temporal diagnostics check
corresponding decoded source/output evidence. Appearance diagnostics can use
crop-aware source evidence and exclude verified padding. Those measurements
help locate a mismatch; they do not turn a photometric proxy into a material
measurement. Semantic material, contact, or task claims still need the
appropriate independent evidence. The current workflow is single-source-video
conditioned: it makes no supported multiview-consistency or multiview-accuracy
claim.

Prepared inputs are letterboxed rather than cropped. When verified source
padding exists, publication restores only those borders and retains the raw
model video plus a padding-preservation receipt. Inspect the receipt, source
edges, alignment/hash fields, and relevant crop/region evidence alongside the
quality report. Lowering `grade_threshold` or `attribute_threshold` only changes
the decision rule; it does not correct pixels, geometry, timing, or contacts.

## 7. Transfer selected objects

For GCS inputs or outputs, follow [scoped GCS/S3 transfers](scoped-storage-transfers.md).
The examples use separate identities, exact objects, full-byte readback checks,
and automatic private-staging cleanup. PAIDF accepts S3 inputs; it does not read
`gs://` objects directly.

## 8. Author a new workflow

The checked-in YAML is the declarative execution contract: `validate-spec`,
`plan-spec`, `preflight-images`, and `submit` operate it. An optional authoring
agent may help draft or explain a new YAML, but it does not replace the
operator's review, access checks, selected input, or submission. Review an
agent-produced YAML like any other candidate, then validate and plan it before
execution. Do not describe drafting, a successful plan, or a submitted job as
evidence of a completed workload.

For Sim2Real, use the current [Sim2Real operator runbook](sim2real-workflow.md)
instead of older controller instructions. Its concrete prerequisites are a
compatible declared input/robot contract, configured run-scoped S3 storage,
validated workflow plan, image preflight, and the exact selected execution
context. A historical report is not a substitute for those checks on a new run.
