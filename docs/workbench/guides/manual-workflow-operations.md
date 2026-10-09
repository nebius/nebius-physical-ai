# Manual workflow operations

[Workbench documentation](../README.md) · [Workflow catalog](../../../workflows/README.md) · [workflow authoring reference](../npa-workflow-guide.md)

Operate one checked-in NPA declarative workflow from a saved local
configuration: choose a candidate, set documented values, verify its inputs and
images, submit one run, and inspect or resume its durable evidence. The
relevant limits are stated beside the batch, input, scheduling, and evaluation
steps below.

Use the installed command as the source of truth for a release:

```bash
npa workbench workflow --help
npa workbench workflow submit --help
```

The command names in this guide are intentional. The current CLI calls the
preparation command `prepare-run`, image checking `preflight-images`, and
resume `submit --resume-run <run-id>`. There is no standalone `prepare`,
`preflight`, or `resume` workflow command. Likewise, the current workflow CLI
does not have `--set`: use repeatable `--var KEY=VALUE` to overlay `config`.

## 1. Keep configuration deterministic and private

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

## 2. Choose a candidate workflow

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

For multiple episodes, use the shipped
[dataset batch driver](paidf-dataset-batches.md). Its private manifest selects
the workflow, inputs, episode/camera selectors, and configuration for independent
runs. Choose concurrency for the available cluster capacity:

```bash
MAX_CONCURRENT_RUNS='<operator-selected-concurrency>'
npa workbench workflow batch plan ./paidf-batch.yaml
npa workbench workflow batch submit ./paidf-batch.yaml \
  --state-dir ./private-batch-state --max-concurrent-runs "$MAX_CONCURRENT_RUNS"
npa workbench workflow batch status '<batch-id>' \
  --state-dir ./private-batch-state
```

Keep the manifest, checkout, and private state directory for recovery. Rerun
the same `batch submit` command with `--resume` to reconcile started runs and
recheck completed runs before admitting pending work. The foreground driver
uses existing capacity; it does not provision or resize node groups. Its local
status is a ledger of client outcomes, so inspect the original workflow run IDs
when remote execution is uncertain. The single-run lifecycle below explains
the planning, input, credential, and recovery checks that each run still needs.

## 3. Configure a checked-in workflow without copying YAML

Set the spec and a private set of values. A value supplied with `--var` overlays
the YAML's `config` for that command. Repeat the same overrides for planning,
image preflight, and submit. `validate-spec` validates the stored specification;
its installed interface accepts `--preset`, not `--var`.

```bash
SPEC='workflows/main/paidf-cosmos3.yaml'
PROJECT_ALIAS='<local-project-alias>'
RUN_ID='<new-run-id-from-prepare-run>'
BUCKET='<your-bucket>'
KUBE_CONTEXT='<your-kubernetes-context>'

npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket="$BUCKET" \
  --var variant_count=1 \
  --check-render --waves --json
```

Do not copy a rendered command string, generated YAML, or an artifact path into
the source workflow. Instead, select only documented `config` keys, keep them
as `--var KEY=VALUE`, and inspect the resolved plan. `--preset <name>` is also
available when a workflow documents a shipped preset; preset-owned dataset,
task, and trigger keys cannot be replaced by `--var`.

The PAIDF Cosmos 3 workflow's useful first overrides include `bucket`,
`variant_count`, `appearance_profiles_json`, `grade_threshold`, and
`attribute_threshold`. The existing [twelve-profile recipe](paidf-appearance-12.md)
is an opt-in configuration overlay; it supplies twelve profiles through
`appearance_profiles_json`. A profile JSON override must provide `lighting`,
`background`, `color_grade`, and `surface_finish` for every profile. More
variants can cycle through profiles; a variant count is not a count of distinct
profiles.

## 4. Single-run lifecycle

Run health checks before a provider operation. `preflight` checks selected
credentials; it does not prove a Kubernetes context, image pullability, or GPU
availability. A `SKIP` result is not a successful access check.

```bash
npa workbench health preflight \
  --project "$PROJECT_ALIAS" --checks nebius,s3,token_factory --json
npa workbench health access --capability cosmos3 --json
npa workbench workflow prepare-run "$SPEC" --project "$PROJECT_ALIAS" --json
```

Copy the returned run ID into the private `RUN_ID` value. `prepare-run` creates
a fresh ID when `--resume-run` is absent; pass `--resume-run "$RUN_ID"` only to
prepare scoped metadata for an existing ID.

The access command above checks the selected Cosmos 3 model closure. Choose the
matching capability when operating another workflow; its model or service access
requirements can differ.

Plan the actual submission configuration, then check every selected image path.
For Kubernetes image verification, `--infra` needs the exact `k8s/<context>`
target; the ambient context is not used as a substitute.

```bash
npa workbench workflow preflight-images "$SPEC" \
  --project "$PROJECT_ALIAS" \
  --infra "k8s/$KUBE_CONTEXT" \
  --var bucket="$BUCKET" --var variant_count=1 --json
```

Use `--plan-only` to inspect a rendered submission without launching it. The
PAIDF Cosmos 3 workflow declares runtime execution, so its actual submit uses
the runtime driver automatically; `--runtime` is shown below to make that
choice visible. Do not use `--assume-decision` for an execution: it is only for
planning a branch.

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

Follow the command's exact recovery message when it requires additional
evidence. A resume preserves the durable ledger; it is not permission to launch
a duplicate run or infer that a missing status response was a failed workload.

Use these read-only commands to save the current schema and inspect one explicit
evaluator report:

```bash
npa workbench workflow schema > ./workflow-schema.json
npa workbench cosmos-evaluator report --input-path '<explicit-report>'
```

The schema command describes the declaration shape; it does not verify that a
particular input, image, or scheduler request can run. The report command
interprets the supplied evaluator report; it does not create evaluation evidence.
See [report inspection](../cosmos-evaluator-report.md) for per-variant scores,
required/advisory/unverified diagnostics, and incomplete-evidence handling.

## 5. PAIDF input and quality boundaries

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

## 6. Move only selected objects between storage systems

Use a run-scoped prefix, separate credentials for each provider, and the
smallest permissions each identity needs. A practical split is: ingress can
read one source object, the workflow can read its input and write its own run
prefix, and egress can read only selected final objects. Do not grant a workflow
identity whole-bucket read/write/delete permissions, and do not share one
credential between the external source and the S3-compatible destination.

NPA workflow paths are `s3://` URIs. Do not claim `gs://` is a native PAIDF
input. Stage one named Cloud Storage object through an owner-only local
directory, verify it, then upload it to one named S3-compatible object. Use the
selected project's verified storage endpoint and private AWS credential profile
explicitly. `configure --show --env` exports host defaults; it does not establish
that its endpoint belongs to `PROJECT_ALIAS`. The
syntax below follows the official [Google Cloud Storage
reference](https://cloud.google.com/sdk/gcloud/reference/storage/cp) and [AWS
CLI `s3 cp` reference](https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html).

```bash
(
set -euo pipefail
umask 077
PRIVATE_INGRESS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-private-ingress.XXXXXX")"
trap 'rm -rf -- "$PRIVATE_INGRESS_DIR"' EXIT
chmod 700 "$PRIVATE_INGRESS_DIR"
S3_ENDPOINT='<verified-selected-project-storage-endpoint>'
S3_PROFILE='<private-scoped-ingress-aws-profile>'
GCS_OBJECT='gs://<source-bucket>/<input-key>.mp4'
LOCAL_INPUT="$PRIVATE_INGRESS_DIR/source.mp4"
S3_READBACK="$PRIVATE_INGRESS_DIR/source.s3-readback.mp4"
S3_BUCKET='<your-bucket>'
S3_KEY="ingress/$RUN_ID/source.mp4"
INPUT_URI="s3://$S3_BUCKET/$S3_KEY"

GCS_SOURCE_BYTES="$(gcloud storage objects describe "$GCS_OBJECT" --format='value(size)')"
gcloud storage cp "$GCS_OBJECT" "$LOCAL_INPUT"
LOCAL_SOURCE_BYTES="$(wc -c < "$LOCAL_INPUT" | awk '{print $1}')"
test "$GCS_SOURCE_BYTES" = "$LOCAL_SOURCE_BYTES"
SOURCE_SHA256="$(sha256sum "$LOCAL_INPUT" | awk '{print $1}')"

aws --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT" s3 cp "$LOCAL_INPUT" "$INPUT_URI"
aws --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT" s3 cp "$INPUT_URI" "$S3_READBACK"
test "$(wc -c < "$S3_READBACK" | awk '{print $1}')" = "$LOCAL_SOURCE_BYTES"
test "$(sha256sum "$S3_READBACK" | awk '{print $1}')" = "$SOURCE_SHA256"
)
```

The first byte and hash checks prove that the exact input object survived the
GCS-to-S3 staging path. Keep the source-provider and destination-provider
credentials separate: the GCS ingress identity reads only `GCS_OBJECT`, while
the S3 identity writes and reads only `INPUT_URI`. Give PAIDF only the staged
exact destination URI through `--input-uri`. The subshell stops at any failed
copy or verification; no later transfer proceeds after a mismatch.

For egress, choose one exact durable output object after reviewing the run.
Copy it with the egress identity to a separate Cloud Storage destination, read
that object back, and compare the complete bytes independently:

```bash
(
set -euo pipefail
umask 077
PRIVATE_EGRESS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-private-egress.XXXXXX")"
trap 'rm -rf -- "$PRIVATE_EGRESS_DIR"' EXIT
S3_ENDPOINT='<verified-selected-project-storage-endpoint>'
S3_PROFILE='<private-scoped-egress-aws-profile>'
OUTPUT_URI='<exact-s3-uri-from-npa-workflow-artifacts>'
LOCAL_OUTPUT="$PRIVATE_EGRESS_DIR/augmented_video.mp4"
GCS_OUTPUT='gs://<egress-bucket>/<run-scoped-prefix>/augmented_video.mp4'
GCS_READBACK="$PRIVATE_EGRESS_DIR/augmented_video.gcs-readback.mp4"

aws --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT" s3 cp "$OUTPUT_URI" "$LOCAL_OUTPUT"
OUTPUT_BYTES="$(wc -c < "$LOCAL_OUTPUT")"
OUTPUT_SHA256="$(sha256sum "$LOCAL_OUTPUT" | awk '{print $1}')"
gcloud storage cp "$LOCAL_OUTPUT" "$GCS_OUTPUT"
gcloud storage cp "$GCS_OUTPUT" "$GCS_READBACK"
test "$(wc -c < "$GCS_READBACK")" = "$OUTPUT_BYTES"
test "$(sha256sum "$GCS_READBACK" | awk '{print $1}')" = "$OUTPUT_SHA256"
cmp -s "$LOCAL_OUTPUT" "$GCS_READBACK"
)
```

This egress step verifies the complete selected object rather than relying on a
listed size or optional provider checksum metadata. Use an egress GCS identity
that can write and read only `GCS_OUTPUT`, distinct from both ingress identities.
Use a recursive copy only for an explicitly reviewed run-scoped prefix, never a
whole bucket.

## 7. Execution and authoring are different jobs

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
