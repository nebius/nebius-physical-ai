# GR00T N1.5 custom-LeRobot offline improvement walkthrough

This guide rehearses the operator journey for the experimental
[`groot-n15-offline-rl.yaml`](../../../workflows/testing/groot-n15-offline-rl.yaml)
workflow. It is an offline learning and evidence workflow: it never connects to
a robot, deploys a policy, simulates physics, or turns a recorded observation
into a predicted next state. A real robot trial is a separate, operator-owned
promotion decision.

The companion [cookbook](../cookbooks/groot-n15-offline-policy-improvement.md)
defines the method and historic evidence boundary. This guide defines the
supported generic Workbench path for an operator's own code and data.

## What an operator supplies

Stage private inputs in an authorized object store before submission. Do not put
credentials, endpoints, private bucket names, recordings, source archives, or
model weights in the checked-in YAML, an override committed to Git, or a PR.
The workflow accepts only worker-readable `s3://` locations.

| Input | Required contract | Why it is explicit |
| --- | --- | --- |
| Dataset archive | Immutable LeRobot source bytes and a SHA-256 | A source location alone is mutable. |
| Input provenance JSON | `npa.groot-input-provenance/v1`: source kind, stable source identifier, revision, archive SHA-256, license, privacy class, media redistribution permission, label provenance | The workflow does not assume that an input is public or Apache-licensed. |
| Action contract JSON | `npa.groot-action-contract/v1`: joint or Cartesian-relative kind, action keys, frame, translation and rotation units/representation, gripper fields, transform identity | Normalization cannot silently reinterpret an action coordinate. |
| External code archive | Content-addressed Python project with `pyproject.toml`, package name, adapter callable, feedback callable | An archive is trusted operator input; dataset metadata cannot select executable code. |
| Adapter configuration JSON | The extension's source mappings, selected camera/state/action fields, split and annotation protocol | The adapter must produce a native LeRobot v2.0/v2.1 layout plus `meta/modality.json`. |
| Runtime model/source pins | Exact GR00T repository/model revisions | Source and weights are fetched only in the worker. |

The custom code is installed with `pip --no-deps --no-compile --target` after its
archive digest is checked. Runtime dependencies remain owned by the pinned
runtime image and native bootstrap. The adapter receives only a JSON request
with controlled input/output paths and must return those exact paths; it cannot
redirect output to an arbitrary location.

### Reference rehearsal profile

The independent reference rehearsal uses a separately maintained, task-private
LeRobot extension archive at an immutable revision. It adapts a public,
recorded real-robot source into a v2.1 dataset with exactly one wrist camera:
`video.wrist`. It maps two state and two action fields, retains source timestamps
and source episode identifiers, and emits upstream per-episode observed
success/failure annotations. Failed rollouts carry an annotation-derived,
per-frame corrective phase; its non-monotonic decrease is retained. A separate
public intervention recording is a gallery-only `operator_recovery` item with
an **unknown** task outcome and is never mixed into the training or evaluation
split.

This profile is deliberately non-default. Its native N1.5 data configuration
comes from the external archive and names only `video.wrist`; it does not add a
front camera. Its recorded action contract is joint-coordinate data. A separate
Cartesian-relative, millimetre/tool-frame contract fixture verifies raw →
min/max model coordinates → raw numerical roundtrip and terminal action masks;
it is a mechanical invariant, not a claim that the reference data have Cartesian
physical semantics.

## Feedback and split contract

The feedback callable writes `npa.groot-feedback/v2`, with one entry per native
episode. Each entry has a `train` or `eval` split, a collection/session/scene
`group`, optional observed `success`, collection kind, tags, source episode ID,
and optional one-value-per-frame progress. A group must be entirely in one
split. An evaluation record always has unit training weight, so no held-out
label fits normalization or weights.

Use these meanings literally:

- `success: true` or `false` is an observed task outcome from the declared label
  protocol.
- `operator_recovery` is collection provenance, not a failure synonym. A
  successful recovery remains useful data; unknown stays unknown.
- `progress_source: measured` is a calibrated recorded signal. `annotation_derived`
  is an explicitly documented annotation transform. `arm` additionally requires
  a reward-checkpoint digest; the workflow does not train or claim ARM.

The `hybrid` feedback mode requires both observed training successes and failures
and combines those labels with aligned progress while retaining regressions. It
is compared with a matched ordinary-BC control where the immutable data, groups,
split, base checkpoint, seed, schedule, and batch size are the same and only
training weights become one. It is not a proof that downweighting failures
learns explicit avoidance, improves unseen states, or improves robot success.

## Write the private override

Keep this file outside Git (for example, an operator's private run directory).
The following is the complete schema for the reference profile; replace every
`<...>` value with one worker-readable, immutable input. `DATASET_ID` and
`DATASET_REVISION` must exactly equal `source_identifier` and `source_revision`
in the provenance JSON. The actual independent rehearsal records those values
and their hashes in a private receipt rather than publishing its storage paths.

```bash
export SPEC=workflows/testing/groot-n15-offline-rl.yaml
export RUN_ID=<new-immutable-run-id>
export PROJECT_ALIAS=<configured-task-project>
export BUCKET=<operator-owned-bucket>
export WORKFLOW_ROOT="s3://${BUCKET}/workflow-ledger/${RUN_ID}"
export OUTPUT_PREFIX="groot-n15-offline-rl/${RUN_ID}"
export DATASET_ID=<stable-source-identifier>
export DATASET_REVISION=<immutable-source-revision>
export DATASET_ARCHIVE_URI="s3://${BUCKET}/inputs/<dataset-sha256>.tar.gz"
export PROVENANCE_URI="s3://${BUCKET}/inputs/<provenance-sha256>.json"
export ACTION_CONTRACT_URI="s3://${BUCKET}/inputs/<action-contract-sha256>.json"
export EXTERNAL_CODE_URI="s3://${BUCKET}/inputs/<extension-sha256>.tar.gz"
export EXTERNAL_CODE_SHA256=<64-lowercase-hex>
export ADAPTER_CONFIG_URI="s3://${BUCKET}/inputs/<adapter-config-sha256>.json"
export DATA_CONFIG=<external_package.config:SingleCameraProfile>
export EVIDENCE_EPISODE=<held-out-episode-with-recorded-wrist-video>
```

The reference profile then supplies these exact configuration overrides:

```bash
CUSTOM_VARS=(
  "bucket=${BUCKET}" "prefix=${OUTPUT_PREFIX}"
  "dataset_repository=${DATASET_ID}" "dataset_revision=${DATASET_REVISION}"
  "dataset_archive_uri=${DATASET_ARCHIVE_URI}"
  "input_provenance_uri=${PROVENANCE_URI}"
  "action_contract_uri=${ACTION_CONTRACT_URI}"
  "external_code_uri=${EXTERNAL_CODE_URI}"
  "external_code_sha256=${EXTERNAL_CODE_SHA256}"
  "external_code_package=<external_package>"
  "adapter_callable=<external_package.adapter:adapt>"
  "feedback_callable=<external_package.adapter:extract_feedback>"
  "adapter_config_uri=${ADAPTER_CONFIG_URI}"
  "feedback_mode=hybrid" "grouping=collection" "split_strategy=adapter"
  "train_episode_count=<actual-train-groups>" "eval_episode_count=<actual-eval-groups>"
  "data_config=${DATA_CONFIG}" "embodiment_tag=<native-embodiment>"
  "action_horizon=16" "video_backend=decord" "derivative_codec=h264"
  "complete_train_passes=1" "batch_size=4" "learning_rate=0.0001"
  "training_seed=42" "inference_seed=1701"
  "evidence_episode_index=${EVIDENCE_EPISODE}" "evidence_video_key=video.wrist"
  "action_target_keys=<action.key.one,action.key.two>"
)
VAR_ARGS=()
for value in "${CUSTOM_VARS[@]}"; do
  VAR_ARGS+=(--var "$value")
done
```

These are shell values, not a shell job. Expand them into the generic CLI; do
not place S3 credentials in `CUSTOM_VARS`. The native adapter supports exactly
one visible GPU. It fails rather than silently using DataParallel or DDP.

## Validate, plan, preflight, and submit

Start with a clean supported checkout and its editable `npa/.venv`. The generic
commands below are the same for private data, an authenticated public archive,
or the independent rehearsal.

```bash
npa/.venv/bin/npa workbench health preflight --project "${PROJECT_ALIAS}" \
  --checks s3,nebius --json
npa/.venv/bin/npa workbench health access --capability groot --json
npa/.venv/bin/npa workbench workflow validate-spec "${SPEC}" --json
npa/.venv/bin/npa workbench workflow plan-spec "${SPEC}" --run-id "${RUN_ID}" \
  "${VAR_ARGS[@]}" --check-render --json
```

Provision or select a task-owned compatible one-GPU RTX PRO 6000 cluster using
the current GPU and capacity guidance. First use a compatible strict reservation
binding and non-preemptible worker. Use an authorized preemptible RTX fallback
only after fresh capacity evidence shows that compatible reserved placement is
unavailable; image, IAM, source, or application errors are not capacity evidence.
Persist staged inputs and every checkpoint before a preemptible submission.

With the exact selected Kubernetes context, prove the image can pull before a
training job is created, then submit the workflow runtime. The named secret
variables are resolved from the project credential store or environment and are
not serialized into the YAML, rendered plan, archive, or receipts.

```bash
npa/.venv/bin/npa workbench workflow preflight-images "${SPEC}" \
  --project "${PROJECT_ALIAS}" --infra "k8s/<exact-context>" \
  "${VAR_ARGS[@]}" --json
npa/.venv/bin/npa workbench workflow submit "${SPEC}" --runtime \
  --project "${PROJECT_ALIAS}" --run-id "${RUN_ID}" \
  --workflow-s3-uri "${WORKFLOW_ROOT}" --infra "k8s/<exact-context>" \
  "${VAR_ARGS[@]}" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --output-format json
```

The expected sequence is `prepare-custom-lerobot`, `train-native-n15`,
`evaluate-recorded-observations`, then `verify-durable-publication`. Preparation
verifies the external archive, source/provenance binding, native v2 layout,
single-camera modality, group split, train-only statistics, raw/model/raw action
roundtrip, feedback alignment, and recorded recovery reference. Training derives
`ceil(train_frames / batch_size) * complete_train_passes` updates rather than
using a time cap, saves and reloads both policies, and records GPU/CUDA evidence.

## Observe and interpret the run

Use only the generic lifecycle commands against the same immutable run identity:

```bash
npa/.venv/bin/npa workbench workflow status "${RUN_ID}" --project "${PROJECT_ALIAS}" \
  --workflow-s3-uri "${WORKFLOW_ROOT}" --json
npa/.venv/bin/npa workbench workflow logs "${RUN_ID}" train-native-n15 \
  --project "${PROJECT_ALIAS}" --workflow-s3-uri "${WORKFLOW_ROOT}"
npa/.venv/bin/npa workbench workflow artifacts "${RUN_ID}" --project "${PROJECT_ALIAS}" \
  --workflow-s3-uri "${WORKFLOW_ROOT}" --json
```

The durable provenance, training receipt, evaluation, actions, media validation,
gallery, and readback receipt bind the source/configuration/checkpoint hashes.
Inspect class counts, group split, progress/weight provenance, optimizer count,
foundation/control/candidate errors, reload checks, and readback inventory there.
The standalone HTML embeds the factual MP4 and labelled gallery so it remains
usable offline after an authorized download. Its labels identify recorded source
episode/frame/split/group, outcome (including unknown), collection kind, feedback
provenance, progress summary, and whether predictions are available. Recovery
references can be displayed with no candidate prediction.

An MP4 decode, representative decoded frames, source identity, time/frame range,
action transform, units/frame contract, and checkpoint hashes are verified by
the workflow before publication. Curves are offline predictions versus recorded
action targets in the declared coordinates; lower error is not robot success.

## Recover safely and iterate feedback

For an interruption, first inspect the durable ledger and exact job status. Resume
only the same `RUN_ID` and `WORKFLOW_ROOT` after its checkpoint/output contract
is reconciled; never create a second job for the same prefix. The runtime refuses
incompatible source, output, or checkpoint identities.

To rehearse the next feedback iteration, preserve the first archive and output
prefix. Create a new immutable dataset and/or feedback archive revision, update
its provenance digest and source revision, choose a new `RUN_ID` and
`OUTPUT_PREFIX`, and repeat the same commands. Keep the earlier evaluation
collection groups held out; compare the two candidates through their recorded
lineage and held-out diagnostics, reporting a regression as a regression. New
operator recovery recordings may be added with their own collection kind and
outcome; do not label them as failures merely because intervention occurred.

## Fail-closed checks and promotion boundary

The rehearsal exercises these correction paths before GPU work:

| Input problem | What fails | Operator correction |
| --- | --- | --- |
| Missing, non-finite, or misaligned progress | Feedback preparation rejects the document before a manifest is written. | Supply one calibrated value per recorded frame and state its source/units/direction/alignment. |
| Success/failure group crosses train/eval | Feedback preparation rejects group leakage. | Move the complete collection/session/scene group to one split. |
| Camera/action key or units disagree with the profile | Custom preparation rejects modality/action-contract mismatch. | Correct the explicit contract or adapter; never rename or rescale actions implicitly. |
| Archive digest or provenance identifier/revision differs | Preparation rejects it before adapter code runs. | Stage the intended immutable bytes and update the matching provenance document. |

The independent reference rehearsal also tests a Cartesian relative-action
fixture numerically. That fixture does not substitute for real Cartesian
calibration evidence. A future matched-trial comparison should record frozen
baseline/candidate checkpoints, task/reset protocol, observation/action schema,
and real outcomes under operator control. It is the promotion boundary: neither
offline loss nor this gallery authorizes robot deployment.

## Verification record

This table is filled with content-bound receipts when the independent reference
rehearsal completes. It intentionally does not treat the historical uniform
public proof as coverage for the labelled custom path.

| Step | Evidence type | Observed outcome and limit |
| --- | --- | --- |
| External code/archive staging | Runtime receipt | Pending live reference run; local contract test alone is insufficient. |
| Single-camera custom preparation | Runtime receipt | Pending live reference run; no fabricated front camera. |
| Mixed outcome/progress training and ordinary-BC control | Training receipt | Pending live reference run; no convergence or robot-success claim. |
| Held-out prediction and labelled positive/negative/recovery gallery | Evaluation and decode receipt | Pending live reference run; images remain recorded observations. |
| Second immutable feedback revision | Separate run and lineage receipt | Pending live reference run; first outputs remain preserved. |
| Cartesian action numerical contract | Hermetic fixture | Verified mechanically only; not real-robot Cartesian calibration evidence. |
