# GR00T N1.5 custom-LeRobot offline improvement walkthrough

This guide rehearses the operator journey for the experimental
[`groot-n15-offline-rl.yaml`](../../../workflows/testing/groot-n15-offline-rl.yaml)
workflow. It is an offline learning and evidence workflow: it never connects to
a robot, deploys a policy, simulates physics, or turns a recorded observation
into a predicted next state. A real robot trial is a separate, operator-owned
promotion decision.

The companion [cookbook](../cookbooks/groot-n15-offline-policy-improvement.md)
defines the method and historic evidence boundary. This guide defines the
supported generic Workbench path for a checked-in reference, standard input,
or an operator's own code and data.

## One-time supported environment setup

From a clean checkout, create the repository-local environment used by every
command in this guide:

```bash
python3.12 -m venv npa/.venv
npa/.venv/bin/python -m pip install --upgrade pip
npa/.venv/bin/python -m pip install -e "npa[dev,adapter]"
```

Install and authenticate the tested Nebius CLI (`0.12.298`) as described in
the repository [installation guide](../../install.md). `npa configure` records
the normal project and storage setup. If it has already bootstrapped a tested
NPA-local CLI, the shell setup below selects that executable; otherwise the
tested CLI must be on `PATH`. Provider calls fail closed on an untested version.
The GPU worker independently records its CUDA, Python, SM120, and native
dependency closure before it trains.

## Turnkey public reference (the default)

The checked-in YAML now defaults to `input_mode: reference`. It fetches a small,
immutable selection from public ViFailback LeRobot v2.1 at
`619791b94f61667333fe179d471051e0bca0e06f`, using one right-wrist camera and
the source's documented per-episode success/failure plus avoidance/correction
annotations. The runtime, not a task-private archive, writes the provenance,
action contract, aligned hybrid-feedback document, and H.264 derivative. It
also fetches one separately pinned CC-BY recovery reference whose outcome is
explicitly **unknown**; recovery provenance never becomes a success label.

After normal `npa configure` and project selection, load the saved non-secret
project, bucket, and owned cluster context once in each new shell. This is a
supported convenience export; it does not print or export credentials. Where a
tested NPA-local Nebius CLI exists it also exports that non-secret executable
path, so an older ambient binary cannot silently change the provider client.
The reference submission is then one generic command, with no hand-built
sidecars or hidden task values:

```bash
eval "$(npa/.venv/bin/python -m npa configure --show --env)"
```

```bash
npa/.venv/bin/python -m npa workbench workflow submit \
  workflows/testing/groot-n15-offline-rl.yaml \
  --project "$NPA_PROJECT_ALIAS" --infra "k8s/$NPA_KUBE_CONTEXT" \
  --run-id "groot-n15-reference-<unique-id>" --var bucket="$NPA_BUCKET" \
  --runtime --output-format json
```

Before spending GPU time, run the commands below verbatim from the same checkout
and configured project. They are supported generic operations, not a private
controller bootstrap:

```bash
npa/.venv/bin/python -m npa workbench health preflight \
  --project "$NPA_PROJECT_ALIAS" --checks nebius,s3 --json
npa/.venv/bin/python -m npa workbench health access --capability groot --json
npa/.venv/bin/python -m npa workbench workflow validate-spec \
  workflows/testing/groot-n15-offline-rl.yaml --json
npa/.venv/bin/python -m npa workbench workflow plan-spec \
  workflows/testing/groot-n15-offline-rl.yaml --run-id groot-n15-reference-plan \
  --var bucket="$NPA_BUCKET" --check-render --json
npa/.venv/bin/python -m npa cluster list --project "$NPA_PROJECT_ALIAS" --format json
npa/.venv/bin/python -m npa workbench workflow gpus \
  --project "$NPA_PROJECT_ALIAS" --cluster "$NPA_KUBE_CONTEXT" \
  --spec workflows/testing/groot-n15-offline-rl.yaml --json
npa/.venv/bin/python -m npa workbench workflow preflight-images \
  workflows/testing/groot-n15-offline-rl.yaml --project "$NPA_PROJECT_ALIAS" \
  --infra "k8s/$NPA_KUBE_CONTEXT" --var bucket="$NPA_BUCKET" --json
```

Use a compatible task-owned RTX PRO 6000 context selected through normal
provisioning and `workflow gpus`; `configure --show --env` emits the saved
project-scoped context, and `preflight-images` proves the same target can pull
the rendered image before submit. The run itself verifies one visible sm_120
GPU, the pinned native dependency closure, decoded recorded video, checkpoint
reload, and durable artifact readback. It does not claim a hardware rollout or
policy promotion. The new turnkey receipt and fresh-checkout evidence will be
recorded here after the current reference run completes; earlier
private-extension rehearsal rows below are historical only.

For a completed run, use a fresh shell and only the documented generic commands:

```bash
npa/.venv/bin/python -m npa workbench workflow status "<run-id>" \
  --project "$NPA_PROJECT_ALIAS" --json
npa/.venv/bin/python -m npa workbench workflow logs "<run-id>" train-native-n15 \
  --project "$NPA_PROJECT_ALIAS" --json
npa/.venv/bin/python -m npa workbench workflow artifacts "<run-id>" \
  --project "$NPA_PROJECT_ALIAS" --json
```

`artifacts` intentionally has no SkyPilot `--isolated-config-dir` or
`--config-path` flags. If a live status route needs isolated state, use only the
saved submission receipt's documented isolation/config bindings; never recover
them from a daemon process environment.

## Input contracts

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

The custom code is installed with
`pip --no-deps --no-build-isolation --no-compile --target` after its archive
digest is checked. Both runtime dependencies and package build tooling must
already be present in the selected runtime image; an archive that needs an
undeclared build dependency fails instead of resolving it from a package index.
The adapter receives only a JSON request with controlled input/output paths and
must return those exact paths; it cannot redirect output to an arbitrary
location.

## Standard LeRobot data without custom code

Use `input_mode=standard` when the source is already a LeRobot v2.0/v2.1 tree
in an authorized S3 prefix. The only operator declaration is a JSON document;
it binds the input-tree digest, observed-label sidecar digest, source/provenance,
one-camera/state/action/instruction mapping, split membership, action semantics,
evidence selection, and model revisions. The checked-in
[`groot-n15-standard-input.example.json`](../../../workflows/testing/groot-n15-standard-input.example.json)
is the exact schema. Copy it to private storage, replace the placeholder values,
and do not commit the private copy.

The standard adapter does not execute code from the dataset or declaration. It
downloads the declared tree and label sidecar, checks both content digests,
writes `meta/modality.json` for a single logical `video.wrist`, `state.robot`,
and `action.robot` profile, validates native fields and task indices, freezes
training-only statistics, verifies raw/model/raw action roundtrip, transcodes a
separate H.264 derivative, creates the feedback manifest, and derives recorded
success/failure/recovery gallery examples only from supplied labels. It rejects
missing labels, group leakage, dimensions, media, units, or digests rather than
guessing them.

The label sidecar is actual operator feedback, not generated setup boilerplate.
It must be `npa.groot-feedback/v2` with one record per episode. It is normally
the only companion object besides the existing LeRobot data tree and the one
configuration file. A hybrid run needs observed successful and failed training
episodes and one aligned progress value per recorded frame. Recovery is its own
collection kind; an unknown task outcome remains unknown.

For example, after staging a private copy of the JSON declaration, submit the
same workflow with only the input mode and declaration location changed:

```bash
npa/.venv/bin/python -m npa workbench workflow submit \
  workflows/testing/groot-n15-offline-rl.yaml \
  --project "$NPA_PROJECT_ALIAS" --infra "k8s/$NPA_KUBE_CONTEXT" \
  --run-id "groot-n15-standard-<unique-id>" --var bucket="$NPA_BUCKET" \
  --var input_mode=standard \
  --var standard_input_config_uri="s3://<operator-inputs>/standard-input.json" \
  --runtime --output-format json
```

The config's model pins override the checked-in reference values inside each
native stage and are included in the prepare receipt. The run never treats a
free-form dataset field as Python or a physical action contract as optional.
For multiple cameras, noncontiguous logical fields, conversion from another
layout, or a custom reward extractor, use the explicit custom-code contract
below instead.

### Optional trusted custom-code extension

Custom code is not needed for the shipped reference or standard LeRobot v2
input. When it is genuinely required, provide the existing explicit contract:
an immutable dataset archive and source provenance/action documents, plus a
separately pinned Python archive, package, adapter callable, feedback callable,
and optional JSON configuration. The archive is an operator-authorized trusted
input; no dataset metadata selects a module or callable. The workflow validates
the archive digest, installs with no dependency resolution, limits adapter output
paths, and rejects a mismatch before training.

```toml
# pyproject.toml in the trusted external archive
[project]
name = "my-lerobot-extension"
version = "0.1.0"
requires-python = ">=3.10"
```

```python
# my_lerobot_extension/adapter.py
def adapt(request: dict) -> dict:
    return {"dataset_root": request["output_dataset_root"],
            "gallery_root": request["output_gallery_root"]}


def extract_feedback(request: dict) -> dict:
    return {"feedback_path": request["output_feedback_path"]}
```

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
  is an explicitly documented annotation transform. The workflow does not train
  or invoke ARM in this operator path.

The `hybrid` feedback mode requires both observed training successes and failures
and combines those labels with aligned progress while retaining regressions. It
is compared with a matched ordinary-BC control where the immutable data, groups,
split, base checkpoint, seed, schedule, and batch size are the same and only
training weights become one. It is not a proof that downweighting failures
learns explicit avoidance, improves unseen states, or improves robot success.

## Validate, plan, preflight, and submit

Start with a clean supported checkout and its own `npa/.venv`. Run the
non-secret `configure --show --env` command in the default-reference section
once in this shell. The generic commands below are the same for the reference,
standard, and trusted-custom modes; only normal project configuration and the
selected input declaration vary.

```bash
npa/.venv/bin/python -m npa workbench health preflight --project "$NPA_PROJECT_ALIAS" \
  --checks s3,nebius --json
npa/.venv/bin/python -m npa workbench health access --capability groot --json
npa/.venv/bin/python -m npa workbench workflow validate-spec \
  workflows/testing/groot-n15-offline-rl.yaml --json
npa/.venv/bin/python -m npa workbench workflow plan-spec \
  workflows/testing/groot-n15-offline-rl.yaml --run-id groot-n15-plan \
  --var bucket="$NPA_BUCKET" --check-render --json
npa/.venv/bin/python -m npa workbench workflow preflight-images \
  workflows/testing/groot-n15-offline-rl.yaml --project "$NPA_PROJECT_ALIAS" \
  --infra "k8s/$NPA_KUBE_CONTEXT" --var bucket="$NPA_BUCKET" --json
```

Provision or select a task-owned compatible one-GPU RTX PRO 6000 cluster using
the current GPU and capacity guidance. First use a compatible strict reservation
binding and non-preemptible worker. Use an authorized preemptible RTX fallback
only after fresh capacity evidence shows that compatible reserved placement is
unavailable; image, IAM, source, or application errors are not capacity evidence.
Persist staged inputs and every checkpoint before a preemptible submission.

Use the normal project provisioning route to select a task-owned compatible RTX
PRO 6000, then submit with the one command shown in the default reference
section. Do not add undocumented controller flags or pass credentials on argv.
The expected sequence is `prepare-custom-lerobot`, `train-native-n15`,
`evaluate-recorded-observations`, then `verify-durable-publication`. Preparation
validates the selected input contract, group split, train-only statistics,
raw/model/raw action roundtrip, feedback alignment, and recorded gallery.
Training derives `ceil(train_frames / batch_size) * complete_train_passes`
updates, saves/reloads both policies, and records GPU/CUDA evidence.

## Observe and interpret the run

Use only the generic lifecycle commands against the same immutable run identity:

```bash
npa/.venv/bin/python -m npa workbench workflow status "<run-id>" \
  --project "$NPA_PROJECT_ALIAS" --json
npa/.venv/bin/python -m npa workbench workflow logs "<run-id>" train-native-n15 \
  --project "$NPA_PROJECT_ALIAS" \
  --cached
npa/.venv/bin/python -m npa workbench workflow artifacts "<run-id>" --project "$NPA_PROJECT_ALIAS" \
  --json
```

The submission receipt binds the content-addressed durable run root. If a
supported recovery command needs task-local routing, use only receipt-provided
bindings from that same submit; do not inspect another process environment or
guess a controller identity.

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

For an interruption, first inspect durable artifacts and exact job status. Resume
only the same run ID after its checkpoint/output contract is reconciled; never
create a second job for the same prefix. The runtime refuses incompatible source,
output, or checkpoint identities.

```bash
npa/.venv/bin/python -m npa workbench workflow submit \
  workflows/testing/groot-n15-offline-rl.yaml --runtime --resume \
  --project "$NPA_PROJECT_ALIAS" --infra "k8s/$NPA_KUBE_CONTEXT" \
  --resume-run "<same-run-id>" --var bucket="$NPA_BUCKET" --output-format json
```

For a feedback revision, preserve the first run and create a new source or
feedback sidecar identity under a new run ID. Keep all held-out collection groups
isolated, submit the same spec again, and compare candidates through the two
receipts. Report a regression as a regression. New recovery recordings retain
their own collection kind and actual outcome; intervention does not imply
failure.

## Archived v4 rehearsal boundary

An earlier two-iteration rehearsal used independently staged public recordings
and a task-private external extension. It demonstrated the then-custom adapter
contract, matched ordinary-BC versus hybrid weighting, held-out recorded-action
inference, gallery generation, checkpoint reload, durable readback, and an
honestly reported offline regression. It did not use customer code/data,
customer hardware, a simulator, or a robot connection.

That archived result is not validation of the checked-in reference builder, the
standard no-code input declaration, the commands above, or a fresh checkout.
It remains private operational evidence with its original source, config, and
receipt identities. The current turnkey run and fresh-checkout receipt replace
the pending rows below only after the new workflow finishes.

## Fail-closed checks and promotion boundary

The rehearsal exercises these correction paths before GPU work:

| Input problem | What fails | Operator correction |
| --- | --- | --- |
| Missing, non-finite, or misaligned progress | Feedback preparation rejects the document before a manifest is written. | Supply one calibrated value per recorded frame and state its source/units/direction/alignment. |
| Success/failure group crosses train/eval | Feedback preparation rejects group leakage. | Move the complete collection/session/scene group to one split. |
| Camera/action key or units disagree with the profile | Standard or custom preparation rejects modality/action-contract mismatch. | Correct the explicit contract or adapter; never rename or rescale actions implicitly. |
| Source-tree or feedback digest differs | Preparation rejects it before a derivative or manifest is written. | Stage the intended immutable bytes and update the one matching declaration. |

The archived custom rehearsal also used a Cartesian-relative fixture numerically.
That fixture does not substitute for real Cartesian calibration evidence. A
future matched-trial comparison should record frozen baseline/candidate
checkpoints, task/reset protocol, observation/action schema, and real outcomes
under operator control. It is the promotion boundary: neither offline loss nor
this gallery authorizes robot deployment.

## Verification record

This table records the current shipped path. It intentionally does not treat the
archived custom rehearsal or historical uniform public proof as coverage for the
reference, standard, or fresh-checkout rows.

| Step | Evidence type | Observed outcome and limit |
| --- | --- | --- |
| Shipped public reference preparation | Source/unit/integration tests; fresh checkout/live run pending | The checked-in source pin, wrist mapping, outcomes, derived progress, and unknown-outcome recovery reference are code-covered. No fresh-run claim yet. |
| Standard no-code LeRobot input | Strict declaration and staged-input prepare test | Tree/feedback digests, mapping, action contract, split, and model pins are exercised hermetically. A real operator-S3 rehearsal is pending. |
| Optional trusted custom code | Archived v4 result and existing contract tests | Retained for arbitrary mappings/hooks; it is not required by the reference or standard path. |
| Mixed outcome/progress training and ordinary-BC control | Fresh RTX reference run pending | No current turnkey metric or robot-success claim until the new immutable run completes. |
| Held-out prediction, labelled gallery, MP4/HTML, and durable readback | Fresh RTX reference run pending | Recorded observations remain offline evidence only. |
| Second feedback revision and supported resume | Fresh reference iteration pending | The new run must preserve earlier artifacts and group isolation before this row becomes demonstrated. |
| Missing/misaligned feedback, split leakage, contract mismatch | Hermetic contract tests | Inputs fail closed before training; these are not physical-rollout evidence. |
