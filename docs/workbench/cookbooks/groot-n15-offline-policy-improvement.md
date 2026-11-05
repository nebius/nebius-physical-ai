# Experimental GR00T N1.5 offline policy improvement

Use a dedicated Nebius CUDA GPU environment with a native GR00T N1.5 installation
to fine-tune from recorded real-robot rollouts. The loop is: freeze a dataset and
feedback revision, prepare weights, train a candidate, evaluate matched robot
trials, and collect the next revision. Robot operation and deployment remain
operator controlled. No simulator or robot connection is required for training.

This is a runtime-local research recipe, separate from Workbench's promoted
[N1.7 image and workflow](groot-1-7-training.md). It does not install N1.5 into
that image or change `npa workbench groot finetune`. The Python entrypoint is
`npa.workbench.groot.offline`, with `prepare`, `train`, and `compare` commands.
`prepare_feedback`, `train_offline`, and `compare_rollouts` in the corresponding
Workbench modules provide the same operations to Python callers.

## Declarative public-data workflow

For a reproducible, public-data execution use
[`workflows/testing/groot-n15-offline-rl.yaml`](../../../workflows/testing/groot-n15-offline-rl.yaml).
It is experimental and deliberately remains under `workflows/testing/`; it does
not promote N1.5 into the N1.7 image or command surface. The four standard
runtime stages are public-input preparation, native training, whole held-out
native inference/video evidence, and independent durable readback.

The checked-in defaults pin `youliangtan/so101-table-cleanup` at a public
Apache-2.0 revision and N1.5 source/model revisions. The model weights are
fetched only in the worker and are neither committed nor included in the HTML.
The dataset's AV1 observations are retained as the raw source and converted to
an H.264 derivative because the native N1.5 reader is configured for `decord`.
The receipt binds sample source/derivative hashes and frame-count, size, and
rate parity; it explicitly calls the derivative lossy.

Validate and preview with an operator-owned bucket placeholder before allocating
compute. `PROJECT_ALIAS` and `BUCKET` are local configuration values, not values
to commit.

```bash
SPEC=workflows/testing/groot-n15-offline-rl.yaml
RUN_ID=groot-n15-offline-public-001

npa/.venv/bin/npa workbench workflow validate-spec "$SPEC" --json
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" --var bucket="$BUCKET" --check-render --json
npa/.venv/bin/npa workbench workflow stage-src \
  --project "$PROJECT_ALIAS" --bucket "$BUCKET" --run-id "$RUN_ID"
npa/.venv/bin/npa workbench workflow submit "$SPEC" --runtime \
  --project "$PROJECT_ALIAS" --run-id "$RUN_ID" \
  --var bucket="$BUCKET" --var prefix="groot-n15-offline-rl/$RUN_ID" \
  --output-format json
```

Use the generic `status`, `logs`, and `artifacts` operations with the same run
and project identity. A recoverable interruption must resume that run only after
its durable ledger and checkpoint pointer are reconciled; do not submit a second
run with the same output prefix.

The typed YAML customization surface includes public data/source/model revisions,
feedback mode and worker-readable feedback URI, episode split counts, action
horizon, native modality and embodiment, `decord` backend, codec provenance,
batch/learning-rate/seed/pass schedule, checkpoint cadence and root, output
prefix, and held-out evidence clip selection. `outcome` and `progress` modes
require supplied recorded labels/progress; uniform mode intentionally fabricates
neither. The schedule computes `ceil(train_frames / batch_size)` per requested
complete pass after the pinned data revision is materialized, rather than using a
time-based step limit. The N1.5 adapter requires exactly one visible GPU; it
does not use DataParallel or DDP.

The resulting MP4 contains held-out **recorded** front/wrist observations, plus
aligned recorded action targets and baseline/candidate native `get_action`
predictions. It labels raw action-coordinate units when the source metadata does
not declare physical units, uses the same per-frame diffusion seed for both
models, and says prominently that it is offline prediction rather than a policy
rollout. A simulator is not required for that evidence and is not constructed to
manufacture success footage. Hardware trials remain necessary for robot success,
next-state, deployment, or policy-improvement claims.

## Learning method and ARM boundary

The initial `outcome` mode implements return-weighted behavior cloning: an
episode's observed success is its undiscounted terminal return, and every chunk
in that episode receives the same score. Both successful and failed episodes
remain available. At least one training success is required. Binary labels are
sufficient; they cannot identify which actions inside a failed episode were
useful recovery actions.

The `progress` mode uses explicitly supplied, normalized per-frame progress.
For a contiguous action horizon H, the score at frame t is
`progress[min(t + H, length - 1)] - progress[t]`. Progress may decrease.
Negative gains receive less weight than positive gains; recovery segments in
failed episodes can still receive substantial weight.

Both modes exponentiate scores divided by `temperature` (default 1), using a
stable shift by the maximum training score. Training weights are normalized to
mean one across training frames. Evaluation labels never fit these statistics.
Lower temperatures concentrate weight; extremely small values can zero out
lower-score samples. The loss weights valid action elements and retains the
effect of weights with batch size one. Recorded action values remain intact;
actions beyond the recorded episode are excluded from the loss.

The `uniform` mode is ordinary, unweighted behavior cloning for demonstrations
that have no recorded outcome or dense progress. Its feedback entries omit both
`success` and `progress`; every valid training action has weight one. It exists
so a demonstration-only corpus is not relabelled as successful. It proves data
and policy-training integration, not reward discrimination, policy improvement,
or robot success.

[ARM's authors](https://aiming1998.github.io/ARM/) describe relative-advantage
reward modeling and GR00T N1.5 policy reweighting. Their
[implementation guide](https://github.com/FluxVLA/FluxVLA/blob/main/docs/arm.md)
uses additional reward-model supervision and a different weighting rule. This
recipe accepts ARM-produced progress with a reward-checkpoint digest; it does
not train ARM, reproduce its weighting, or implement a learned critic, IQL, or
CQL. Episode outcomes must never be interpolated into invented dense progress.

## Inputs and prerequisites

- A dedicated environment with exactly one visible BF16-capable CUDA GPU. Runtime
  dependencies must follow the upstream
  [N1.5 instructions](https://github.com/NVIDIA/Isaac-GR00T/tree/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96).
  On Linux, that source's `base` dependency set places `decord==0.6.0` in an
  optional extra even though its processor import needs it. Environments that
  install a separately validated CUDA Torch build therefore still need to
  install that exact `decord` requirement; do not omit it merely because the
  source package's core dependency list does not include it.
  The source checkout must be clean under `gr00t` and exactly at commit
  `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`. The adapter verifies imported
  modules resolve to that checkout. The adapter rejects distributed execution and
  any process that sees other than one CUDA GPU; its receipt records the visible
  GPU count.
- An authorized, locally staged N1.5 checkpoint with `model_type=gr00t_n1_5`.
  Model, source, data, and upstream dependency access are prepared separately;
  this recipe does not provision infrastructure or download policy weights.
- A native-compatible LeRobot v2.0/v2.1 dataset: `meta/info.json`,
  `meta/episodes.jsonl`, native `meta/modality.json`, `meta/tasks.jsonl`, frozen
  `meta/stats.json`, per-episode parquet, and the observation videos. Convert
  v3 data before using this native N1.5 reader. Symlinks are rejected.
- A video reader backend that can decode the dataset's actual observation
  codec. Pass one of the pinned native reader choices with `--video-backend`
  (`torchcodec`, `decord`, or `torchvision_av`) and decode a real observation
  before spending GPU time. Do not silently replace source video. If an
  authorized codec derivative is necessary, retain and report the public source
  revision, original and derivative hashes, codec/frame-count parity, exact
  conversion command, and the lossy-conversion limitation.
- A native modality configuration matching the actual cameras, state fields,
  action frame, rotation convention, units, and checkpoint dimensions. Actions
  must use contiguous indices `0..H-1`. This adapter does not infer or convert
  robot action semantics, resize the action head, or add missing camera views.
- A full Workbench source checkout. In the dedicated native environment,
  the adapter additionally imports `pydantic>=2`, NumPy, and PyArrow. Use
  `PYTHONPATH` to expose Workbench source without replacing native N1.5's
  dependency pins with the N1.7 or full Workbench dependency closure.

Freeze normalization statistics before preparing weights: use statistics from
training data only, or a pre-existing calibration set. The adapter requires
valid precomputed statistics and refuses the native reader's fallback that
would recompute them from all episodes. It cannot establish their collection
provenance. Keep an external record of that provenance and the split protocol.

Transfer private data and artifacts through your authorized S3 storage, then
stage them in the dedicated runtime. Local flags below address those staged
files; they are not public Workbench S3 handoff commands. Customer data, labels,
recordings, runtime paths, and receipts stay outside Git and public PRs. The
adapter enables no external experiment tracker.

## Prepare one revision

Create a private JSON array with one entry per dataset episode, including both
training and evaluation episodes. `episode_index` is the integer native
identifier; `group` identifies a collection session, scene, or other correlated
group that must stay entirely within one split. In `outcome` mode, `success` is
an observed JSON boolean. In `uniform` mode it must be omitted, not inferred.
The following values are synthetic schema examples:

```json
[
  {"episode_index": 0, "split": "train", "success": true, "group": "training-session-a"},
  {"episode_index": 1, "split": "train", "success": false, "group": "training-session-b"},
  {"episode_index": 2, "split": "eval", "success": true, "group": "evaluation-session-a"}
]
```

For progress mode, each entry also needs a `progress` array with exactly one
finite value in [0, 1] per recorded frame and `progress_source: "measured"`
or `"arm"`. ARM progress additionally requires `reward_model_sha256`, the
64-character checkpoint digest. Calibrate measured signals before normalization.
Train reward models only on training groups; assess their calibration on
separate held-out data. Episode success remains the observed label, independent
of predicted progress. No dense-label or ARM-checkpoint example is fabricated.

Set these variables in the dedicated runtime. All paths refer to operator-owned
staged inputs or new output files; `WORKBENCH_ROOT` is the source checkout,
`GROOT_SOURCE` the pinned native checkout, and `GROOT_PYTHON` its interpreter.

```bash
: "${WORKBENCH_ROOT:?Set the Workbench source checkout}"
: "${GROOT_PYTHON:?Set the native N1.5 interpreter}"
: "${DATASET_ROOT:?Set the staged dataset directory}"
: "${FEEDBACK_PATH:?Set the private feedback JSON file}"
: "${WEIGHTS_PATH:?Set a new weights JSON path outside the dataset}"
: "${ACTION_HORIZON:?Set the checkpoint action horizon}"
export PYTHONPATH="$WORKBENCH_ROOT/npa/src"
"$GROOT_PYTHON" -m npa.workbench.groot.offline prepare \
  --dataset-root "$DATASET_ROOT" --feedback-path "$FEEDBACK_PATH" \
  --output-path "$WEIGHTS_PATH" --action-horizon "$ACTION_HORIZON" \
  --mode outcome
```

Use `--mode progress` when dense feedback is available. `--temperature` defaults
to 1 in every mode. Use `--mode uniform` for unlabeled demonstrations.
Preparation verifies episode coverage, row order, lengths, literal label types,
group isolation, and progress provenance. It hashes all
`meta`, `data`, and `videos` file paths and bytes. Hashing large video datasets
requires a complete read. It writes weights and identities, without copying
observations or modifying the dataset. Output files are never overwritten.

## Train and evaluate a candidate

```bash
: "${GROOT_SOURCE:?Set the pinned native N1.5 checkout}"
: "${BASE_MODEL:?Set the staged N1.5 checkpoint directory}"
: "${CANDIDATE_DIR:?Set a new candidate output directory}"
: "${DATA_CONFIG:?Set the native modality configuration name or module:ClassName}"
: "${EMBODIMENT_TAG:?Set the native embodiment tag}"
: "${TRAIN_STEPS:?Choose the optimizer step count}"
"$GROOT_PYTHON" -m npa.workbench.groot.offline train \
  --dataset-root "$DATASET_ROOT" --manifest-path "$WEIGHTS_PATH" \
  --groot-source "$GROOT_SOURCE" --base-model "$BASE_MODEL" \
  --output-dir "$CANDIDATE_DIR" --data-config "$DATA_CONFIG" \
  --embodiment-tag "$EMBODIMENT_TAG" --max-steps "$TRAIN_STEPS"
```

Training defaults to batch size 8, learning rate 0.0001, and seed 42. The vision
and language backbones stay frozen; the native action projector and diffusion
model train. The source and dataset identities are rechecked before model
loading. Only training frames reach the trainer; the held-out split is evaluated
after training with its unit weights. Native checkpoint weights,
`experiment_cfg/metadata.json`, trainer state, and `offline-training.json` are
written under the new output directory. After saving, the adapter reloads the
candidate with the native N1.5 model contract and records the verified action
dimensions and checkpoint digest. The receipt binds dataset, feedback manifest,
upstream source, and baseline/candidate checkpoint identities, and explicitly
records training-weight statistics, one decoded terminal observation's shape,
and the action-horizon rows masked beyond that recorded episode. It also records
`robot_evaluation_performed: false`. `optimizer_loss_trace` contains the native
trainer's finite, ordered, per-step optimizer loss records and must reach the
completed step. Native logs go to stderr; the module's result is JSON on stdout.
Resume is not supported; retain the original revision and use a new candidate
directory for another iteration.

Run baseline and candidate robot trials using the same frozen task distribution,
reset procedure, observation/action conventions, and trial schedule. Keep trials
independent of the training collection groups. Include an ordinary BC control
and multiple training seeds when assessing whether weighting helps. Offline
training loss is not a success-rate estimate or a deployment gate.

Record private rollout JSON with schema version
`npa.groot-offline-rollouts/v1`, `baseline_checkpoint_sha256` and
`candidate_checkpoint_sha256` from the training receipt, `protocol_sha256` for
the frozen evaluation protocol, and a nonempty `trials` array. Each trial has a
unique `trial_id`, `baseline_success`, and `candidate_success`, with literal
boolean outcomes. Then compare:

```bash
: "${ROLLOUTS_PATH:?Set the private matched-rollout JSON file}"
: "${COMPARISON_PATH:?Set a new comparison JSON path}"
"$GROOT_PYTHON" -m npa.workbench.groot.offline compare \
  --rollouts-path "$ROLLOUTS_PATH" \
  --training-receipt-path "$CANDIDATE_DIR/offline-training.json" \
  --output-path "$COMPARISON_PATH"
```

The report checks checkpoint identities against the receipt and computes counts,
success rates, per-policy Wilson 95% intervals, and paired wins/losses. These
intervals are descriptive per-policy intervals, not a paired significance test.
The report summarizes operator-recorded evidence; it does not attest that the
named policies actually ran or that resets were independent. It always leaves
deployment to operator review. Use observed failures to collect the next
revision, preserve the evaluation holdout, and repeat. Promote a checkpoint only
after real task evaluation; return artifacts to authorized private storage and
release only this run's compute through its owning deployment workflow.

## Validation boundary

Hermetic tests exercise real parquet alignment, byte identities, split isolation,
reward weights, terminal action masks, differentiable Torch loss weighting,
batch-size-one gradients, checkpoint binding, and comparison arithmetic. They
do not establish policy improvement or hardware compatibility.

`npa/tests/e2e/test_groot_n15_offline_live.py` runs real native CUDA training when
`NPA_INTEGRATION_E2E=1` and `NPA_GROOT_N15_OFFLINE_LIVE_CONFIG` points to a private
JSON configuration. Required fields are `python`, `dataset_root`,
`feedback_path`, `groot_source`, `base_model`, `data_config`, `embodiment_tag`,
`action_horizon`, `mode`, `max_steps`, `batch_size`, and
`video_backend=decord`; `temperature` defaults to 1. A supplied invalid config fails. An absent config skips and proves no
native training. This test checks finite training and held-out losses, completed
optimizer steps, a single visible CUDA device, saved-checkpoint reload evidence,
changed checkpoint identity, and changed native action-decoder tensors in
safetensors checkpoints; robot success still needs separate
matched trials. No GPU or robot result is claimed by this recipe's unit tests.
