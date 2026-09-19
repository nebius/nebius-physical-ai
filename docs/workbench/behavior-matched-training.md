# Matched stage-conditioning training for BEHAVIOR

This reference implementation fine-tunes the action path of the released RLC
BEHAVIOR policy while holding the task and stage predictors fixed. It compares
two training arms that use the same examples, ordering, optimizer, random seed,
and parameter partition:

- **teacher** uses the released demonstration's equal-time stage bin;
- **replay** uses the published parent policy's filtered stage prediction from
  the preceding replan point.

The implementation is in
[`workflows/implementations/behavior-matched-training`](../../workflows/implementations/behavior-matched-training/).
The native B200 preflight has passed checkpoint restoration, stage-inference
consistency, and one discarded optimizer update. Full matched training and
policy-improvement evaluation remain pending.

## What is trained

Both arms run 3,600 updates with batch size 16, four flow samples, a 600-step
warmup to `5e-6`, and cosine decay to `1e-6`. The sampler contributes an equal
number of examples from tasks 0, 1, and 22. At the planned 57,600 examples,
each task contributes 19,200 examples.

Only the 23 action-path leaves declared in `action_partition.py` are trainable.
They cover the second language-model expert, action input/output projections,
time MLP, and key/value transform. Vision, language, task selection, stage
classification, gates, and fusion remain frozen. Stage and FAST auxiliary loss
weights are zero. The released normalization statistics and 30-action targets
remain unchanged.

## Runtime inputs

The repository contains no checkpoint, training data, credentials, or runtime
cache. Supply the paths described by `runtime-inputs.schema.json`. The source,
checkpoint, dataset, split, validation record, and adapter must match the
identities pinned in `trace-identities.json` and the checks in
`matched_train.py`.

The released policy source, weights, and demonstrations retain their upstream
terms. Fetch them at runtime using the operator's own authorized access. Do not
bake or redistribute those payloads with this implementation.

The dataset view must contain ordinary RGB observations, state, actions, and
training metadata. The split contains 180 training and 20 holdout episodes for
each task. Development, reporting, hidden-test, success, and termination data
must not enter prefix replay or optimization.

## Run the matched pair

Use the Python environment from the pinned upstream RLC/OpenPI checkout. The
commands below show the data flow; replace the paths with one validated runtime
input record.

```bash
IMPL=workflows/implementations/behavior-matched-training
RUNTIME=/runtime

$RUNTIME/openpi/.venv/bin/python $IMPL/gpu_preflight.py \
  --source-root $RUNTIME/source/behavior-policy \
  --adapter-root $RUNTIME/npa-behavior-adapter \
  --checkpoint $RUNTIME/checkpoints/published-parent \
  --checkpoint-archive $RUNTIME/checkpoints/published-parent.zip \
  --dataset-root $RUNTIME/datasets/released-rgb-view \
  --episode-split $RUNTIME/manifests/episode-split.json \
  --config $IMPL/config-teacher.json \
  --output $RUNTIME/output/gpu-prefix-preflight.json
```

The preflight requires exactly one B200. It verifies source and checkpoint
identities, loads one real training sample for each task, and checks that
canonical batch-one inference produces identical stage logits with one and 20
denoising steps. The sampled actions are discarded. It also runs one discarded
native update, checks that the 23 action paths are the only parameter and EMA
paths that can change, and verifies that all frozen paths remain byte-equal.
The training wrapper preserves frozen EMA leaves explicitly: applying the
native moving-average arithmetic to an unchanged low-precision weight can
otherwise change its stored value through rounding.

Prefix replay uses those canonical batch-one, one-step stage logits. The
direct 15-way classifier output is stored as auxiliary diagnostic data and
never drives the controller. The selected LeRobot v3 reader derives episode
boundaries from validated metadata and checks episode, frame, task, and
absolute-row identities before decoding observations.
CPU workers decode and transform fixed batches in order. Auxiliary classifier
batches pad their final partial batch and discard the padding. Canonical replay
decisions still use one original sample per inference call.

Generate prefix records and replay traces for both released splits:

```bash
for split in training holdout; do
  $RUNTIME/openpi/.venv/bin/python $IMPL/generate_prefix_records.py \
    --source-root $RUNTIME/source/behavior-policy \
    --adapter-root $RUNTIME/npa-behavior-adapter \
    --checkpoint $RUNTIME/checkpoints/published-parent \
    --checkpoint-archive $RUNTIME/checkpoints/published-parent.zip \
    --dataset-root $RUNTIME/datasets/released-rgb-view \
    --episode-split $RUNTIME/manifests/episode-split.json \
    --config $IMPL/config-teacher.json --split $split \
    --output $RUNTIME/output/$split-prefix.jsonl

  $RUNTIME/openpi/.venv/bin/python $IMPL/build_replay_trace.py \
    --prefix-records $RUNTIME/output/$split-prefix.jsonl \
    --identities $IMPL/trace-identities.json --split $split \
    --output $RUNTIME/output/$split-trace.jsonl
done
```

### Generate prefixes on two GPUs

Run the generation command on each GPU with `--shard-count 2` and a distinct
`--shard-index 0` or `--shard-index 1`. Name each output
`$split-prefix-$SHARD_INDEX.jsonl`. Shards contain alternating complete episodes
in the frozen split order; they keep every replan frame and the same native
batch-one inference. No training examples or stage-controller settings change.

After transferring both outputs to the same runtime, merge each split before
building its replay trace:

```bash
for split in training holdout; do
  $RUNTIME/openpi/.venv/bin/python $IMPL/merge_prefix_records.py \
    --episode-split $RUNTIME/manifests/episode-split.json --split $split \
    --shard-input $RUNTIME/output/$split-prefix-0.jsonl \
    --shard-input $RUNTIME/output/$split-prefix-1.jsonl \
    --output $RUNTIME/output/$split-prefix.jsonl \
    --receipt $RUNTIME/output/$split-prefix-merge.json
done
```

The merger requires exact shard membership and frame order. Its receipt records
the input and output hashes. Both training arms consume the same merged traces.

Freeze each arm's inputs, then invoke `matched_train.py`. The trainer performs
two consecutive compiled updates on disposable state before the planned run.
The first checks finite loss, gradient norm, and parameter norm; byte equality
for every frozen leaf; and changes confined to the action allowlist. Both
updates pass through the native metric reduction and numeric formatter.
Reporting metrics are converted to scalar float32 after the model update.
The planned run rebuilds its state and data loader from the original seed.

```bash
for arm in teacher replay; do
  $RUNTIME/openpi/.venv/bin/python $IMPL/freeze_inputs.py \
    --config $IMPL/config-$arm.json \
    --episode-split $RUNTIME/manifests/episode-split.json \
    --source-manifest $RUNTIME/manifests/source-manifest.json \
    --dataset-validation $RUNTIME/manifests/dataset-validation.json \
    --training-trace $RUNTIME/output/training-trace.jsonl \
    --trace-identities $IMPL/trace-identities.json \
    --adapter-files $RUNTIME/manifests/adapter-files.json \
    --output $RUNTIME/output/$arm-inputs.json

  $RUNTIME/openpi/.venv/bin/python $IMPL/matched_train.py \
    --source-root $RUNTIME/source/behavior-policy \
    --adapter-root $RUNTIME/npa-behavior-adapter \
    --checkpoint $RUNTIME/checkpoints/published-parent \
    --checkpoint-archive $RUNTIME/checkpoints/published-parent.zip \
    --dataset-root $RUNTIME/datasets/released-rgb-view \
    --output-root $RUNTIME/output/$arm \
    --config $IMPL/config-$arm.json \
    --input-manifest $RUNTIME/output/$arm-inputs.json \
    --episode-split $RUNTIME/manifests/episode-split.json \
    --source-manifest $RUNTIME/manifests/source-manifest.json \
    --dataset-validation $RUNTIME/manifests/dataset-validation.json \
    --training-trace $RUNTIME/output/training-trace.jsonl
done
```

## Holdout selection and export

Each arm retains updates 600, 1200, 1800, 2400, 3000, and 3599. Score every
checkpoint on all 20 held-out released episodes per task with eight deterministic
flow draws. Selection minimizes the equal-task mean action loss; exact ties pick
the earlier update. Stage cross-entropy and per-action-dimension losses are
diagnostics and do not affect selection.

Run `evaluate_holdout.py` with the arm's checkpoint directory, frozen input
manifest, and holdout trace. Then pass its `selection.json` to
`export_selected.py`. The export contains the selected EMA parameters and the
unchanged normalization assets. A selected offline checkpoint is still
`not_rollout_evaluated`; measure it with the unchanged official evaluator before
making a policy-quality claim.

```bash
for arm in teacher replay; do
  candidate=$(case $arm in
    teacher) echo stage_teacher_action_only_3600 ;;
    replay) echo stage_parent_replay_action_only_3600 ;;
  esac)
  checkpoint_root=$RUNTIME/output/$arm/checkpoints/pi_behavior_b1k_fast/$candidate

  $RUNTIME/openpi/.venv/bin/python $IMPL/evaluate_holdout.py \
    --source-root $RUNTIME/source/behavior-policy \
    --adapter-root $RUNTIME/npa-behavior-adapter \
    --checkpoint $RUNTIME/checkpoints/published-parent \
    --checkpoint-archive $RUNTIME/checkpoints/published-parent.zip \
    --checkpoint-root $checkpoint_root \
    --dataset-root $RUNTIME/datasets/released-rgb-view \
    --episode-split $RUNTIME/manifests/episode-split.json \
    --config $IMPL/config-$arm.json \
    --input-manifest $RUNTIME/output/$arm-inputs.json \
    --holdout-trace $RUNTIME/output/holdout-trace.jsonl \
    --output $RUNTIME/output/$arm-holdout-losses.jsonl \
    --selection $RUNTIME/output/$arm-selection.json

  $RUNTIME/openpi/.venv/bin/python $IMPL/export_selected.py \
    --checkpoint-root $checkpoint_root \
    --selection $RUNTIME/output/$arm-selection.json \
    --output-root $RUNTIME/output/$arm-selected
done
```
