# GR00T N1.7 closed-loop task performance

Use the checked-in `groot-1-7-finetune.yaml` workflow to compare identity-pinned
baseline and trained NVIDIA GR00T N1.7 checkpoints on the exact task established
from dataset provenance. The checked-in reference proves canonical PushT and
runs real `Gr00tPolicy.get_action` outputs in live `gym_pusht/PushT-v0` physics;
it is not offline dataset replay or action-regression scoring. Its serial graph is:

```text
access_capacity_preflight -> resolve_task_contract -> prepare_retraining_split ->
evaluate_offline_baseline -> retrain_task_policy -> resolve_trained_checkpoint ->
probe_checkpoint_2500 -> probe_checkpoint_5000 -> probe_checkpoint_10000 ->
select_offline_checkpoint -> compare_offline_learning -> emit_offline_mcap ->
emit_offline_rrd -> evaluate_validation_baseline ->
evaluate_validation_candidate -> analyze_validation_outcomes ->
select_checkpoint -> evaluate_baseline_closed_loop ->
evaluate_trained_closed_loop -> analyze_paired_outcomes -> render_task_rollouts ->
emit_mcap -> emit_rrd -> publish
```

Both checkpoints use the same reserved deterministic seeds, initial states,
horizon, current simulator observations, action clipping, and success predicate.
Every environment action comes from the corresponding model. The paired report
includes task success, maximum goal coverage, returns, termination reasons,
confidence intervals, and a paired test. Publication fails unless actual task
outcomes improve. The UI and recordings label the platform **Simulated**; they
make no physical-robot claim.

## Reproducible contract

The workbench image and command pin and verify:

- base model `nvidia/GR00T-N1.7-3B` at Hugging Face revision
  `2fc962b973bccdd5d8ce4f67cc63b264d6886495`;
- GR00T package version `0.1.0`;
- Isaac-GR00T source revision
  `3df8b3825d67f755e69141446f4315f281b9b7e6`.

The task-contract stage pins `lerobot/pusht`, `gym-pusht==0.1.6`, the upstream
source revision, action meaning (absolute pusher x/y in workspace pixels), the
96×96 RGB plus pusher-position observation, and the exact `coverage > 0.95`
success implementation. It compares logical data rows to the immutable upstream
dataset and checks that simulator frames change after a physics transition.
Checkpoint directory identities are verified before either policy is loaded.

## Validate and inspect

```bash
SPEC=npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml
RUN_ID=groot-n1-7-example

npa workbench workflow validate-spec "$SPEC"
npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID"
```

Preview the validated render without launching it:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --plan-only
```

The plan must show all 24 semantic phases. The default run uses eight GPUs, but
`gpu_count` is parameterized from one to the maximum schedulable count. Keep
`global_batch_size = gpu_count * per_device_batch_size *
gradient_accumulation_steps`; the checked-in default is 128. Preflight also
requires more than one epoch of train-split coverage. Validation and final
evaluation each use at least 24 paired episodes, distinct seed namespaces, and
identical conditions within each pair.

## Submit evaluation

The source data and baseline checkpoint must exist at immutable refs. The
workflow materializes the split, trains and hashes the candidate, selects it on
validation seeds, and evaluates fresh final seeds:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket=<bucket> \
  --var source_data_uri=s3://<bucket>/datasets/my-groot-dataset/ \
  --var gpu_count=8 \
  --var per_device_batch_size=1 \
  --var gradient_accumulation_steps=16 \
  --var global_batch_size=128 \
  --var train_episodes=154 \
  --var heldout_episodes=26 \
  --var final_episodes=26 \
  --var max_steps=10000 \
  --var validation_episodes=24 \
  --var paired_episodes=24 \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --secret-env HF_TOKEN \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Do not reduce either seed set below 24, substitute a scripted controller, or
change the final seeds after inspecting their results. The workflow never reads
the final split or final seed namespace until positive-skill, nontrivial-effect,
repeat-noise, per-dimension, robust-loss, and paired validation gates pass.

## Find the run and outputs

Successful text output includes `run_id: <value>`; JSON automation can add
`--output-format json` and read the top-level `run_id` field. To rediscover
runs later from their persisted NPA manifests:

```bash
npa workbench workflow list \
  --s3-bucket <bucket> \
  --workflow-s3-prefix groot-1-7-task-performance \
  --json
```

With the default prefix, checkpoints and provenance are under:

```text
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/task-contract.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/rigor-preflight.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/split/manifest.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/offline/baseline/evaluation.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/offline/probes/checkpoint-{2500,5000,10000}.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/offline/selected-checkpoint.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/checkpoints/candidate/
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/learning-rigor-report.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/offline-learning.{mcap,rrd}
s3://<bucket>/groot-1-7-task-performance/<run-id>/validation/report.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/selected-checkpoint.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/eval/baseline/evaluation.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/eval/trained/evaluation.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/rollouts/{baseline,trained}/
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/task-performance-report.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/paired-task-performance.mp4
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/task-performance.mcap
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/task-performance.rrd
s3://<bucket>/groot-1-7-task-performance/<run-id>/reports/publish-manifest.json
s3://<bucket>/groot-1-7-task-performance/<run-id>/workflow.yaml
```

MCAP exposes `/rollout/{baseline,trained}/camera` and action topics, plus
object/goal pose, task progress, success, aggregate rates, paired deltas, and
logs. Rerun opens on synchronized baseline/trained cameras, object/goal
trajectory, coverage, success/termination, aggregate rates, paired deltas, and
the confidence interval. Training loss and offline action MSE are secondary.
The offline recordings additionally expose exactly one per-horizon MSE value
for every configured action-horizon position, the checkpoint validation curve,
real optimizer-step loss, trivial floors, repeat spread, and weight identities.

The terminal `publish` stage downloads and independently parses both recording
formats. It succeeds only after their run IDs, schemas/topics, Rerun identity,
timeline/entities, provenance, nonzero sizes, and the exact submitted workflow
artifact have all passed validation.

If `prefix` or `checkpoint_uri` is overridden, use those resolved locations
and pass the corresponding parent prefix to `workflow list`.

## Kubernetes image prerequisites

The image keeps its non-root `ubuntu` runtime while enabling the repository's
SkyPilot Kubernetes prerequisite layer. That layer supplies the system Python,
rsync/SSH client, netcat, and passwordless `sudo` required by SkyPilot's in-pod
bootstrap. If a historical image exits before setup with `sudo: command not
found`, build the additive repair tag with
`npa/docker/workbench/groot/Dockerfile.k8s-prereqs`; do not clear the workflow's
image pin or switch the task to root.

Historical `0.1.0` images also removed `linux-libc-dev` after build while
retaining `libc6-dev`, which makes later `apt` operations fail dependency
resolution. The derived recipe repairs that package relationship before adding
the prerequisites. Fresh source builds retain the current header package so
SkyPilot can install any remaining SSH runtime packages during bootstrap.
