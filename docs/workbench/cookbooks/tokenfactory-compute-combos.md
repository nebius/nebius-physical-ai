# Token Factory with Nebius GPU workloads

These workflows train or roll out a policy on Nebius GPUs, then use hosted
Token Factory inference to interpret the resulting artifacts. Start with the
[composition guide](../composing-cloud-and-token-factory.md) for credentials,
storage, and the producer/consumer contract.

## Prepare the runtime and inputs

Complete [Workbench setup](../getting-started.md), including the selected
Kubernetes context, S3 credentials, and `npa skypilot bootstrap`. The checked-in
GPU stages request one H100, 16 CPUs, and 64 GiB memory; hosted stages request
4 CPUs and 16 GiB. Verify the models available to your Token Factory key.

The two rollout specs use the hosted judge's `MiniMaxAI/MiniMax-M3` default.
The judge toolRefs now pass an explicit `vlm_model` override through to the CLI;
use the same `--var vlm_model=...` in planning and submission to select another
available vision model. An empty override preserves the backend default. Confirm
inference access before launching; model-list membership alone is insufficient.
You can also score existing rollouts directly with
`vlm-eval run --backend api --model "<available-vision-model>"`, or use
`vlm-eval loop` for separate rollout scores.

Use a unique run ID and replace the bucket, project, context, and checkpoint
placeholders. The rollout examples need a checkpoint compatible with the
selected LeRobot image, including its processor configuration. The legacy
`lerobot/diffusion_pusht` checkpoint lacks the processor files required by
LeRobot 0.6; the specs do not supply a ready-to-use policy automatically.
Stage a compatible checkpoint under the input URI before submitting.

## Roll out a policy, then judge the result

The first state runs `workbench.lerobot.policy_rollout` and uploads videos; the
second runs hosted `workbench.vlm_eval.run` on that same prefix.

```bash
spec=npa/workflows/workbench/npa-workflows/tokenfactory-rollout-judge-combo.yaml
bucket="<your-bucket>"
run_id="<unique-run-id>"
checkpoint="s3://${bucket}/inputs/policy/"

npa workbench workflow validate-spec "$spec"
npa workbench workflow plan-spec "$spec" --run-id "$run_id" \
  --var "bucket=${bucket}" --var "policy_checkpoint=${checkpoint}" --json

npa workbench workflow submit "$spec" \
  --project "<project-alias>" --infra "k8s/<context>" \
  --run-id "$run_id" --stage-src \
  --var "bucket=${bucket}" --var "policy_checkpoint=${checkpoint}" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

The spec defaults to the `pusht` environment. Set `--var rollout_env=...` only
with a compatible policy and environment. For custom destinations, use
`rollouts_uri` and `scores_uri`; uppercase `ROLLOUTS_URI` and `JUDGE_URI` do not
override those config keys. Images resolve from the supported release catalog;
use `--image-override TOOL_REF=IMAGE` for a deliberate per-tool override.

Outputs are under
`s3://<your-bucket>/tokenfactory-rollout-judge/<run-id>/`: inspect videos in
`rollouts/` and the report in `scores/vlm_eval_stub.json`. The filename is
historical; check that the report records the `api` backend, actual rollout
inputs, requested `model`, actual `served_model`, score, rationale, and pass/fail
outcome. Hosted judges reject incomplete responses, malformed JSON, and invalid
scores without repair. A successful command alone does not establish that the
policy completed the task.

The shipped judge uses `vlm-eval run`, which produces one score across the
selected frames in its input prefix. Its default two rollout episodes are not
scored independently. Use `vlm-eval loop` on a compatible set of rollout
directories when you need per-rollout reports and an aggregate success rate.

## Plan from a scene, then judge the rollout against it

Use `tokenfactory-scene-to-rollout-judge.yaml` for the three-stage chain:
scene reasoning → GPU rollout → judgment against the saved plan. The scene,
task, policy, and rollout environment must describe the same task.

The reasoner toolRef passes the input/output paths and `reason_model` to the
CLI; the spec selects `MiniMaxAI/MiniMax-M3`. Use the same
`--var reason_model=...` in planning and submission for an explicit alternative.
The `reason_task` config key still does not reach this command: it uses the
CLI's built-in task. Inspect the saved task and plan. For a chosen task, run
`token-factory reason --model ... --task ...` directly as shown in the
[integration guide](../token-factory.md#generate-and-inspect-artifacts).

```bash
spec=npa/workflows/workbench/npa-workflows/tokenfactory-scene-to-rollout-judge.yaml
run_id="<another-unique-run-id>"
scene_uri="s3://${bucket}/inputs/scene/"

npa workbench workflow validate-spec "$spec"
npa workbench workflow plan-spec "$spec" --run-id "$run_id" \
  --var "bucket=${bucket}" --var "policy_checkpoint=${checkpoint}" \
  --var "scene_uri=${scene_uri}" --json

npa workbench workflow submit "$spec" \
  --project "<project-alias>" --infra "k8s/<context>" \
  --run-id "$run_id" --stage-src \
  --var "bucket=${bucket}" --var "policy_checkpoint=${checkpoint}" \
  --var "scene_uri=${scene_uri}" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Upload scene images before submission. The plan is `plan/scene_reasoning.json`;
the judge consumes its `analysis`
through `--task-from`. Inspect that plan, generated videos in `rollouts/`, and
`vlm-judge/vlm_eval_stub.json` under
`s3://<your-bucket>/tokenfactory-scene-to-rollout-judge/<run-id>/`.

## Train, then triage

[`tokenfactory-train-triage.yaml`](../../../npa/workflows/workbench/npa-workflows/tokenfactory-train-triage.yaml)
runs LeRobot training and publishes its checkpoint, config, logs, and metrics
before a hosted text model produces a triage report. Use the same
validate/plan/submit sequence above with this spec and your `bucket` override.

Set `lerobot_dataset`, `policy_type`, `train_steps`, and `train_batch_size` for
your training objective, and `triage_model` for your available text model. The
shipped values are a one-step wiring check, not evidence of a learned policy.
The training output is under `artifacts/`; the hosted report is
`triage/generations.jsonl`, beneath
`s3://<your-bucket>/tokenfactory-train-triage/<run-id>/`. Compare its claims with
the original metrics and logs.

## Serverless runner alternatives

The existing Python runners use Serverless GPU Jobs for training:

| Runner | Behavior |
| --- | --- |
| `npa/scripts/run_tokenfactory_train_triage.py` | Train, download textual artifacts, then write a triage report. `--from-output-path` triages an existing run. |
| `npa/scripts/run_tokenfactory_sim_sweep.py` | Generate experiment rationale, train a deterministic grid, then rank the real run artifacts. `--rank-existing` ranks supplied run prefixes. |

From the repository root, inspect their plans and options with the repository
environment:

```bash
npa/.venv/bin/python npa/scripts/run_tokenfactory_train_triage.py --render-only
npa/.venv/bin/python npa/scripts/run_tokenfactory_train_triage.py --help
npa/.venv/bin/python npa/scripts/run_tokenfactory_sim_sweep.py --render-only
npa/.venv/bin/python npa/scripts/run_tokenfactory_sim_sweep.py --help
```

Both runners default to smoke training. `--no-smoke` selects the full training
path; the triage runner also accepts `--steps`. The sweep varies training steps
on a fixed grid; the model supplies rationale and ranking, not executable
hyperparameters. Configure real training settings before treating the output
as policy-quality evidence.

For all paths, inspect actual output artifacts and use the
[run lifecycle](../../run-lifecycle.md) for status and recovery. Finish or
cancel owned jobs before following [safe teardown](../../teardown.md).
