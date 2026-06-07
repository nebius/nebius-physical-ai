# Sim2Real VLM-to-RL Runbook

This workflow runs the full Sim2Real chain as one inspectable pipeline:

`LeRobot dataset trigger -> augment -> env generation -> train/held-out split -> action rollouts -> VLM critique -> RL signal -> trainer update -> held-out eval -> promote or loop back -> external validation stub -> retrigger`.

Steps 2 and 12 are documented external stubs. Every other step writes local
artifacts and, when `--upload-artifacts` is set, uploads the run tree to S3.

## Easy-Parameters Quickstart

Use this when you want the canonical `lerobot/pusht` demo shape with the fewest
knobs. The trigger path is the input path: dropping a LeRobot dataset there is
what starts a run. Keep the trigger path, simulation asset source path, and
output prefix separate.

```bash
# 1. Easy parameters.
export NPA_SIM2REAL_RUN_ID=pusht-demo-$(date -u +%Y%m%dT%H%M%SZ)
export NPA_SIM2REAL_BUCKET=<default-platform-bucket>
export NPA_SIM2REAL_PREFIX=""
export NPA_SIM2REAL_TRIGGER_DATASET_ID=lerobot/pusht
export NPA_SIM2REAL_TRIGGER_DATASET_URI="s3://${NPA_SIM2REAL_BUCKET}/sim2real-triggers/${NPA_SIM2REAL_RUN_ID}/lerobot-pusht/"
export ASSETS_URI="s3://${NPA_SIM2REAL_BUCKET}/sim2real-assets/pusht/"
export SCENE_SPEC_URI="s3://${NPA_SIM2REAL_BUCKET}/sim2real-assets/pusht/scene-spec.json"

# 2. Credentials and endpoints.
export AWS_ACCESS_KEY_ID=<access-key>
export AWS_SECRET_ACCESS_KEY=<secret-key>
export AWS_ENDPOINT_URL=<default-platform-s3-endpoint>

# Non-default S3-compatible endpoints supported but untested.

# 3. Single LeRobot trainer image override. Leave unset to use the reference image.
export TRAINER_IMAGE=<registry>/npa-lerobot-vlm-rl:0.1.0

# 4. Reference image defaults. Override only if you have a newer pushed image.
export AUGMENT_IMAGE=<registry>/npa-sim2real-envgen:0.1.1
export POLICY_IMAGE=<registry>/npa-sim2real-reference-policy:0.1.1
export VLM_IMAGE=<registry>/npa-cosmos3-reason:3.0.0
export EVAL_IMAGE=<registry>/npa-sim2real-eval:0.1.0

# 5. Demo scale. Increase these for larger production runs.
export INNER_ITERATIONS=2
export OUTER_ITERATIONS=1
export LOOP_OF_LOOPS_ITERATIONS=1
export ROLLOUT_COUNT=3
export STEPS_PER_ROLLOUT=4
export HELDOUT_ENV_COUNT=8

npa workbench sim2real run \
  --run-id "${NPA_SIM2REAL_RUN_ID}" \
  --s3-bucket "${NPA_SIM2REAL_BUCKET}" \
  --s3-prefix "${NPA_SIM2REAL_PREFIX}" \
  --s3-endpoint "${AWS_ENDPOINT_URL}" \
  --trigger-dataset-uri "${NPA_SIM2REAL_TRIGGER_DATASET_URI}" \
  --trigger-dataset-id "${NPA_SIM2REAL_TRIGGER_DATASET_ID}" \
  --assets-uri "${ASSETS_URI}" \
  --scene-spec-uri "${SCENE_SPEC_URI}" \
  --inner-iterations "${INNER_ITERATIONS}" \
  --outer-iterations "${OUTER_ITERATIONS}" \
  --loop-of-loops-iterations "${LOOP_OF_LOOPS_ITERATIONS}" \
  --rollout-count "${ROLLOUT_COUNT}" \
  --steps-per-rollout "${STEPS_PER_ROLLOUT}" \
  --heldout-env-count "${HELDOUT_ENV_COUNT}" \
  --upload-artifacts
```

Canonical S3 layout for the quickstart:

```text
s3://<bucket>/sim2real-triggers/<run-id>/lerobot-pusht/    # Step 1 trigger path
s3://<bucket>/sim2real-assets/pusht/                       # Step 2 sim assets and SceneSpec stub input
s3://<bucket>/<run-id>/                                     # Per-run output tree
```

The output tree includes:

```text
augment/
envs/raw/
envs/train/
envs/heldout/
actions/
vlm_eval/
training_signal/
inner_loop/
checkpoints/
eval/heldout/
outer_loop/decision.json
stage_13_retrigger/retrigger.json
reports/sim2real-report.json
```

## Prerequisites

- Python 3.11 or newer and this package installed in `npa/.venv`.
- A Kubernetes GPU cluster with schedulable RTX PRO 6000 class `sm_120` GPUs.
- Pushed reference images:
  - `npa-sim2real-envgen:0.1.1`
  - `npa-sim2real-reference-policy:0.1.1`
  - `npa-cosmos3-reason:3.0.0`
  - `npa-lerobot-vlm-rl:0.1.0`
  - `npa-sim2real-eval:0.1.0`
- Gated model repository access accepted where required by the VLM image.
- `HF_TOKEN` and `NGC_API_KEY` supplied through environment variables or a
  Kubernetes secret such as `hf-ngc-tokens`.
- S3-compatible storage credentials and endpoint configured through environment
  variables, project config, or Kubernetes secrets.

## Run All Three Tiers

Raw SkyPilot:

```bash
cat > /tmp/sim2real-skypilot-k8s.yaml <<'YAML'
kubernetes:
  pod_config:
    spec:
      serviceAccountName: agent-sa
      envFrom:
        - secretRef:
            name: hf-ngc-tokens
YAML

sky jobs launch \
  --config /tmp/sim2real-skypilot-k8s.yaml \
  --infra k8s/<cluster-name> \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  npa/workflows/workbench/sim2real/runbook.yaml
```

SDK:

```python
from npa.sdk.workbench import sim2real

report = sim2real.run(
    run_id="pusht-sdk-demo",
    s3_bucket="<bucket>",
    s3_prefix="pusht-sdk-demo",
    trigger_dataset_uri="s3://<bucket>/sim2real-triggers/pusht-sdk-demo/lerobot-pusht/",
    trigger_dataset_id="lerobot/pusht",
    assets_uri="s3://<bucket>/sim2real-assets/pusht/",
    scene_spec_uri="s3://<bucket>/sim2real-assets/pusht/scene-spec.json",
    threshold=0.75,
    inner_iterations=2,
    outer_iterations=1,
    upload_artifacts=True,
)
print(report["outer_loop"]["latest_decision"])
```

CLI:

```bash
npa workbench sim2real run \
  --run-id pusht-cli-demo \
  --s3-bucket <bucket> \
  --s3-prefix pusht-cli-demo \
  --trigger-dataset-uri s3://<bucket>/sim2real-triggers/pusht-cli-demo/lerobot-pusht/ \
  --trigger-dataset-id lerobot/pusht \
  --assets-uri s3://<bucket>/sim2real-assets/pusht/ \
  --scene-spec-uri s3://<bucket>/sim2real-assets/pusht/scene-spec.json \
  --inner-iterations 2 \
  --outer-iterations 1 \
  --upload-artifacts
```

Inner loop only:

```bash
npa workbench sim2real inner-loop \
  --run-id sim2real-inner-example \
  --output-dir /tmp/sim2real-inner-example \
  --inner-iterations 2
```

## Stages And Artifacts

1. Trigger: consumes `--trigger-dataset-uri` and writes
   `stage_01_trigger/trigger.json`.
2. External assets and SceneSpec: documented BYO stub at
   `stage_02_assets/external_stub.json`.
3. Augmentation: writes `augment/manifest.json`.
4. Environment generation: writes `envs/raw/manifest.json`.
5. Train and held-out split: writes `envs/train/manifest.json` and
   `envs/heldout/manifest.json` using an 80/20 split.
6. Token manifest: writes `tokens/manifest.json`.
7. Action-conditioned rollouts: writes `actions/train/.../rollout-*/`.
8. VLM eval: writes structured critique JSON under `vlm_eval/train/`.
9. RL signal and trainer update: writes `training_signal/train/` and
   `inner_loop/.../evidence.json`.
10. Held-out eval: writes `eval/heldout/report.json`.
11. Threshold gate: writes `outer_loop/decision.json`; when the threshold is
    met it writes `checkpoints/candidate/candidate.json`, otherwise
    `outer_loop/loopback.json` points back to Stage 7.
12. Real-robot validation: documented external stub at
    `stage_12_external_validation/external_stub.json`.
13. Retrigger: writes `stage_13_retrigger/retrigger.json`, targeting Stage 1
    when a new real-world LeRobot dataset lands in the trigger path.

## Loops

Inner loop, Stages 7 to 9:

```text
Reference action generation -> VLM eval -> critique-to-reward signal -> trainer update
```

Outer loop, Stages 10 to 11:

```text
held-out eval -> threshold gate -> promote checkpoint or loop back to Stage 7
```

Loop-of-loops, Stages 12 to 13 to 1:

```text
real-robot validation stub -> retrigger manifest -> next LeRobot dataset batch in trigger path
```

The VLM eval schema is:

```json
{
  "schema": "npa.sim2real.vlm_eval.v1",
  "rollout_id": "rollout-0000",
  "success": false,
  "per_step": [
    {"step": 0, "critique_text": "...", "error_tags": ["missed_target"]}
  ],
  "summary": "..."
}
```

The RL signal schema is:

```json
{
  "schema": "npa.sim2real.rl_signal.v1",
  "rollout_id": "rollout-0000",
  "per_step": [
    {
      "step": 0,
      "reward": -0.35,
      "advantage": -0.1,
      "target": {
        "nl_correction": "Move the end effector toward the object center before closing.",
        "action_delta": [0.12, 0.02, 0.0]
      }
    }
  ]
}
```

The reference trainer integration point is after the LeRobot policy forward pass
and before `optimizer.step()`:

```text
loss = imitation_loss
     + signal_loss_weight * corrective_mse
     - advantage * policy_logit_proxy
```

## BYO Seams

Every seam is available in raw SkyPilot envs, SDK keyword arguments, and CLI
options:

- `s3_endpoint`, `s3_bucket`, `s3_prefix`
- `trigger_dataset_uri`, `trigger_dataset_id`
- `assets_uri`, `scene_spec_uri`
- `augment_image`
- `action_rollouts_uri`, `train_envs_uri`, `heldout_envs_uri`
- `policy_image`
- `vlm_image`, `vlm_model`, `byo_vlm_command`
- `byo_signal_converter`
- `trainer_image`, `byo_trainer_command`
- `eval_image`, `byo_eval_command`
- `threshold`
- `inner_iterations`, `outer_iterations`, `loop_of_loops_iterations`
- `rollout_count`, `steps_per_rollout`, `heldout_env_count`
- `signal_loss_weight`, `learning_rate`
- `no_guardrails`

## Scale Knobs

The demo scale intentionally exercises every stage with small numbers. To scale
toward large environment generation runs, increase:

- `HELDOUT_ENV_COUNT` for generated environment count and held-out eval breadth.
- `ROLLOUT_COUNT` and `STEPS_PER_ROLLOUT` for action and VLM-eval volume.
- `INNER_ITERATIONS` for repeated critique-to-reward trainer updates.
- `OUTER_ITERATIONS` for held-out failures to loop back through Stage 7.
- `LOOP_OF_LOOPS_ITERATIONS` when real-world validation should start a next
  dataset-triggered run.

For augmentation-heavy runs, shard the trigger dataset by prefix and submit
multiple SkyPilot jobs with distinct `NPA_SIM2REAL_RUN_ID` values and output
prefixes. Keep each job pointed at the same pushed reference images unless a
new image has already been built and pushed.
