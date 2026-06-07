# Sim2Real VLM-to-RL Runbook

This workflow wires a full Sim2Real loop from trigger through augmentation,
environment split, action-conditioned rollouts, VLM critique, VLM-derived RL
signal conversion, policy update, held-out evaluation, threshold gating, and
retrigger.

## Prerequisites

- Python 3.11 or newer and the `npa` package installed in `npa/.venv`.
- A Kubernetes GPU cluster with RTX PRO 6000 class `sm_120` GPUs.
- A registry containing these images:
  - `npa-cosmos3-reason:3.0.0`
  - `npa-lerobot-vlm-rl:0.1.0`
  - `npa-sim2real-eval:0.1.0`
  - the CUDA 13 multi-arch base and Genesis reference images used by the site.
- Hugging Face gated repository access accepted for:
  - `nvidia/Cosmos-Transfer2.5-2B`
  - `nvidia/Cosmos-Predict2.5-2B`
  - `nvidia/Cosmos-Guardrail1`, unless running with `--no-guardrails`
- `HF_TOKEN` and `NGC_API_KEY` supplied through environment variables or a
  Kubernetes secret such as `hf-ngc-tokens`.
- S3-compatible storage credentials through environment variables or the NPA
  credentials loader. GCP/GCS S3-compatible endpoints are supported through the
  same `AWS_ENDPOINT_URL`/`S3_ENDPOINT_URL` seam but are not yet covered by CI.

## Run

Standalone raw SkyPilot:

```bash
cat > /tmp/sim2real-skypilot-k8s.yaml <<'YAML'
kubernetes:
  pod_config:
    spec:
      serviceAccountName: agent-sa
      imagePullSecrets:
        - name: <registry-pull-secret>
      envFrom:
        - secretRef:
            name: hf-ngc-tokens
YAML

export NPA_SIM2REAL_RUN_ID=sim2real-example
export NPA_SIM2REAL_BUCKET=<bucket>
export NPA_SIM2REAL_PREFIX=sim2real-b
export AWS_ENDPOINT_URL=<s3-compatible-endpoint>
export ACTION_ROLLOUTS_URI=s3://<bucket>/<trigger-run>/actions/train/
export TRAIN_ENVS_URI=s3://<bucket>/<trigger-run>/envs/train/envs.jsonl
export HELDOUT_ENVS_URI=s3://<bucket>/<trigger-run>/envs/heldout/envs.jsonl
export ASSETS_URI=s3://<bucket>/<asset-prefix>/
export SCENE_SPEC_URI=s3://<bucket>/<asset-prefix>/scene-spec.json
export NPA_SOURCE_REPO=<https-git-source-url>
export TRAINER_IMAGE=<registry>/npa-lerobot-vlm-rl:0.1.0
export VLM_IMAGE=<registry>/npa-cosmos3-reason:3.0.0
export EVAL_IMAGE=<registry>/npa-sim2real-eval:0.1.0
sky jobs launch \
  --config /tmp/sim2real-skypilot-k8s.yaml \
  --infra k8s/npa-rtxpro-mk8s \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  npa/workflows/workbench/sim2real/runbook.yaml
```

SDK:

```python
from npa.sdk.workbench import sim2real

report = sim2real.run(
    run_id="sim2real-sdk-example",
    output_dir="/tmp/sim2real-sdk-example",
    s3_bucket="<bucket>",
    threshold=0.75,
    inner_iterations=2,
    outer_iterations=1,
)
print(report["outer_loop"]["latest_decision"])
```

CLI:

```bash
npa workbench sim2real run \
  --run-id sim2real-cli-example \
  --output-dir /tmp/sim2real-cli-example \
  --s3-bucket <bucket> \
  --threshold 0.75 \
  --inner-iterations 2 \
  --outer-iterations 1 \
  --upload-artifacts
```

Individual inner loop:

```bash
npa workbench sim2real inner-loop \
  --run-id sim2real-inner-example \
  --output-dir /tmp/sim2real-inner-example \
  --inner-iterations 2
```

## Stages

1. Trigger: writes `stage_01_trigger/trigger.json`.
2. External assets and SceneSpec: documented BYO stub at
   `stage_02_assets/external_stub.json`.
3. Cosmos transfer augmentation: writes `augment/manifest.json`.
4. Environment generation: writes `envs/raw/manifest.json`.
5. Train and held-out split: writes `envs/train/manifest.json` and
   `envs/heldout/manifest.json`.
6. Token manifest: writes `tokens/manifest.json`.
7. Action-conditioned rollouts: writes `actions/train/.../rollout-*/`.
8. VLM eval: writes one structured JSON per rollout under `vlm_eval/train/`.
9. RL signal and trainer update: writes `training_signal/train/` plus
   `inner_loop/.../evidence.json`.
10. Held-out eval: writes `eval/heldout/report.json`.
11. Threshold gate: writes `outer_loop/decision.json`; when met, writes
   `checkpoints/candidate/candidate.json`; otherwise writes
   `outer_loop/loopback.json`.
12. External validation: documented BYO stub at
   `stage_12_external_validation/external_stub.json`.
13. Retrigger: writes `stage_13_retrigger/retrigger.json`.

## Loops

Inner loop, Stages 7 to 9:

`action gen -> VLM eval -> signal conversion -> policy update`

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

Outer loop, Stages 10 to 11:

`held-out eval -> threshold gate -> promote or loop back`

Loop-of-loops, Stages 12 to 13 to 1:

`external validation stub -> retrigger manifest -> next run`

## BYO Seams

- `--s3-endpoint`, `--s3-bucket`, `--s3-prefix`
- `--assets-uri`, `--scene-spec-uri`
- `--augment-image`
- `--action-rollouts-uri`, `--train-envs-uri`, `--heldout-envs-uri`
- `--policy-image`
- `--vlm-image`, `--vlm-model`, `--byo-vlm-command`
- `--byo-signal-converter`
- `--trainer-image`, `--byo-trainer-command`
- `--eval-image`, `--byo-eval-command`
- `--threshold`
- `--inner-iterations`, `--outer-iterations`, `--loop-of-loops-iterations`
- `--signal-loss-weight`, `--learning-rate`
- `--no-guardrails`

All defaults are reference components and every seam can be overridden at
runtime in the YAML env vars, SDK keyword arguments, or CLI options.
