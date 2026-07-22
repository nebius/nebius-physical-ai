# Make a Unitree G1 Humanoid Walk (SONIC + MuJoCo)

**The hook:** take NVIDIA's released **GEAR-SONIC** locomotion checkpoint, warm-
start a fine-tune for the **Unitree G1** humanoid, then watch it walk in a
headless **MuJoCo** rollout — all as one SkyPilot workflow on an H100. This is
whole-body humanoid control, and you don't have to train from scratch.

## Ingredients

- **Robot:** [Unitree G1](https://www.unitree.com/g1) humanoid (legs + whole
  body).
- **Sim / engine:** [MuJoCo](https://mujoco.org/) for the headless eval rollout;
  SONIC's training stack for fine-tuning.
- **Public checkpoint:** NVIDIA
  [`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC) —
  `sonic_release/last.pt` is the warm-start.
- **You need:** Nebius creds, a container registry, an S3 bucket, and **H100**
  capacity (`eu-north1`). SONIC training is state-based and headless, so no RT
  cores are required for this path.

## The shape of the workflow

```text
GEAR-SONIC checkpoint ──finetune (G1, headless)──▶ trained checkpoint
                                                          │
                                                  MuJoCo rollout eval
                                                          │
                                              mujoco_eval_metrics.json
```

One SkyPilot YAML runs both stages: `sonic-g1-finetune` then
`sonic-mujoco-eval`. The fine-tune uses tiny proof defaults so you can prove the
path before scaling iterations.

## Fast path

Meet the tool and check routing first:

```bash
npa workbench sonic list
npa workbench sonic status
```

Submit the fine-tune + MuJoCo-eval workflow on H100 spot (docker-payload mode):

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml \
  --tool sonic \
  --run-id sonic-g1-$(date -u +%Y%m%dT%H%M%SZ) \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --gpu-target h100 \
  --region eu-north1 \
  --use-spot \
  --require-controller-up \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --s3-bucket <bucket> \
  --s3-prefix sonic-g1/<run-id> \
  --var SONIC_PAYLOAD_MODE=docker \
  --var SONIC_MAX_ITERATIONS=1 \
  --var SONIC_MUJOCO_STEPS=64 \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

When it finishes you'll have `mujoco_eval_metrics.json`, `gpu_device.json`, and
`image_pull_proof.json` in S3 — proof the G1 checkpoint ran a real MuJoCo
rollout.

Prefer Python? The SDK runs the same materialize/submit path:

```python
from pathlib import Path
from npa.sdk.workbench import sonic

plan = sonic.materialize_workflow(
    Path("npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml"),
    run_id="sonic-g1-proof",
    registry="cr.eu-north1.nebius.cloud/<registry-id>",
    gpu_target="h100",
    region="eu-north1",
    use_spot=True,
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_bucket="<bucket>",
    s3_prefix="sonic-g1/sonic-g1-proof",
    env_overrides={
        "SONIC_PAYLOAD_MODE": "docker",
        "SONIC_MAX_ITERATIONS": "1",
        "SONIC_MUJOCO_STEPS": "64",
    },
)
```

## Go bigger

- **Train longer:** raise `SONIC_MAX_ITERATIONS` and `SONIC_MUJOCO_STEPS` once
  the proof run is green.
- **Export to ONNX and re-evaluate:** `npa workbench sonic export` produces a
  deterministic-action ONNX graph, and `npa workbench sonic eval` scores it. See
  the [SONIC Export and Eval Runbook](../cookbooks/sonic-eval-runbook.md).
- **Score with MJLab:** the
  [SONIC Locomotion Fine-Tuning cookbook](../cookbooks/sonic-locomotion-finetuning.md)
  adds motion retargeting and MJLab evaluation.

## Heads up

- Use `--gpu-target h200` only as a capacity fallback for the same headless
  workload. `me-west1` is rejected; use `eu-north1`.
- SONIC **render** validation (the Isaac render path) needs RT cores; the walk-
  in-MuJoCo path in this guide does not.

## Dig deeper

- Cookbook: [SONIC G1 Fine-Tune to MuJoCo MVP](../cookbooks/sonic-mvp-g1-mujoco.md)
- Workflow YAML: `npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml`
- Skill: `skills/tools/sonic/SKILL.md`
