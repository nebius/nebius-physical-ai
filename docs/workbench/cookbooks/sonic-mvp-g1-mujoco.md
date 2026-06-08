# SONIC G1 Fine-Tune to MuJoCo MVP

This first milestone fine-tunes Unitree G1 from the released SONIC checkpoint
and evaluates the resulting checkpoint with a headless MuJoCo rollout.
Retargeting, from-scratch training, and the full upstream ONNX/WBC export
contract are not part of this milestone.

## Gate

The pinned SONIC upstream ref used by the Workbench image is
`0a87181c9106d0e49293400714b157676e0ec664`.

Training uses `gear_sonic/train_agent_trl.py` with
`+exp=manager/universal_token/all_modes/sonic_release`. That release config
overrides policy observations to `local_dir_hist` and critic observations to
`privileged_mf_hist`.

The actor policy observation terms are state/proprioceptive:

- `gravity_dir`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `actions`

Camera and render paths are optional. The base env has `render_results: false`,
and `train_agent_trl.py` only enables cameras when `enable_cameras`,
`render_results`, `render_ego`, or `overview_camera` is true. The H100 proof
therefore uses headless state-based training; RT-core rendering is not required.

Combined-image feasibility is positive because `gear_sonic` pins
`numpy==1.26.4` and `gear_sonic[sim]` depends on `mujoco` without forcing
NumPy 2.x. Do not install `decoupled_wbc/sim2mujoco/requirements.txt` for this
image; that file pins `numpy==2.2.6`.

The warm-start checkpoint is `nvidia/GEAR-SONIC:sonic_release/last.pt`, which
the upstream downloader saves as `sonic_release/last.pt`.

## Image

The additive combined runtime is:

```text
npa-sonic-mujoco:0.1.3-mvp
```

It is built from the existing SONIC runtime and adds only MuJoCo/EGL support,
`boto3`, and the checkpoint-to-MuJoCo adapter. The manifest variant is
`sonic-mujoco-h100-mvp`, selected for `h100` and `h200`.

Build and push without overwriting existing tags:

```bash
npa/docker/workbench/sonic/build.sh \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --variant mujoco \
  --tag 0.1.3-mvp \
  --push
```

## Raw YAML

The raw SkyPilot workflow is:

```text
npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml
```

It has two stages:

- `sonic-g1-finetune`: real SONIC training with `SONIC_RUN_REAL_TRAIN=1`,
  `SONIC_TRAIN_MODE=finetune`, `+checkpoint=sonic_release/last.pt`, and tiny
  configurable proof defaults.
- `sonic-mujoco-eval`: downloads
  `training/checkpoints/last.pt`, runs a real MuJoCo rollout, and writes
  `mujoco_eval_metrics.json`, `gpu_device.json`, and `image_pull_proof.json`.

## CLI Submit

Use H100 spot in `eu-north1`; `me-west1` is rejected by the materializer.
For the proven VM path, use docker-payload mode plus registry auth:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml \
  --tool sonic \
  --run-id sonic-mvp-$(date -u +%Y%m%dT%H%M%SZ) \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --gpu-target h100 \
  --region eu-north1 \
  --use-spot \
  --require-controller-up \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --s3-bucket <bucket> \
  --s3-prefix sonic-mvp-proof/<run-id> \
  --var SONIC_PAYLOAD_MODE=docker \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Set `--gpu-target h200` only as a capacity fallback for the same headless
workload.

## SDK

```python
from pathlib import Path

from npa.sdk.workbench import sonic

plan = sonic.materialize_workflow(
    Path("npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml"),
    run_id="sonic-mvp-proof",
    registry="cr.eu-north1.nebius.cloud/<registry-id>",
    gpu_target="h100",
    region="eu-north1",
    use_spot=True,
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_bucket="<bucket>",
    s3_prefix="sonic-mvp-proof/sonic-mvp-proof",
    env_overrides={
        "SONIC_PAYLOAD_MODE": "docker",
        "SONIC_MAX_ITERATIONS": "1",
        "SONIC_MUJOCO_STEPS": "64",
    },
)
```

`sonic.submit_workflow(...)` accepts the same parameters plus
`require_controller_up=True`.
