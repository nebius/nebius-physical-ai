# SONIC G1 MuJoCo Image Status

The historical `sonic-mujoco-h100-mvp` / `npa-sonic-mujoco:0.1.3-mvp`
variant is quarantined. It inherits `npa-sonic:0.1.2`, whose built bytes include
an old `nvcr.io/nvidia/isaac-lab` base and baked NVIDIA driver libraries.

Do not submit, mirror, or publish that variant. The resolver intentionally
rejects `h100`, `h200`, `mujoco`, and the explicit legacy variant ID. Supplying
credentials or EULA acceptance at runtime cannot change the licensing of bytes
already baked into the image.

A replacement must:

1. Build from the active runtime-fetch SONIC base without NVIDIA Isaac or driver
   payloads.
2. Add only redistributable MuJoCo/EGL dependencies.
3. Pass the built-byte Omniverse payload scan and no-baked-consent checks.
4. Record a new additive tag and immutable digest in
   `npa/src/npa/deploy/sonic_image_manifest.json`.
5. Pass real GPU training/evaluation validation before its status becomes
   `active`.

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
  --registry <your-registry>/<namespace> \
  --variant mujoco \
  --tag 0.1.3-mvp \
  --push
```

## Raw YAML

The raw SkyPilot workflow is:

```text
npa/workflows/workbench/npa-workflows/sonic-locomotion-finetuning.yaml
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
  npa/workflows/workbench/npa-workflows/sonic-locomotion-finetuning.yaml \
  --tool sonic \
  --run-id sonic-mvp-$(date -u +%Y%m%dT%H%M%SZ) \
  --registry <your-registry>/<namespace> \
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
    Path("npa/workflows/workbench/npa-workflows/sonic-locomotion-finetuning.yaml"),
    run_id="sonic-mvp-proof",
    registry="<your-registry>/<namespace>",
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
