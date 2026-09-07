---
name: lerobot
description: Use when working on LeRobot workbench training, evaluation, serving, inference, dataset conversion, or robot policy workflows.
---

# LeRobot

LeRobot is the default robot policy training framework. It supports ACT, Diffusion Policy, and SmolVLA (and additional VLAs / world models in 0.6.0).

Use it as the data standard and policy interface layer, not as a managed-service competitor to Hugging Face.

## Supported versions

| Version | Role | Image tag | Notes |
| --- | --- | --- | --- |
| **0.5.1** | **Default** | `npa-lerobot:cuda13-b300-0.5.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Accepted public default; the plain `0.5.1` alias is historical |
| **0.6.0** | Additional package support | Operator-built image required | No accepted public image pin/digest; lean extras (`training,evaluation,pusht,libero,diffusion,smolvla`); `--eval_freq` → `--env_eval_freq` |

Select the package with `--lerobot-version`. For serverless training on 0.6.0,
supply a validated operator image through `train --image`; version selection
alone does not publish an image. VM deployment installs the selected package
and has no `--image` option. The anonymous 2026-09-05 audit returned
`404 MANIFEST_UNKNOWN` for the official `npa-lerobot:0.6.0` tag. It is outside
the current public release plan and must not be treated as an accepted release.

Canonical manifest: `npa/src/npa/deploy/lerobot_version_manifest.json`.

Upstream release notes: https://huggingface.co/blog/lerobot-release-v060

## Interfaces

API:

- `POST /train`
- `POST /eval`
- `POST /serve`
- `POST /infer`
- `GET /list-checkpoints`

CLI:

```bash
npa workbench lerobot deploy
npa workbench lerobot deploy --runtime vm --lerobot-version 0.6.0
npa workbench lerobot train
npa workbench lerobot train --runtime serverless --lerobot-version 0.6.0 --image '<validated-operator-image>@sha256:<digest>' ...
npa workbench lerobot eval
npa workbench lerobot serve
npa workbench lerobot infer
npa workbench lerobot list-checkpoints
```

Build operator/BYOF image variants (official publication has separate gates):

```bash
npa/docker/workbench/lerobot/build.sh --registry '<operator-registry>' --all-versions
# or
npa/docker/workbench/lerobot/build.sh --registry '<operator-registry>' --version 0.6.0
```

The datacenter-Blackwell variant is `npa/docker/workbench/lerobot/Dockerfile.b300`.
LeRobot 0.5.1 requires Python 3.12, so this variant adds a dedicated
`/opt/lerobot/venv` with torch 2.9/cu130 and the matching torchcodec 0.8 line;
it does not reuse npa-base's Python 3.11 venv for the policy workload. The base
venv remains available at `/opt/npa/venv` for the baked architecture and kernel
validators. Keep both checks in hardware validation: validate the inherited
CUDA base, then run a real ACT training step from the LeRobot venv.

## Data Contract

Input format is `LeRobotDataset` in Hugging Face format. Use the SimToLeRobot adapter to convert Genesis or other simulation outputs.

Output is a policy checkpoint on S3.

## Validation

- 9/9 E2E serverless tests pass on Nebius (default 0.5.1 image).
- Tier 1 validated on B300.
- 0.6.0: build `npa-lerobot:0.6.0` and run the same golden env/functional smokes with `NPA_LEROBOT_VERSION=0.6.0`.
