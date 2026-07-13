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
| **0.5.1** | **Default** | `npa-lerobot:0.5.1` | Current golden-eval pin; keep for GR00T N1.5 / sim2real policy image parity |
| **0.6.0** | Additional | `npa-lerobot:0.6.0` | Lean extras (`training,evaluation,pusht,libero`); `--eval_freq` → `--env_eval_freq` |

Select with `--lerobot-version` on deploy / serverless train, or override the image with `--image .../npa-lerobot:0.6.0`.

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
npa workbench lerobot deploy --lerobot-version 0.6.0
npa workbench lerobot train
npa workbench lerobot train --runtime serverless --lerobot-version 0.6.0 ...
npa workbench lerobot eval
npa workbench lerobot serve
npa workbench lerobot infer
npa workbench lerobot list-checkpoints
```

Build both image tags:

```bash
npa/docker/workbench/lerobot/build.sh --all-versions
# or
npa/docker/workbench/lerobot/build.sh --version 0.6.0
```

## Data Contract

Input format is `LeRobotDataset` in Hugging Face format. Use the SimToLeRobot adapter to convert Genesis or other simulation outputs.

Output is a policy checkpoint on S3.

## Validation

- 9/9 E2E serverless tests pass on Nebius (default 0.5.1 image).
- Tier 1 validated on B300.
- 0.6.0: build `npa-lerobot:0.6.0` and run the same golden env/functional smokes with `NPA_LEROBOT_VERSION=0.6.0`.
