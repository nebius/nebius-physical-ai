---
name: lerobot
description: Use when working on LeRobot workbench training, evaluation, serving, inference, dataset conversion, or robot policy workflows.
---

# LeRobot

LeRobot is the default robot policy training framework. It supports ACT, Diffusion Policy, and SmolVLA.

Use it as the data standard and policy interface layer, not as a managed-service competitor to Hugging Face.

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
npa workbench lerobot train
npa workbench lerobot eval
npa workbench lerobot serve
npa workbench lerobot infer
npa workbench lerobot list-checkpoints
```

## Data Contract

Input format is `LeRobotDataset` in Hugging Face format. Use the SimToLeRobot adapter to convert Genesis or other simulation outputs.

Output is a policy checkpoint on S3.

## Validation

- 9/9 E2E serverless tests pass on Nebius.
- Tier 1 validated on B300.
