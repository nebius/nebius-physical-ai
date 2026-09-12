# Train a Reachy 2 Humanoid Policy from a Public Dataset

[Guides](README.md)

Train a LeRobot imitation policy from a Reachy 2 dataset on Nebius, then
inspect its checkpoint. Choose a real Hub dataset with observation and action
schemas supported by your policy; the dataset ID below is a placeholder.

## Ingredients

- **Robot:** [Reachy 2](https://huggingface.co/docs/lerobot/main/en/reachy2)
  (Pollen Robotics). Fully supported by LeRobot for teleop, training, and eval.
- **Sim / engine:** LeRobot training and task-compatible evaluation. The
  [Sim2Real workflow](sim2real-workflow.md) has additional simulator and data requirements.
- **Public dataset:** Reachy datasets published on the Hub under
  [`pollen-robotics`](https://huggingface.co/pollen-robotics) and
  [`lerobot`](https://huggingface.co/lerobot). Use a `LeRobotDataset` whose
  robot, camera, and action schemas match your training configuration.
- **You need:** Nebius creds + a GPU (see [getting-started](../getting-started.md)).

## The shape of the workflow

```text
public Reachy dataset (HF Hub or S3) ──▶ npa workbench lerobot train
                                                  │
                                          policy checkpoint on S3
                                                  │
                              ┌───────────────────┼───────────────────┐
                              ▼                    ▼                   ▼
                        lerobot eval         lerobot serve       lerobot infer
```

LeRobot is the default policy framework and the data standard. It ships ACT,
Diffusion Policy, and SmolVLA — start with `act` for a quick single-task
baseline.

## Fast path

These examples assume a configured LeRobot workbench. Use the
[runtime guide](../runtime-modes.md) to deploy one; select an existing workbench
with `-p <project-alias> -n <workbench-name>` before `train`.

**1. See the tool:**

```bash
npa workbench lerobot list
```

**2. Train an ACT policy** straight from a Hub dataset. Swap `--dataset` for the
Reachy dataset you want (a Pollen Robotics / LeRobot Hub repo ID):

```bash
npa workbench lerobot train \
  --policy-type act \
  --dataset pollen-robotics/<reachy-dataset> \
  --job-name reachy-act-hello \
  --steps 2000 \
  --batch-size 8 \
  --output-path s3://your-bucket-name/checkpoints/reachy-act/
```

Prefer to stage data in your own bucket first? Use `--input-path` with an S3
`LeRobotDataset` URI instead of `--dataset` (it takes priority):

```bash
npa workbench lerobot train \
  --policy-type act \
  --input-path s3://your-bucket-name/datasets/reachy2-grasp/ \
  --job-name reachy-act-byo \
  --output-path s3://your-bucket-name/checkpoints/reachy-act/
```

**3. Evaluate and serve the trained checkpoint:**

```bash
npa workbench lerobot eval --help
npa workbench lerobot serve --help
npa workbench lerobot infer --help
npa workbench lerobot list-checkpoints
```

## Go bigger

- **Run serverless on Nebius:** add `--runtime serverless --project-id
  <your-project-id> --gpu-type h200` (or `b300` / `l40s`) and `train` submits a
  Nebius AI Job. (Serverless needs `--project-id`, or a project configured in
  `~/.npa/config.yaml`.)
- **Try a stronger policy:** `--policy-type smolvla` for a language-conditioned
  VLA baseline, or `diffusion` for smooth continuous actions.
- **Build a full pipeline:** adapt the [Sim2Real workflow](sim2real-workflow.md)
  after checking its [customer input contracts](sim2real-customer-assets.md).
- **Benchmark the GPUs:** see the
  [LeRobot GPU benchmarks](../cookbooks/lerobot-gpu-benchmarks.md) across L40S,
  H200, B300, and RTX PRO 6000.

## Record your own Reachy data

Got a real Reachy 2? LeRobot records directly to the same format with
`lerobot-record --robot.type=reachy2 ...`, then you push to the Hub or stage to
S3 and train with the commands above. Training still requires matching
observation, action, camera, and robot schemas, regardless of how the data was
collected.

## Dig deeper

- LeRobot CLI: `npa workbench lerobot train | eval | serve | infer | list-checkpoints | benchmark`
- Reachy 2 in LeRobot: https://huggingface.co/docs/lerobot/main/en/reachy2
- Skill: `skills/tools/lerobot/SKILL.md`
