# Train a Reachy 2 Humanoid Policy from a Public Dataset

**The hook:** Reachy 2 is the open-source humanoid from Pollen Robotics (now
part of Hugging Face) — a friendly upper-body robot with two arms, a head, and
a mobile base. In this guide you take a **public Reachy dataset off the Hugging
Face Hub** and train an imitation policy on it with LeRobot, no robot hardware
required.

It's the fastest way to feel what teaching a humanoid to manipulate is like.

## Ingredients

- **Robot:** [Reachy 2](https://huggingface.co/docs/lerobot/main/en/reachy2)
  (Pollen Robotics). Fully supported by LeRobot for teleop, training, and eval.
- **Sim / engine:** LeRobot training + evaluation; pair it with the
  [sim-to-real loop](pusht-sim-to-real.md) for closed-loop rollouts.
- **Public dataset:** Reachy datasets published on the Hub under
  [`pollen-robotics`](https://huggingface.co/pollen-robotics) and
  [`lerobot`](https://huggingface.co/lerobot). Any `LeRobotDataset` works — the
  examples below show the swap.
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

- **Run serverless on Nebius:** add `--runtime serverless --gpu-type h200`
  (or `b300` / `l40s`) and `train` submits a Nebius AI Job.
- **Try a stronger policy:** `--policy-type smolvla` for a language-conditioned
  VLA baseline, or `diffusion` for smooth continuous actions.
- **Close the loop:** feed your checkpoint into the
  [sim-to-real loop](pusht-sim-to-real.md) to evaluate rollouts and collect
  feedback, then visualize with Rerun.
- **Benchmark the GPUs:** see the
  [LeRobot GPU benchmarks](../cookbooks/lerobot-gpu-benchmarks.md) across L40S,
  H200, B300, and RTX PRO 6000.

## Record your own Reachy data

Got a real Reachy 2? LeRobot records directly to the same format with
`lerobot-record --robot.type=reachy2 ...`, then you push to the Hub or stage to
S3 and train with the exact commands above. The workbench never cares whether
the data came from a real robot, a teleop session, or a simulator.

## Dig deeper

- LeRobot CLI: `npa workbench lerobot train | eval | serve | infer | list-checkpoints | benchmark`
- Reachy 2 in LeRobot: https://huggingface.co/docs/lerobot/main/en/reachy2
- Skill: `.agents/skills/workbench/lerobot/SKILL.md`
