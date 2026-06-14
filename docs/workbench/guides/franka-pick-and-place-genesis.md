# Pick-and-Place with a Franka Arm in Genesis

**The hook:** spin up thousands of Franka Emika Panda arms in parallel inside
the Genesis physics engine, train one of them to pick up a cube and drop it in a
target zone, then record demonstrations you can turn into a LeRobot dataset —
the same format the famous DROID Franka dataset uses.

This is the classic "hello robot" of manipulation, done at GPU scale.

## Ingredients

- **Robot:** Franka Emika Panda (7-DOF arm + gripper). It's the arm in
  `npa/src/npa/genesis/env_pick_place.py`.
- **Sim / engine:** [Genesis](https://genesis-world.readthedocs.io/) —
  GPU-accelerated parallel physics. Thousands of envs at once.
- **Public dataset:**
  [DROID](https://huggingface.co/docs/lerobot/main/en/porting_datasets_v3) —
  76,000+ real Franka Panda manipulation trajectories. We use it as the
  real-world counterpart to our synthetic demos (a 2 GB `droid_100` sample
  exists for testing).
- **You need:** a GPU with RT/CUDA support — **L40S or better**. Genesis trains
  headless on Nebius. Finish [getting-started](../getting-started.md) first.

## The shape of the workflow

```text
Genesis (Franka cube pick) ──train-teacher──▶ RL teacher (PPO)
        │                                          │
        └──────────── generate-demos ──────────────┘
                              │
                              ▼
                    LeRobotDataset on S3 ──▶ train a student policy
```

A privileged **teacher** learns fast in sim with full state. It then **records
demonstrations** (optionally with domain randomization), which become a standard
`LeRobotDataset` you can train a camera-based **student** on.

## Fast path

**1. See the tool and your GPU.**

```bash
npa workbench genesis list
npa workbench genesis system-info
```

**2. Train the RL teacher** to pick and place the cube (start small to prove the
loop, then scale `--n-envs` / `--max-iterations`):

```bash
npa workbench genesis train-teacher \
  --n-envs 1024 \
  --max-iterations 50 \
  --action-space cartesian \
  --output ./checkpoints/teacher/
```

`--action-space cartesian` gives the policy a 4-D action (delta xyz + gripper)
resolved with inverse kinematics — the most intuitive starting point. Switch to
`joint` for raw 8-D joint control.

**3. Record demonstrations** from the trained teacher, with domain randomization
on so the data is varied:

```bash
npa workbench genesis generate-demos \
  --checkpoint ./checkpoints/teacher/model.pt \
  --n-envs 512 \
  --domain-randomize \
  --output-path ./demos/franka-pick/
```

**4. Check how good the policy is:**

```bash
npa workbench genesis eval-teacher --checkpoint ./checkpoints/teacher/model.pt
```

## Go bigger

- **Full training runs locally or on a workbench VM.** `train-teacher` (and
  `generate-demos` / `eval-teacher`) run on your GPU box, or on a Workbench VM
  when you pass `-p <project> -n <workbench>` (forwarded over SSH). This is where
  real Franka PPO training happens.
- **Validate serverless first.** `train-teacher --runtime serverless --project-id
  <your-project-id> --gpu-type l40s --output-path s3://<bucket>/...` submits a
  Nebius AI Job, but the serverless Genesis path is a **smoke** (it checks the
  Genesis import and writes a placeholder checkpoint, verified end to end). Use
  it to prove credentials, image pull, and S3 output before committing GPU time;
  inspect what it wrote with
  `npa workbench data list --input-path s3://<bucket>/.../`. (Serverless needs
  `--project-id`, or a project configured in `~/.npa/config.yaml`.)
- **Scale up:** the defaults are `--n-envs 4096 --max-iterations 500`. More envs
  and iterations give a stronger teacher.
- **Tune rewards** without editing code via repeatable overrides, e.g.
  `--env-override approach_scale=2.0 --env-override domain_randomize=true`.
- **Train a student policy** on the recorded demos with
  [LeRobot](reachy2-lerobot-policy.md) — the demos are already in LeRobot format.

## Use the real Franka data too

Your synthetic demos share the `LeRobotDataset` format with **DROID**, the
large-scale real-world Franka dataset. That means you can mix or compare sim and
real:

- Grab the 2 GB `droid_100` sample to experiment, or port the full set to
  LeRobot v3.0 (see the
  [LeRobot porting guide](https://huggingface.co/docs/lerobot/main/en/porting_datasets_v3)).
- Point any LeRobot training run at the resulting S3 URI exactly like you would
  your Genesis demos.

## Heads up

- Genesis **training** works headless on Nebius. **Visual demo rendering at
  scale** is currently limited by EGL/DRI device access in containers; 480x640
  targeted renders work via the Mesa fallback. See
  `.agents/skills/workbench/genesis/SKILL.md` for the current state.

## Dig deeper

- Env source: `npa/src/npa/genesis/env_pick_place.py`
- Commands: `npa workbench genesis train-teacher | generate-demos | eval-teacher | eval-student | diagnose | tune`
- Skill: `.agents/skills/workbench/genesis/SKILL.md`
