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
- **You need:** a compatible CUDA GPU for headless PPO training; a full run on
  **one H200** is verified. H200 has no RT cores: visual rendering has separate
  requirements and the container limitations below still apply. Finish
  [getting-started](../getting-started.md) first.

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
  when you pass `-p <project> -n <workbench>` (forwarded over SSH).
- **Serverless runs real PPO training.** `train-teacher --runtime serverless`
  submits the same training implementation as a Nebius AI Job and uploads its
  checkpoint and summaries to `--output-path`. It needs `--project-id`, or a
  project configured in `~/.npa/config.yaml`.
- **Scale up:** the defaults are `--n-envs 4096 --max-iterations 500`. More envs
  and iterations extend training; evaluate the resulting checkpoint to measure
  policy quality.
- **Tune rewards** without editing code via repeatable overrides, e.g.
  `--env-override approach_scale=2.0 --env-override domain_randomize=true`.
- **Train a student policy** on the recorded demos with
  [LeRobot](reachy2-lerobot-policy.md) — the demos are already in LeRobot format.

A serverless H200 training example:

```bash
npa workbench genesis train-teacher \
  --runtime serverless --project-id <your-project-id> \
  --gpu-type h200 --gpu-count 1 \
  --job-name <your-training-job-name> \
  --n-envs 1024 --max-iterations 500 --action-space cartesian \
  --output-path s3://<your-bucket>/<new-training-prefix>/
```

Inspect `model.pt`, `arch_config.json`, `train_teacher_summary.json`, and
`npa_genesis_checkpoint_manifest.json` under the selected output prefix. Keep
`model.pt` and `arch_config.json` together when downloading the checkpoint for
evaluation or demonstration generation. A verified run completed 500 PPO
iterations across 1,024 environments (12,288,000 transitions) and produced the
real checkpoint. A separate evaluation of that checkpoint on 1,024 environments
with held-out seed 7777 produced **zero successful episodes**. The components
completed; this checkpoint does not solve pick-and-place. Evaluate quality before
using a checkpoint to generate successful demonstrations.

**Cold-start recovery:** `PROVISIONING`, `STARTING`, and `IMAGE_PULLING` remain
active startup states. A create response timeout triggers lookup of the existing
job. Genesis records the launch before submitting, then saves the exact provider
ID; transient observation failures keep polling that ID without creating another
job. Unknown phases and authentication failures remain explicit errors.

After a disconnect, repeat the command with the same job name, image, output
prefix, and NPA configuration directory (`NPA_CONFIG_DIR`). Completed jobs reuse
their verified declared outputs. The owner-only local submission journal lives
under `runtime/serverless-submissions/` in that directory; subsequent supervisor
decisions also persist in S3 under the run output prefix. If a create response
and lookup are both inconclusive, the journal blocks another create even when
the job is temporarily invisible. Preserve that journal until the provider
identity is reconciled. Local journal recovery requires the original
configuration directory; it does not coordinate independent operator machines.
An existing job name must also match the saved command, immutable image, output
prefix, and provider ID. A changed request or a legacy job without its original
journal is refused without another create; its status, logs and artifacts remain
available for inspection. `--submit-only` also pins the image before launch so a
later supervised reconnect uses the same image identity.

Before a new submission, Genesis verifies one configured project, tenant and
region, the selected GPU platform/preset, and write/read access to the exact
output prefix using the credentials sent to the workload. Fix failed preflight
settings before launching. A low evaluation score still describes policy quality,
not infrastructure failure.

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
  `skills/tools/genesis/SKILL.md` for the current state.

## Dig deeper

- Env source: `npa/src/npa/genesis/env_pick_place.py`
- Commands: `npa workbench genesis train-teacher | generate-demos | eval-teacher | eval-student | diagnose | tune`
- Skill: `skills/tools/genesis/SKILL.md`
