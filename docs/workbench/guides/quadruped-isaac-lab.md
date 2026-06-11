# Train a Quadruped to Run in Isaac Lab

**The hook:** drop a four-legged robot into NVIDIA **Isaac Lab**, run massively
parallel reinforcement learning, and watch it learn to trot across flat and
rough terrain. Isaac Lab ships dozens of ready-made tasks, so you get a real
locomotion policy without writing an environment from scratch.

## Ingredients

- **Robot:** a quadruped — ANYmal-C is the built-in favorite (swap the task to
  pick another).
- **Sim / engine:** [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) on Isaac
  Sim — GPU RL with thousands of parallel envs.
- **Public tasks / data:** Isaac Lab's built-in task registry, e.g.
  `Isaac-Velocity-Flat-Anymal-C-v0` and `Isaac-Velocity-Rough-Anymal-C-v0`.
- **You need:** Nebius creds + an **RT-core GPU — L40S or RTX PRO 6000**. Isaac
  Lab will **not** run correctly on H100/H200 (no RT cores). Training must run
  headless. See [getting-started](../getting-started.md).

## The shape of the workflow

```text
Isaac Lab task (quadruped) ──train (headless RL)──▶ policy checkpoint
                                                          │
                                                    isaac-lab eval
```

## Fast path

**1. Meet the tool and confirm your GPU is RT-capable:**

```bash
npa workbench isaac-lab list
npa workbench isaac-lab system-info
```

**2. Train a quadruped velocity policy** (start with a small env count / step
budget to prove the loop):

```bash
npa workbench isaac-lab train \
  --task Isaac-Velocity-Flat-Anymal-C-v0 \
  --num-envs 1024 \
  --steps 200 \
  --output-path s3://your-bucket-name/isaac-lab/anymal-flat/
```

**3. Make it harder** — swap to rough terrain once flat-ground walking works:

```bash
npa workbench isaac-lab train \
  --task Isaac-Velocity-Rough-Anymal-C-v0 \
  --num-envs 2048 \
  --steps 500 \
  --output-path s3://your-bucket-name/isaac-lab/anymal-rough/
```

**4. Evaluate the trained policy:**

```bash
npa workbench isaac-lab eval \
  --task Isaac-Velocity-Flat-Anymal-C-v0 \
  --help
```

## Go bigger

- **Serverless is not the validated path yet.** `--runtime serverless
  --project-id <your-project-id> --gpu-type l40s` does submit a Nebius AI Job,
  but Isaac Lab end-to-end serverless training is currently failing (tracked by
  `W9-isaac-lab-e2e-fix`) and RT-core (L40S) serverless capacity is scarce, so
  jobs often queue and then fail without artifacts. Prefer the VM / managed-
  Kubernetes RT-core path and the BYOF flow below until that lands.
- **Try other robots:** the same `--task` flag covers arms (e.g.
  `Isaac-Reach-Franka-v0`), humanoids, and more — Isaac Lab's whole task
  registry is available.
- **Bring your own fork:** layer a custom Isaac Lab image over the digest-pinned
  base with `--image`. See the
  [Isaac Lab BYOF cookbook](../cookbooks/byof-isaac-lab/README.md).
- **Export to LeRobot:** roll out a humanoid/G1 task to a `LeRobotDataset` with
  `npa workbench isaac-lab export-lerobot`, then train a policy with the
  [LeRobot guide](reachy2-lerobot-policy.md).

## Heads up

- **RT cores are mandatory.** L40S or RTX PRO 6000 only. H100/H200 lack RT cores
  and won't render/simulate Isaac Lab correctly.
- Training must invoke **headless** mode — these commands do.

## Dig deeper

- Cookbook: [Isaac Lab BYOF](../cookbooks/byof-isaac-lab/README.md)
- Workflows: `npa/workflows/workbench/skypilot/isaac-lab-rl-train.yaml`,
  `isaac-lab-rl-sweep.yaml`; runner `npa/scripts/run_isaac_lab_rl.py`
- Skill: `.agents/skills/workbench/isaac-lab/SKILL.md`
