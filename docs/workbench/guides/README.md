# Easy Guides

Short, friendly, copy-paste guides for getting a robot doing something
interesting on Nebius Physical AI. Each one picks a **robot**, a **simulation
environment**, and a **cool public dataset**, then walks you from zero to a
result.

New here? Start with the no-GPU guide — it runs on your laptop with no cloud,
no GPU, and no credentials. Then pick a robot and have fun.

| Guide | Robot | Sim / engine | Public dataset | Needs a GPU? |
| --- | --- | --- | --- | --- |
| [Score a robot in 60 seconds](score-a-robot-no-gpu.md) | any | offline | shipped sample rollouts | No |
| [Pick-and-place with a Franka arm](franka-pick-and-place-genesis.md) | Franka Emika Panda | Genesis | DROID (Franka) | Yes (L40S+) |
| [Teach a robot to push a T](pusht-sim-to-real.md) | sim pusher | sim-to-real loop | `lerobot/pusht` | Yes (H100) — local smoke is free |
| [Train a Reachy 2 humanoid policy](reachy2-lerobot-policy.md) | Reachy 2 | LeRobot | Pollen Robotics / LeRobot Hub | Yes |
| [Make a Unitree G1 walk](g1-humanoid-walk-sonic.md) | Unitree G1 | MuJoCo | NVIDIA GEAR-SONIC checkpoint | Yes (H100) |
| [Train a quadruped to run](quadruped-isaac-lab.md) | ANYmal / quadruped | Isaac Lab | Isaac Lab built-in tasks | Yes (RT-core: L40S / RTX PRO 6000) |

## How these guides work

Every guide follows the same shape so you always know where you are:

- **The hook** — what you'll build and why it's fun.
- **Ingredients** — robot, sim, dataset, and what you need installed.
- **Fast path** — the shortest command that produces a result.
- **Go bigger** — turn the toy run into a real GPU run.
- **Look at it** — visualize the result (Rerun, FiftyOne, reports).
- **Dig deeper** — links to the full cookbook and the skill behind it.

## Before you start

Install `npa` once (Python 3.10+). The virtual environment can live anywhere:

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa

npa --version
```

The no-GPU guide needs nothing else. The GPU guides assume you have completed
[../../quickstart.md](../../quickstart.md) and
[../getting-started.md](../getting-started.md) (Nebius auth, an S3 bucket, and
`npa configure`). Each guide calls out exactly when credentials are required.

## Bring your own everything

These guides use public datasets and the shipped robots so you can reproduce
them, but the workbench is built to be swapped:

- **Bring your own dataset** — point any guide at an S3 `LeRobotDataset` URI.
- **Bring your own policy image** — swap the container, keep the contract.
- **Bring your own robot** — Franka, Reachy 2, Unitree G1, quadrupeds, and more
  are all just configs over the same train / eval / serve / infer commands.

When you're ready for the production recipes, head to the
[cookbooks](../cookbooks/README.md).
