# Easy Guides

Short, friendly, copy-paste guides for getting a robot doing something
interesting on Nebius Physical AI. Each one picks a **robot**, a **simulation
environment**, and a **cool public dataset**, then walks you from zero to a
result.

New here? Pick whichever robot sounds like the most fun — the guides are
independent, and each one ends with something you can look at.

| Guide | Robot | Sim / engine | Public dataset | GPU |
| --- | --- | --- | --- | --- |
| [Pick-and-place with a Franka arm](franka-pick-and-place-genesis.md) | Franka Emika Panda | Genesis | DROID (Franka) | L40S+ |
| [Teach a robot to push a T](pusht-sim-to-real.md) | sim pusher | sim-to-real loop | `lerobot/pusht` | H100 |
| [Train a Reachy 2 humanoid policy](reachy2-lerobot-policy.md) | Reachy 2 | LeRobot | Pollen Robotics / LeRobot Hub | yes |
| [Make a Unitree G1 walk](g1-humanoid-walk-sonic.md) | Unitree G1 | MuJoCo | NVIDIA GEAR-SONIC checkpoint | H100 |
| [Train a quadruped to run](quadruped-isaac-lab.md) | ANYmal / quadruped | Isaac Lab | Isaac Lab built-in tasks | RT-core: L40S / RTX PRO 6000 |
| [Turn a photo capture into a 3D scene](neural-reconstruction.md) | n/a (scene capture) | NVIDIA NuRec / NRE | `nvidia/PhysicalAI-NuRec-PPISP` | RT-core: RTX PRO 6000 / L40S |

## How these guides work

Every guide follows the same shape so you always know where you are:

- **The hook** — what you'll build and why it's fun.
- **Ingredients** — robot, sim, dataset, and what you need installed.
- **Fast path** — the shortest command that produces a result.
- **Go bigger** — scale the fast path into a larger GPU run.
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

The guides assume you have completed
[../../quickstart.md](../../quickstart.md) and
[../getting-started.md](../getting-started.md) (Nebius auth, an S3 bucket, and
`npa configure`). Each guide calls out exactly when credentials are required.

## What's been validated on real backends

These guides were exercised against live Nebius (via `npa`), not just read:

| Path | Backend | Result |
| --- | --- | --- |
| `vlm-eval benchmark/run` (stub) | local, offline | works (`accuracy: 1.0`) |
| `lerobot train --runtime serverless --smoke` | Nebius AI Job (H200) | works — produced a real ACT checkpoint (`model.safetensors`) in S3 |
| `genesis train-teacher --runtime serverless` | Nebius AI Job (H100) | works, but is a **smoke** (import check + placeholder checkpoint); real Genesis training is local/VM |
| `sim_to_real.local_smoke` | local, no cluster | runs the spine; reports `blocked` unless `lerobot` is installed locally |
| `isaac-lab train --runtime serverless` | Nebius AI Job (`gpu-l40s-a`) | **capacity-blocked** — `NotEnoughResources` / VM schedule timeout |
| `isaac-lab train --runtime serverless` | Nebius AI Job (`gpu-l40s-d`) | job schedules and completes; minimal run produced no artifact yet (small step budget / `W9-isaac-lab-e2e-fix`) |
| `lerobot` / `fiftyone` deploy `--preemptible --dry-run` | Nebius Terraform VM path | CLI + dry-run OK; full apply needs IAM bootstrap on your project |
| Preemptible VM flags and resume | — | [preemptible-vms.md](../preemptible-vms.md) |

Isaac Lab needs RT cores, and serverless RT-core capacity varies by SKU: the
default `gpu-l40s-a` pool failed to schedule, while `gpu-l40s-d` had capacity and
ran to completion. `gpu-rtx6000` is **not** a serverless platform (use the
managed-Kubernetes path). For real Isaac Lab training prefer an RT-core VM /
managed-K8s + BYOF; for a serverless capacity retry use `--gpu-type gpu-l40s-d`.

SONIC G1 (MuJoCo) is documented from its cookbook and not yet re-run here.

## Bring your own everything

These guides use public datasets and the shipped robots so you can reproduce
them, but the workbench is built to be swapped:

- **Bring your own dataset** — point any guide at an S3 `LeRobotDataset` URI.
- **Bring your own policy image** — swap the container, keep the contract.
- **Bring your own robot** — Franka, Reachy 2, Unitree G1, quadrupeds, and more
  are all just configs over the same train / eval / serve / infer commands.

When you're ready for the production recipes, head to the
[cookbooks](../cookbooks/README.md).

## Physical AI Data Factory (video data augmentation)

Turn a handful of frames into a labeled, curated, multiplied dataset: annotate →
Cosmos Transfer augment → evaluate/validate → re-label → FiftyOne curate → Rerun
visualize. Runs on Nebius + SkyPilot (no OSMO); pure composition of workbench
tools.

| Doc | Use when |
| --- | --- |
| [physical-ai-data-factory-deploy.md](physical-ai-data-factory-deploy.md) | **Copy-paste runbook** — from zero to a running blueprint (includes a one-block Quick start that stages input frames and submits) |
| [physical-ai-data-factory.md](physical-ai-data-factory.md) | Conceptual guide — blueprint→stage mapping, S3 layout, viewing results |

> **Fastest start:** the deploy runbook's [Quick start](physical-ai-data-factory-deploy.md#quick-start-copy-paste)
> seeds captionable frames (no dataset needed) and submits an input-conditioned
> Cosmos run in a single block. The GPU runner turns those same frames into a
> temporary clip; no upstream example media is packaged or required.

## Sim-to-real (14-stage production loop)

These guides are separate from the easy PushT walkthrough above. They document the
full VLM→RL loop, data contracts, and cluster operations:

| Doc | Use when |
| --- | --- |
| [sim2real-workflow.md](sim2real-workflow.md) | Run the loop (quickstart, CLI) |
| [sim2real-data-contracts.md](sim2real-data-contracts.md) | **Canonical** formats, schemas, S3 layout |
| [sim2real-customer-assets.md](sim2real-customer-assets.md) | Customer uploads, scorecard |
| [sim2real-architecture.md](sim2real-architecture.md) | Standard-runtime graph, loops, parallel waves, resume |
| [sim2real-demo-script-10min.md](sim2real-demo-script-10min.md) | Presentation walkthrough |
