---
name: isaac-lab
description: Use when working on Isaac Lab RL simulation, deployment, SkyPilot workflows, or customer custom-fork support.
---

# Isaac Lab

Isaac Lab is the RL simulation framework. It requires RT cores: use L40S or RTX Pro 6000 only. It will not run correctly on H100 or H200 because those GPUs do not provide RT cores.

Training must invoke headless mode. Verify training commands do not trigger rendering paths.

## Interfaces

API:

- `POST /train`
- `POST /eval`
- `GET /status`
- `GET /system-info`
- `GET /list`

CLI:

```bash
npa workbench isaac-lab deploy
npa workbench isaac-lab train
npa workbench isaac-lab eval
npa workbench isaac-lab status
npa workbench isaac-lab system-info
npa workbench isaac-lab list
```

## Custom Forks

Customers can bring their own Isaac Lab fork through an `image_id` override in the SkyPilot YAML. The workbench provides a validated base container; the customer layers their fork on top.

The replacement image must preserve the expected Isaac Lab entry point or runner contract.

Cookbook: `docs/cookbooks/byof-isaac-lab/README.md`.

Validated BYOF surfaces:

- image override through `npa/scripts/run_isaac_lab_rl.py --image`
  (`w10-byof-image-only-20260520T232650Z`);
- command override through a SkyPilot YAML `run:` block variant passed with
  `--yaml`, invoking `/opt/byof/custom_train.py`
  (`w10-byof-image-and-cmd-20260520T233113Z`).

The runner exposes `--image` directly. It does not expose a `--run-cmd` flag, so
custom entrypoints should use a customer-owned YAML variant that preserves the
runtime contract, checkpoint discovery, manifest creation, and S3 upload block.

## Workflows

- Single RL job: `npa/workflows/skypilot/isaac-lab-rl-train.yaml`.
- Parameter sweep: `npa/workflows/skypilot/isaac-lab-rl-sweep.yaml`.
- Runner: `npa/scripts/run_isaac_lab_rl.py`.

E2E is pending the training command fix tracked by `W9-isaac-lab-e2e-fix`.
