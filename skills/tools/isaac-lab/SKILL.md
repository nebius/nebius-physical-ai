---
name: isaac-lab
description: Use when working on Isaac Lab RL simulation, deployment, SkyPilot workflows, or customer custom-fork support.
---

# Isaac Lab

Isaac Lab is the RL simulation framework. It requires RT cores: use L40S or RTX Pro 6000 only. It will not run correctly on H100 or H200 because those GPUs do not provide RT cores.

Training must invoke headless mode. Verify training commands do not trigger rendering paths.

## Runtime Isaac bootstrap (the container ships no Isaac Sim)

The `npa-isaac-lab` image contains **no NVIDIA Isaac Sim or Isaac Lab code**. It used to bake
Omniverse Kit, which made it non-redistributable; Isaac is now downloaded on first use of
`/isaac-sim/python.sh` from `https://pypi.nvidia.com`, into a cache volume, under the
**operator's own EULA acceptance**. Full rationale:
`docs/workbench/container-packaging.md` and `skills/atomic/solution-licensing/SKILL.md`.

What this changes in practice:

- **Set both variables, or Isaac will not start.** Missing either
  `OMNI_KIT_ACCEPT_EULA=YES` or `ISAACSIM_ACCEPT_EULA=YES` makes the container exit **78**
  with a message naming them. That refusal is deliberate and load-bearing — do not "fix"
  it by baking acceptance into the image; a guard fails the build if anyone does.
- **Reach Isaac through `/isaac-sim/python.sh`** (the value of `ISAAC_LAB_PYTHON`). That is
  the bootstrap shim, and it is what every SkyPilot template, the sim2real engine and the
  workbench CLI already use. A bare `python3` is the *system* interpreter and will not
  find Isaac.
- **Never invoke the shim from a Dockerfile `RUN`.** It would download and bake ~4.5 GB of
  Isaac into a layer. Build-time work uses the image's own venv python.
- **Budget the first start.** Measured on RTX PRO 6000: 111 s cold, 32 ms warm, 10.04 GiB
  of cache. Pre-warm a shared volume once per node/PVC with
  `npa/docker/workbench/common/warm-isaac-cache.yaml`, then run workload pods with
  `NPA_ISAAC_CACHE_READONLY=1`. Otherwise every pod pays it, and 8 GPU pods on a node
  download ~36 GB.
- `isaac-bootstrap status` reports what is cached without needing acceptance or network;
  `isaac-bootstrap verify` additionally launches Isaac Sim headless (needs a GPU).
- No NGC credentials are needed to build or run this image.

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

Canonical onboarding starts at `docs/workbench/getting-started.md`; do not
duplicate credential, S3, Kubernetes, registry, or SkyPilot bootstrap setup here.

Customers can bring their own Isaac Lab fork through an `image_id` override in the SkyPilot YAML. The workbench provides a validated base container; the customer layers their fork on top.

The replacement image must preserve the expected Isaac Lab entry point or runner contract.

Cookbook: `docs/workbench/cookbooks/byof-isaac-lab/README.md`.

Validated BYOF surfaces:

- image override through `npa/scripts/run_isaac_lab_rl.py --image`
  (`w10-byof-image-only-20260520T232650Z`);
- command override through a SkyPilot YAML `run:` block variant passed with
  `--yaml`, invoking `/opt/byof/custom_train.py`
  (`w10-byof-image-and-cmd-20260520T233113Z`).

The runner exposes `--image` directly. It does not expose a `--run-cmd` flag, so
custom entrypoints should use a customer-owned YAML variant that preserves the
runtime contract, checkpoint discovery, manifest creation, and S3 upload block.

## Sim2Real Held-Out Backend

Isaac Lab is also the default sim engine for the Sim2Real loop's non-VLM
held-out rollout eval. The held-out eval is backend-pluggable:

- `sim_backend=isaac` (default): the held-out rollout runs headless Isaac Sim
  inside the Isaac Lab image as the eval component Job. It uses the Isaac Lab
  manipulation task (`Isaac-Lift-Cube-Franka-v0` by default) for a Franka
  pick/lift rollout.
- `sim_backend=genesis`: the existing Genesis `FrankaPickPlaceEnv` path, kept
  fully intact.

Select with `--sim-backend`, env `NPA_SIM2REAL_SIM_BACKEND`, or the runbook
YAML. Both backends emit the identical `npa.sim2real.heldout_eval.v1` per-env
schema (`env_id`/`score`/`success`/`details`), so `report.json` and the
outer-loop gate are backend-agnostic. The VLM eval (Cosmos-Reason) is unchanged.

Asset handling mirrors the Genesis no-fallback provenance discipline:

- Stock: the built-in Isaac lift-cube manipuland, recorded as
  `asset_source=isaac_stock` (no sha256).
- BYO mesh: a customer mesh/URDF imported to USD via Isaac Lab's offline
  converters (`isaaclab.sim.converters.MeshConverter` / `UrdfConverter`),
  recorded as `asset_source=byo_mesh` with a sha256. A mesh that fails to
  import or load raises; there is no silent fallback to the stock asset.

The Isaac Lab image bakes no `npa` code, and Isaac Sim is only importable via
its bundled interpreter `/isaac-sim/python.sh`. The eval component injects
branch `npa` code into that interpreter at start from an S3 source tarball
(`NPA_SIM2REAL_SOURCE_TARBALL_URI`) or, when the repo is reachable, a git clone
(`NPA_SOURCE_REPO`/`NPA_SOURCE_REF`), and ensures `boto3` for the S3 client.

Architecture + licensing rationale: `docs/architecture/sim-backend-selection.md`.

## Operational Safety

Managed VM `deploy` defaults to in-place updates for existing aliases. Terraform
plans that would destroy or replace critical infrastructure are blocked unless
the operator passes `--replace` and confirms with `--yes` for automation.

## Workflows

- Single RL job: `npa/src/npa/workflows/byof/profiles/isaac-lab-rl-train.yaml`.
- Parameter sweep: `npa/workflows/workbench/npa-workflows/isaac-lab-rl-sweep.yaml`.
- Runner: `npa/scripts/run_isaac_lab_rl.py`.

E2E is pending the training command fix tracked by `W9-isaac-lab-e2e-fix`.
