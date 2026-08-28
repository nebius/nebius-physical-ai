---
name: isaac-lab
description: Use when working on Isaac Lab RL simulation, deployment, SkyPilot workflows, or customer custom-fork support.
---

# Isaac Lab

Isaac Lab is the RL simulation framework. The canonical workbench pins the
latest published 3.Y beta point, `v3.0.0-beta2.patch1` / wheel
`3.0.0b2.post1`, with Isaac Sim `6.0.1.0`. Upstream has not labeled this a GA
release; keep the beta limitation explicit. It requires RT cores: use L40S or
RTX Pro 6000 only. Do not route this graphics/PhysX path to B200, H100, or H200.

Generation 3 training must invoke `--visualizer none`. A compatibility path
that can also receive a generation 2 image must select `--headless` only when
the image's `ISAAC_LAB_VERSION` starts with `2.`.

Before provisioning, building, downloading, or submitting an Isaac workload,
load `skills/atomic/third-party-eula-preflight/SKILL.md`. NPA defaults the
run-scoped `ACCEPT_EULA=Y` value for non-interactive use and preserves a
recognized negative value or `--no-accept-eula` opt-out. Legacy affirmative
spellings normalize case-insensitively; unrecognized values are errors.

## Runtime Isaac bootstrap (the container ships no Isaac Sim)

The `npa-isaac-lab` image contains **no NVIDIA Isaac Sim, Isaac Lab, Omniverse
Client, or other proprietary NVIDIA runtime code**. It used to bake Omniverse
Kit, which made it non-redistributable; the exact runtime wheel set is now
downloaded on first use of `/isaac-sim/python.sh` from
`https://pypi.nvidia.com`, hash-verified, and installed into a cache volume
under the **operator's own EULA acceptance**. Full rationale:
`docs/workbench/container-packaging.md` and `skills/atomic/solution-licensing/SKILL.md`.

What this changes in practice:

- **Unset acceptance succeeds; explicit opt-out refuses.** NPA defaults
  `ACCEPT_EULA=Y`. Empty, `N`, `NO`, `0`, `FALSE`, or `--no-accept-eula`
  exit **78** before download. `Y`, `YES`, `1`, and `TRUE` are accepted
  case-insensitively; other values are invalid. The launcher derives
  `OMNI_KIT_ACCEPT_EULA=YES` internally; do not expose duplicate user plumbing.
  Keep `PRIVACY_CONSENT` and telemetry off.
- **Reach Isaac through `/isaac-sim/python.sh`** (the value of `ISAAC_LAB_PYTHON`). That is
  the bootstrap shim, and it is what every SkyPilot template, the sim2real engine and the
  workbench CLI already use. A bare `python3` is the *system* interpreter and will not
  find Isaac.
- **Never invoke the shim from a Dockerfile `RUN`.** It would download and bake
  the multi-gigabyte Isaac runtime into a layer. Build-time work uses the
  image's own venv python.
- **Budget the first start.** Use the generation-specific measurements in
  `docs/workbench/isaac-lab-3.md`; do not reuse generation 2 cache numbers for
  generation 3. Pre-warm a shared volume once per node/PVC with
  `npa/docker/workbench/common/warm-isaac-cache.yaml`, then run workload pods
  with `NPA_ISAAC_CACHE_READONLY=1`. Otherwise every pod pays the cold fetch.
- `isaac-bootstrap status` reports what is cached without needing acceptance or network;
  `isaac-bootstrap verify` additionally launches Isaac Sim headless (needs a GPU).
- No NGC credentials are needed to build or run this image. Native
  `--runtime vm` installation is intentionally unsupported for generation 3;
  managed and BYOVM deployments use the payload-clean container contract.

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

### Standalone checkpoint eval

`npa workbench isaac-lab eval` runs the supplied RSL-RL policy headlessly; it
does not invoke a VLM. Checkpoint loading is fail-closed: a load error produces
a structured failure artifact and a non-zero command result, never a
random-action substitute.

Choose the success predicate to match the task:

- `--success-metric survival` for locomotion, with termination as failure;
- `--success-metric goal-distance --success-distance-m <metres>` for
  manipulation or reach tasks;
- `--success-metric auto` to prefer a simulator-native success signal, then a
  measurable goal distance, and otherwise survival.

Use `--seed`, `--num-episodes`, `--max-steps-per-episode`, and
`--min-success-rate` for a repeatable held-out evaluation. The output is
`npa_isaac_lab_eval_summary.json` with format `npa.isaac_lab.eval.v1`; it
records checkpoint provenance, `policy_loaded`, per-episode metrics,
`success_rate`, and `passed`. An S3 output prefix is uploaded on both runtime
success and failure so failed evaluations remain diagnosable. `passed=false`
does not turn a completed rollout into a runtime error; automation should gate
on `passed` (the Sim2Real workflow does this in Stage 11). With
`--output-format json`, `eval_status`, `policy_loaded`, `success_rate`, and
`passed` are top-level structured CLI fields; callers do not need to scrape
the remote log tail.

### Trained-policy visual export

`npa workbench isaac-lab train --export-trajectories` captures genuine Isaac
Sim RGB by default during the post-training policy rollout. The exporter must
load the trained RSL-RL checkpoint fail-closed, launch Isaac with cameras
enabled, call the environment's `rgb_array` renderer, and write one `rgb.npy`
array per episode with exactly the same frame count as `state.npy` and
`actions.npy`. It records runtime version, checkpoint hash, renderer, image
dimensions/count, `policy_loaded`, and the shared
episode/frame/timestamp timeline in `trajectories/meta.json`.

The Isaac-to-LeRobot adapter automatically encodes these frames as the
`observation.images.workspace` video feature and preserves their provenance.
The LeRobot-to-Rerun adapter maps each episode to its own video asset and
`VideoFrameReference` timeline, makes the trained-policy environment the
prominent view, and omits evaluation panes when no evaluation entities exist.
Metadata-only rollouts remain valid but must not be described as containing an
environment visual. `--no-export-rgb` is the explicit scalar-only opt-out.

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

When Stage 10 has a genuine Isaac trainer checkpoint, object storage, and a
registry-qualified Isaac image, it selects `byo_isaac_eval`. That vectorized
adapter uses the standalone evaluator's shared `load_rsl_rl_policy`, metric,
`npa.isaac_lab.eval.v1`, and failure-summary implementation while retaining
generated-environment IDs/seeds and held-out camera capture. It writes
`eval/heldout/isaac-eval-summary.json` and nests the same evidence in
`eval/heldout/report.json`. Runtime, checkpoint, or policy-load failure aborts
Stage 10; `passed=false` means the eval completed below its quality bar and is
handled by the Stage 11 threshold gate. Stage 14 remains responsible for the
run-level RRD/MCAP visualization artifacts.

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
