# OSS Physical AI Solution Candidates

This catalog tracks open-source Physical AI projects that are being onboarded as
Workbench registry candidates through BYOF. These entries are **not** first-class
`npa workbench <tool>` commands yet. Promotion requires the pushed registry image
to run in a real NPA/SkyPilot/Kubernetes E2E workflow and produce declared
artifacts.

Authoring skill: `skills/workflows/oss-solution-registry-onboard/SKILL.md`.

**Live status:** all five containers below pass build/push, Kubernetes image
pull, capability-specific `solution-smoke`, and S3 artifact upload.

## Candidate Matrix

| Candidate | Pinned source | Live status | Accepted capability | Artifact | NPA workflow |
| --- | --- | --- | --- | --- | --- |
| ManiSkill | `mani-skill/ManiSkill` `v3.0.1` | pass | `gymnasium_pickcube_registration` (`sim_env`) | `maniskill_pickcube_step.json` | `byof-maniskill.yaml` |
| MuJoCo Playground | `google-deepmind/mujoco_playground` `v0.2.0` | pass | `mjx_cartpole_step` (`sim_env`) | `mujoco_playground_cartpole_step.json` | `byof-mujoco-playground.yaml` |
| RoboCasa | `robocasa/robocasa` `v1.0` | pass | `kitchen_task_registration` (`sim_env` + assets) | `robocasa_kitchen_env_reset.json` | `byof-robocasa.yaml` |
| OpenPI | `Physical-Intelligence/openpi` `15a9616a…` | pass | `policy_config_materialization` (`policy_config`) | `openpi_pi05_droid_config.json` | `byof-openpi.yaml` |
| DROID policy learning | `droid-dataset/droid_policy_learning` `9a29c832…` | pass | `rlds_config_generator_contract` (`dataset_contract`) | `droid_rlds_config_generator.json` | `byof-droid-policy-learning.yaml` |

## Native Capabilities Per Container

Only **accepted** rows are registry claims. Deferred rows stay documented so
agents know what to promote next with stronger smokes.

### ManiSkill

| Capability | Family | Status | Notes |
| --- | --- | --- | --- |
| Gymnasium `PickCube-v1` registration | `sim_env` | accepted | Live golden smoke |
| GPU-parallel multi-env simulation | `sim_env` | deferred | Needs stable SAPIEN scene on cluster |
| Headless Vulkan rendering | `render_headless` | deferred | Vulkan ICD incompatible on prior live attempt |
| RL/IL baselines | `policy_train` | deferred | Not smoked yet |
| Demo / real2sim paths | `datagen` | deferred | Asset path not wired |

### MuJoCo Playground

| Capability | Family | Status | Notes |
| --- | --- | --- | --- |
| MJX `CartpoleBalance` reset/step | `sim_env` | accepted | Live golden smoke with JAX PRNG + zero action |
| Broader locomotion/manipulation suite | `sim_env` | deferred | Only Cartpole exercised |
| PPO / training recipes | `policy_train` | deferred | Full train not in smoke matrix |

### RoboCasa

| Capability | Family | Status | Notes |
| --- | --- | --- | --- |
| Kitchen task Gymnasium registration | `sim_env` | accepted | `robocasa/PickPlaceCounterToCabinet` |
| Packaged assets root present | `dataset_contract` | accepted | Proven in same artifact |
| Headless EGL env create/reset | `render_headless` | deferred | Full kitchen assets blocked prior runs |
| Benchmark / demo eval | `eval_benchmark` | deferred | Not smoked yet |
| Scene / demo datagen | `datagen` | deferred | Needs asset staging contract |

### OpenPI

| Capability | Family | Status | Notes |
| --- | --- | --- | --- |
| `pi05_droid` policy config materialization | `policy_config` | accepted | Live golden smoke |
| Checkpoint download + inference | `policy_infer` | deferred | GCS/HF + VRAM |
| LoRA / fine-tune | `policy_train` | deferred | Not smoked yet |

### DROID policy learning

| Capability | Family | Status | Notes |
| --- | --- | --- | --- |
| RLDS language-conditioned generator contract | `dataset_contract` | accepted | `EXP_NAMES` / helper signature |
| Debug subset train (`droid_100`) | `policy_train` | deferred | Needs staged debug data |
| Full DROID training | `policy_train` | deferred | Large dataset; out of smoke scope |

## Why These Make Sense For Cloud

### ManiSkill

ManiSkill is a SAPIEN-based robotics simulation and benchmark framework focused
on manipulation. It exposes Gymnasium environments, GPU-parallel simulation, fast
rendering, RL/IL baselines, and real2sim/sim2real examples. It maps well to
Nebius GPU Kubernetes because the expensive work is simulation/rendering
throughput. The main cloud gate for fuller smokes is Vulkan: `nvidia-smi` alone
is insufficient; a future accepted `render_headless` claim must prove the NVIDIA
Vulkan stack inside the pod.

### MuJoCo Playground

MuJoCo Playground is a JAX/MJX robot-learning suite for GPU-accelerated
locomotion, manipulation, and sim-to-real research. It is a useful Workbench
addition because it gives customers a lightweight, non-Isaac, non-Genesis path
for fast single-GPU training. The accepted smoke uses the upstream
`CartpoleBalance` registry path and sets `JAX_DEFAULT_MATMUL_PRECISION=highest`,
matching upstream guidance for NVIDIA GPUs where TF32 can affect stability.

### RoboCasa

RoboCasa/RoboCasa365 is a household manipulation benchmark with diverse kitchen
scenes, tasks, assets, demonstrations, and benchmarking hooks for generalist
robot policies. It complements existing GR00T and LeRobot surfaces as an eval
and dataset-generation target. The accepted smoke proves task registration and
assets packaging; fuller EGL env reset remains deferred until kitchen assets are
staged reliably.

### OpenPI

OpenPI contains open robot foundation-policy code and checkpoints for the
Physical Intelligence pi model family. It is valuable as a Workbench candidate
because it brings generalist VLA inference and fine-tuning into the marketplace
beside LeRobot and GR00T. The accepted smoke loads a documented policy config
without downloading large checkpoints; full readiness should add checkpoint
download/inference and LoRA fine-tuning once GCS access and VRAM routing are
validated.

### DROID Policy Learning

DROID policy learning provides training code for the DROID in-the-wild robot
manipulation dataset. It belongs in Workbench as a data and policy-training
integration rather than a simulator. The accepted smoke validates the RLDS
config generator contract; progress next to the debug `droid_100` subset before
full training jobs.

## Capability Testing In The Onboarding Skill

When creating or onboarding solutions, agents must follow
`skills/workflows/oss-solution-registry-onboard/SKILL.md`:

1. Discover native capabilities from upstream docs (not marketing copy).
2. Map each claim to a capability family (`sim_env`, `render_headless`,
   `datagen`, `policy_config`, `policy_infer`, `policy_train`,
   `dataset_contract`, `eval_benchmark`, `serve`).
3. Encode an accepted claim as `--workload solution-smoke` with
   `--solution-name`, `--capability-name`, and `--smoke-artifact-name`.
4. Require live Kubernetes pull of the pushed image plus S3 upload of the named
   JSON artifact.
5. Keep deferred capabilities explicit; never mark them accepted.

## Validation Commands

Validate all candidate workflow specs locally:

```bash
npa/.venv/bin/python -m pytest npa/tests/smoke/test_all_workflow_yamls.py -q
npa/.venv/bin/python -m pytest npa/tests/workflows/test_byof_solution_smokes.py -q
```

Plan an individual candidate:

```bash
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-maniskill.yaml \
  --run-id byof-maniskill-smoke --json
```

Run live BYOF E2E for a candidate by materializing the candidate config into
`run_byof_repo.py`. Example:

```bash
npa/.venv/bin/python npa/scripts/run_byof_repo.py \
  --repo-url https://github.com/mani-skill/ManiSkill.git \
  --repo-ref v3.0.1 \
  --base-profile ubuntu \
  --base-image maniskill/base:latest \
  --build-command 'python3 -m pip install --no-cache-dir -e .' \
  --workload solution-smoke \
  --smoke-command '<documented smoke command>' \
  --solution-name maniskill \
  --capability-name gymnasium_pickcube_registration \
  --smoke-artifact-name maniskill_pickcube_step.json \
  --project <project-alias> \
  --run-id byof-maniskill-smoke \
  --cleanup
```

The registry-ready gate is not satisfied until the live run pulls the pushed
image, executes the smoke command, and writes `npa_byof_summary.json`, smoke
logs, and the named capability artifact to object storage.

If live SkyPilot reports `FAILED_PRECHECKS` before the pod starts, first verify
that the BYOF runner passed `--infra k8s/<context>` for the configured
cluster and that the cluster can schedule the container-smoke request. A
precheck failure is infrastructure evidence, not a passing solution smoke.
