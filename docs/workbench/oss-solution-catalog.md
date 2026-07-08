# OSS Physical AI Solution Candidates

This catalog tracks open-source Physical AI projects that are being onboarded as
Workbench registry candidates through BYOF. These entries are **not** first-class
`npa workbench <tool>` commands yet. Promotion requires the pushed registry image
to run in a real NPA/SkyPilot/Kubernetes E2E workflow and produce declared
artifacts.

Authoring skill: `skills/workflows/oss-solution-registry-onboard/SKILL.md`.

## Candidate Matrix

| Candidate | Pinned source | Cloud fit | NPA workflow | Required E2E evidence |
| --- | --- | --- | --- | --- |
| ManiSkill | `mani-skill/ManiSkill` `v3.0.1` | Strong fit for Linux NVIDIA cloud GPUs; rendering paths require Vulkan inside the container. | `npa/workflows/workbench/npa-workflows/byof-maniskill.yaml` | Build/push image, pull it on Kubernetes, run `PickCube-v1` Gymnasium smoke, upload `npa_byof_summary.json` and smoke logs. |
| MuJoCo Playground | `google-deepmind/mujoco_playground` `v0.2.0` | Strong fit for JAX/MJX GPU training; no RT-core dependency. | `npa/workflows/workbench/npa-workflows/byof-mujoco-playground.yaml` | Build/push image, pull it on Kubernetes, run documented `CartpoleBalance` PPO entrypoint, upload summary and logs. |
| RoboCasa | `robocasa/robocasa` `v1.0` | Good fit for benchmark/eval workloads; headless MuJoCo requires EGL/GLVND/NVIDIA graphics libraries in the container. | `npa/workflows/workbench/npa-workflows/byof-robocasa.yaml` | Build/push image, pull it on Kubernetes, create a `robocasa/PickPlaceCounterToCabinet` env with `MUJOCO_GL=egl`, upload summary and logs. |
| OpenPI | `Physical-Intelligence/openpi` `15a9616a00943ada6c20a0f158e3adb39df2ccac` | Strong fit for GPU inference/fine-tuning; checkpoint access uses upstream GCS paths and VRAM needs vary by mode. | `npa/workflows/workbench/npa-workflows/byof-openpi.yaml` | Build/push image, pull it on Kubernetes, load the documented `pi05_droid` policy config, upload summary and logs. |
| DROID policy learning | `droid-dataset/droid_policy_learning` `9a29c832b4c81bf38401111f5e4cdddaca217581` | Good fit for data/training workflows; full RLDS data is large, so start with config/debug data before full-scale jobs. | `npa/workflows/workbench/npa-workflows/byof-droid-policy-learning.yaml` | Build/push image, pull it on Kubernetes, import the documented RLDS config generator, upload summary and logs. |

## Why These Make Sense For Cloud

### ManiSkill

ManiSkill is a SAPIEN-based robotics simulation and benchmark framework focused
on manipulation. It exposes Gymnasium environments, GPU-parallel simulation, fast
rendering, RL/IL baselines, and real2sim/sim2real examples. It maps well to
Nebius GPU Kubernetes because the expensive work is simulation/rendering
throughput. The main cloud gate is Vulkan: `nvidia-smi` alone is insufficient;
the live image must prove the NVIDIA Vulkan stack is visible inside the pod.

### MuJoCo Playground

MuJoCo Playground is a JAX/MJX robot-learning suite for GPU-accelerated
locomotion, manipulation, and sim-to-real research. It is a useful Workbench
addition because it gives customers a lightweight, non-Isaac, non-Genesis path
for fast single-GPU training. The smoke uses the upstream `CartpoleBalance`
training entrypoint and sets `JAX_DEFAULT_MATMUL_PRECISION=highest`, matching
upstream guidance for NVIDIA GPUs where TF32 can affect stability.

### RoboCasa

RoboCasa/RoboCasa365 is a household manipulation benchmark with diverse kitchen
scenes, tasks, assets, demonstrations, and benchmarking hooks for generalist
robot policies. It complements existing GR00T and LeRobot surfaces as an eval
and dataset-generation target. The main cloud risk is headless rendering:
containers must set `MUJOCO_GL=egl` and include EGL/GLVND libraries.

### OpenPI

OpenPI contains open robot foundation-policy code and checkpoints for the
Physical Intelligence pi model family. It is valuable as a Workbench candidate
because it brings generalist VLA inference and fine-tuning into the marketplace
beside LeRobot and GR00T. The first smoke loads a documented policy config
without downloading large checkpoints; full readiness should add checkpoint
download/inference and LoRA fine-tuning once GCS access and VRAM routing are
validated.

### DROID Policy Learning

DROID policy learning provides training code for the DROID in-the-wild robot
manipulation dataset. It belongs in Workbench as a data and policy-training
integration rather than a simulator. The full dataset is large, so registry
onboarding starts with RLDS config/import validation and should progress to the
debug `droid_100` subset before full training jobs.

## Validation Commands

Validate all candidate workflow specs locally:

```bash
npa/.venv/bin/python -m pytest npa/tests/smoke/test_all_workflow_yamls.py -q
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
  --project <project-alias> \
  --run-id byof-maniskill-smoke \
  --cleanup
```

The registry-ready gate is not satisfied until the live run pulls the pushed
image, executes the smoke command, and writes `npa_byof_summary.json` plus smoke
logs to object storage.
