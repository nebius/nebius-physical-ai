# OSS Physical AI Solution Candidates

This catalog tracks open-source Physical AI projects that are being onboarded as
Workbench registry candidates through BYOF. These entries are **not** first-class
`npa workbench <tool>` commands yet. Promotion requires the pushed registry image
to run in a real NPA/SkyPilot/Kubernetes E2E workflow and produce declared
artifacts.

Authoring skill: `skills/workflows/oss-solution-registry-onboard/SKILL.md`.

Capabilities are **solution-specific** (upstream env ids, configs, scripts). Do
not collapse them into a shared cross-solution taxonomy — every solution is
unique and must be tested with its own upstream-named capabilities.

## Candidate Matrix

| Candidate | Pinned source | Primary (hard-gate) capability | Artifact | NPA workflow |
| --- | --- | --- | --- | --- |
| ManiSkill | `mani-skill/ManiSkill` `v3.0.1` | `gymnasium_pickcube_registration` | `maniskill_pickcube_step.json` | `byof-maniskill.yaml` |
| MuJoCo Playground | `google-deepmind/mujoco_playground` `v0.2.0` | `mjx_cartpole_step` (+ CheetahRun) | `mujoco_playground_cartpole_step.json` | `byof-mujoco-playground.yaml` |
| RoboCasa | `robocasa/robocasa` `v1.0` | `kitchen_task_registration` | `robocasa_kitchen_env_reset.json` | `byof-robocasa.yaml` |
| OpenPI | `Physical-Intelligence/openpi` `15a9616a…` | `policy_config_materialization` | `openpi_pi05_droid_config.json` | `byof-openpi.yaml` |
| DROID policy learning | `droid-dataset/droid_policy_learning` `9a29c832…` | `rlds_config_generator_contract` | `droid_rlds_config_generator.json` | `byof-droid-policy-learning.yaml` |

## Live capability results

| Solution | Capability | Live status | Run / evidence |
| --- | --- | --- | --- |
| ManiSkill | `gymnasium_pickcube_registration` | **accepted** | `defcap-maniskill-20260708-230227` (81 `-v1` envs) |
| ManiSkill | `pickcube_cpu_step` / `pickcube_parallel_envs` / `pickcube_gpu_rgb_render` | **accepted** | `defcap11-maniskill-20260709-043408` (sapien 3.0.3 on CUDA Ubuntu22.04/py3.10; Blackwell render OK) |
| MuJoCo Playground | `mjx_cartpole_step` | **accepted** | `defcap8-mujoco-playground-20260709-024455` (+ prior `…-005745`) |
| MuJoCo Playground | `mjx_cheetah_run_step` | **accepted** | Same runs; CheetahRun reward≈0.0019 |
| MuJoCo Playground | `train_jax_ppo_cartpole_smoke` | **accepted** | `defcap9-mujoco-playground-20260709-034059` (`brax_ppo_train_api`, jax 0.8.0) |
| RoboCasa | `kitchen_task_registration` | **accepted** | `defcap8-robocasa-20260709-024455` (+ prior `…-011138`) |
| RoboCasa | `download_kitchen_assets_lw` | **accepted** | `defcap17-robocasa-20260709-060243` (IIFAN fixtures+objects; restored git accessories) |
| RoboCasa | `kitchen_egl_env_reset` | **accepted** | `defcap17-robocasa-20260709-060243` (post-download subprocess; 58 lightwheel cats; obs dict) |
| RoboCasa | `kitchen_random_rollout` | **accepted** | `defcap20-robocasa-20260710-032142` (`run_random_rollouts` + mp4 `22150` bytes; `gymnasium==0.29.1` + `env.sim` bind) |
| OpenPI | `policy_config_materialization` | **accepted** | `defcap9-openpi-20260709-034059` (+ prior) |
| OpenPI | `pi05_droid_checkpoint_download` | **accepted** | `defcap9-openpi-20260709-034059` via `maybe_download` |
| OpenPI | `pi05_droid_checkpoint_infer` | **accepted** | `defcap9-openpi-20260709-034059` (`make_droid_example`, actions `[15,8]`) |
| DROID | `rlds_config_generator_contract` | **accepted** | `defcap8-droid-policy-learning-20260709-024455` (+ prior) |
| DROID | `droid_100_download` | **accepted** | Same run (`https_meta` `dataset_info.json`) |
| DROID | `droid_100_config_gen` | **accepted** | Same run (`EXP_NAMES` droid_100 wiring) |

## Native Capabilities Per Container

### ManiSkill

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `gymnasium_pickcube_registration` | accepted (live) | Gymnasium env id listing |
| `pickcube_cpu_step` | accepted (live) | Isolated subprocess; sapien 3.0.3 + physx_cpu |
| `pickcube_parallel_envs` | accepted (live) | Isolated subprocess `num_envs=4` physx_cuda |
| `pickcube_gpu_rgb_render` | accepted (live) | Isolated subprocess GPU rgb render on Blackwell |

### MuJoCo Playground

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `mjx_cartpole_step` | accepted (live) | `registry.load("CartpoleBalance")` reset/step |
| `mjx_cheetah_run_step` | accepted (live) | Additional registered env beyond Cartpole |
| `train_jax_ppo_cartpole_smoke` | accepted (live) | brax PPO train API reduced timesteps (jax&lt;0.8.1) |

### RoboCasa

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `kitchen_task_registration` | accepted hard gate (live) | Gymnasium `robocasa/PickPlaceCounterToCabinet` |
| `download_kitchen_assets_lw` | accepted (live) | `download_kitchen_assets --type tex tex_generative fixtures_lw` |
| `kitchen_egl_env_reset` | accepted (live) | `MUJOCO_GL=egl` gym.make + reset after asset download |
| `kitchen_random_rollout` | accepted (live) | `run_random_rollouts` after post-download EGL reset |

### OpenPI

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `policy_config_materialization` | accepted hard gate (live) | `get_config("pi05_droid")` |
| `pi05_droid_checkpoint_download` | accepted (live) | `download.maybe_download(gs://openpi-assets/…)` |
| `pi05_droid_checkpoint_infer` | accepted (live) | `make_droid_example()` + `policy.infer` |

### DROID policy learning

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `rlds_config_generator_contract` | accepted hard gate (live) | `droid_runs_language_conditioned_rlds` module contract |
| `droid_100_download` | accepted (live) | HTTPS metadata pull of `droid_100/1.0.0/dataset_info.json` |
| `droid_100_config_gen` | accepted (live) | Documented `EXP_NAMES` debug subset wiring |

## Capability Testing In The Onboarding Skill

When creating or onboarding solutions, agents must follow
`skills/workflows/oss-solution-registry-onboard/SKILL.md`:

1. Discover **this solution's** native capabilities from upstream docs.
2. Name capabilities with upstream vocabulary (env ids, configs, scripts).
3. Encode one hard-gate capability as `--capability-name` and attempt related
   deferred capabilities in the same smoke with explicit evidence.
4. Require live Kubernetes pull of the pushed image plus S3 upload of the named
   JSON artifact.
5. Keep deferred capabilities explicit; never mark them accepted.
6. Do **not** invent a shared family taxonomy across solutions.

## Validation Commands

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

The registry-ready gate is not satisfied until the live run pulls the pushed
image, executes the smoke command, and writes `npa_byof_summary.json`, smoke
logs, and the named capability artifact to object storage.
