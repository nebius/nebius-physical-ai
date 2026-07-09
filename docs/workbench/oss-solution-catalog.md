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
| ManiSkill | `pickcube_cpu_step` / `pickcube_parallel_envs` / `pickcube_gpu_rgb_render` | deferred | Isolated subprocess SAPIEN segfault (`returncode=-11`) |
| MuJoCo Playground | `mjx_cartpole_step` | **accepted** | `defcap-mujoco-playground-20260709-005745` (+ prior `…-20260708-230227`) |
| MuJoCo Playground | `mjx_cheetah_run_step` | **accepted** | Same runs; CheetahRun reward≈0.0019 |
| MuJoCo Playground | `train_jax_ppo_cartpole_smoke` | deferred | Attempted; PPO script video/EGL path remains heavy |
| RoboCasa | `kitchen_task_registration` | **accepted** | `defcap-robocasa-20260709-011138` (`smoke_exit_code=0`) |
| RoboCasa | `download_kitchen_assets_lw` / `kitchen_egl_env_reset` / `kitchen_random_rollout` | deferred / attempted | Same run uploaded per-capability JSON evidence |
| OpenPI | `policy_config_materialization` | **accepted** | `defcap-openpi-20260709-011138` (`smoke_exit_code=0`) |
| OpenPI | `pi05_droid_checkpoint_download` / `pi05_droid_checkpoint_infer` | deferred / attempted | Same run uploaded `openpi_pi05_droid_checkpoint_infer.json` |
| DROID | `rlds_config_generator_contract` | **accepted** | `defcap-droid-policy-learning-20260709-011138` (`smoke_exit_code=0`) |
| DROID | `droid_100_download` / `droid_100_config_gen` | deferred / attempted | Same run uploaded `droid_100_download.json` + `droid_100_config_gen.json` |

## Native Capabilities Per Container

### ManiSkill

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `gymnasium_pickcube_registration` | accepted (live) | Gymnasium env id listing |
| `pickcube_cpu_step` | deferred (segfault) | Isolated subprocess; SAPIEN `gym.make` |
| `pickcube_parallel_envs` | deferred (segfault) | Isolated subprocess `num_envs=4` |
| `pickcube_gpu_rgb_render` | deferred (segfault) | Isolated subprocess GPU rgb render |

### MuJoCo Playground

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `mjx_cartpole_step` | accepted (live) | `registry.load("CartpoleBalance")` reset/step |
| `mjx_cheetah_run_step` | accepted (live) | Additional registered env beyond Cartpole |
| `train_jax_ppo_cartpole_smoke` | deferred | `learning/train_jax_ppo.py` reduced timesteps |

### RoboCasa

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `kitchen_task_registration` | accepted hard gate (live) | Gymnasium `robocasa/PickPlaceCounterToCabinet` |
| `download_kitchen_assets_lw` | attempted / deferred | `download_kitchen_assets --type tex …` |
| `kitchen_egl_env_reset` | attempted / deferred | `MUJOCO_GL=egl` gym.make + reset |
| `kitchen_random_rollout` | best-effort | `run_random_rollouts` |

### OpenPI

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `policy_config_materialization` | accepted hard gate (live) | `get_config("pi05_droid")` |
| `pi05_droid_checkpoint_download` | attempted / deferred | `download.maybe_download(gs://openpi-assets/…)` |
| `pi05_droid_checkpoint_infer` | attempted / deferred | `create_trained_policy` + `policy.infer` |

### DROID policy learning

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `rlds_config_generator_contract` | accepted hard gate (live) | `droid_runs_language_conditioned_rlds` module contract |
| `droid_100_download` | attempted / deferred | `gsutil cp gs://gresearch/robotics/droid_100` |
| `droid_100_config_gen` | attempted / deferred | Documented `EXP_NAMES` debug subset wiring |

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
