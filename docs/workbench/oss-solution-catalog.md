# OSS Physical AI Solution Candidates

This catalog tracks open-source Physical AI projects that are being onboarded as
Workbench registry candidates through BYOF. These entries are **not** first-class
`npa workbench <tool>` commands yet. Promotion requires the pushed registry image
to run in a real NPA/SkyPilot/Kubernetes E2E workflow and produce declared
artifacts.

Authoring skill: `skills/workflows/oss-solution-registry-onboard/SKILL.md`.

Capabilities are **solution-specific** (upstream env ids, configs, scripts). Do
not collapse them into a shared cross-solution taxonomy.

## Candidate Matrix

| Candidate | Pinned source | Primary capability under test | Artifact | NPA workflow |
| --- | --- | --- | --- | --- |
| ManiSkill | `mani-skill/ManiSkill` `v3.0.1` | `gymnasium_pickcube_registration` (+ isolated cpu/parallel/render probes) | `maniskill_pickcube_step.json` | `byof-maniskill.yaml` |
| MuJoCo Playground | `google-deepmind/mujoco_playground` `v0.2.0` | `mjx_cartpole_step` (+ CheetahRun step, reduced `train-jax-ppo` attempt) | `mujoco_playground_cartpole_step.json` | `byof-mujoco-playground.yaml` |
| RoboCasa | `robocasa/robocasa` `v1.0` | `kitchen_task_registration` (+ lightweight asset download, EGL reset) | `robocasa_kitchen_env_reset.json` | `byof-robocasa.yaml` |
| OpenPI | `Physical-Intelligence/openpi` `15a9616a…` | `policy_config_materialization` (+ checkpoint download, `policy.infer`) | `openpi_pi05_droid_config.json` | `byof-openpi.yaml` |
| DROID policy learning | `droid-dataset/droid_policy_learning` `9a29c832…` | `rlds_config_generator_contract` (+ `droid_100` download, config gen) | `droid_rlds_config_generator.json` | `byof-droid-policy-learning.yaml` |

## Live deferred-capability results

| Solution | Capability | Live status | Evidence / blocker |
| --- | --- | --- | --- |
| ManiSkill | `gymnasium_pickcube_registration` | **accepted** | Live pass `defcap-maniskill-20260708-230227`; 81 registered `-v1` envs |
| ManiSkill | `pickcube_cpu_step` | deferred | Isolated subprocess segfault (`returncode=-11`) in SAPIEN `gym.make` |
| ManiSkill | `pickcube_parallel_envs` | deferred | Same SAPIEN segfault |
| ManiSkill | `pickcube_gpu_rgb_render` | deferred | Same SAPIEN segfault |
| MuJoCo Playground | `mjx_cartpole_step` | **accepted** | Live pass `defcap-mujoco-playground-20260709-005745` (and prior `…-20260708-230227`); Cartpole reward≈0.998 |
| MuJoCo Playground | `mjx_cheetah_run_step` | **accepted** | Live pass; CheetahRun reward≈0.0019 |
| MuJoCo Playground | `train_jax_ppo_cartpole_smoke` | deferred | Entrypoint discovery fixed; full PPO script still heavy (video/EGL deps). Prior run missing console script; later run uploaded PPO logdir artifacts |
| RoboCasa | `kitchen_task_registration` | **accepted** (artifact produced) | Live `defcap-robocasa-20260709-005745` uploaded `robocasa_kitchen_task_registration.json` |
| RoboCasa | `download_kitchen_assets_lw` | deferred / attempted | Smoke exit 1 after registration; asset download attempted with evidence |
| RoboCasa | `kitchen_egl_env_reset` | deferred / attempted | Same run; EGL reset attempted with evidence |
| OpenPI | `policy_config_materialization` | attempted → hard-gate softened | Live `defcap-openpi-20260709-005745` produced config/infer artifacts; smoke exit 1 under old all-or-nothing gate |
| OpenPI | `pi05_droid_checkpoint_download` | deferred / attempted | GCS checkpoint download failed or incomplete in live smoke |
| OpenPI | `pi05_droid_checkpoint_infer` | deferred / attempted | Depends on checkpoint download |
| DROID | `rlds_config_generator_contract` | image pushed; smoke blocked | Image `defcap-droid-policy-learning-20260709-004249` pushed; launch hit SkyPilot API disconnect during `sky check` |
| DROID | `droid_100_download` / `droid_100_config_gen` | deferred / queued | Re-run after softened hard gate + stable Sky API |

## Native Capabilities Per Container

### ManiSkill

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `gymnasium_pickcube_registration` | accepted (live) | Gymnasium env id listing |
| `pickcube_cpu_step` | deferred (segfault) | Isolated subprocess; SAPIEN `gym.make` crashes on cluster |
| `pickcube_parallel_envs` | deferred (segfault) | Isolated subprocess `num_envs=4` |
| `pickcube_gpu_rgb_render` | deferred (segfault) | Isolated subprocess GPU rgb render |
| RL/IL baselines / demos | deferred follow-up | `mani_skill.examples.*` |

### MuJoCo Playground

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `mjx_cartpole_step` | accepted (live) | `registry.load("CartpoleBalance")` reset/step |
| `mjx_cheetah_run_step` | accepted (live) | Additional registered env beyond Cartpole |
| `train_jax_ppo_cartpole_smoke` | attempted | `learning/train_jax_ppo.py` with reduced timesteps |

### RoboCasa

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `kitchen_task_registration` | accepted hard gate | Gymnasium `robocasa/PickPlaceCounterToCabinet` |
| `download_kitchen_assets_lw` | attempted / deferred | `download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw` |
| `kitchen_egl_env_reset` | attempted / deferred | `MUJOCO_GL=egl` gym.make + reset |
| `kitchen_random_rollout` | best-effort | `run_random_rollouts` |

### OpenPI

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `policy_config_materialization` | accepted hard gate | `get_config("pi05_droid")` |
| `pi05_droid_checkpoint_download` | attempted / deferred | `download.maybe_download(gs://openpi-assets/checkpoints/pi05_droid)` |
| `pi05_droid_checkpoint_infer` | attempted / deferred | `create_trained_policy` + `policy.infer` |
| LoRA / fine-tune recipes | deferred follow-up | Upstream fine-tune docs |

### DROID policy learning

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `rlds_config_generator_contract` | accepted hard gate | `droid_runs_language_conditioned_rlds` module contract |
| `droid_100_download` | attempted / deferred | `gsutil cp gs://gresearch/robotics/droid_100` |
| `droid_100_config_gen` | attempted / deferred | Documented `EXP_NAMES` debug subset wiring |
| Full DROID train / `train.py --debug` | deferred follow-up | Needs staged data + longer job |

## Capability Testing In The Onboarding Skill

When creating or onboarding solutions, agents must follow
`skills/workflows/oss-solution-registry-onboard/SKILL.md`:

1. Discover **this solution's** native capabilities from upstream docs.
2. Name capabilities with upstream vocabulary (env ids, configs, scripts).
3. Encode accepted claims as `--workload solution-smoke` with
   `--solution-name`, `--capability-name`, and `--smoke-artifact-name`.
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
