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
| ManiSkill | `mani-skill/ManiSkill` `v3.0.1` | `pickcube_cpu_step` (+ registration, parallel envs; GPU rgb render probed) | `maniskill_pickcube_step.json` | `byof-maniskill.yaml` |
| MuJoCo Playground | `google-deepmind/mujoco_playground` `v0.2.0` | `mjx_cartpole_step` (+ CheetahRun step, reduced `train-jax-ppo`) | `mujoco_playground_cartpole_step.json` | `byof-mujoco-playground.yaml` |
| RoboCasa | `robocasa/robocasa` `v1.0` | `kitchen_egl_env_reset` (+ task registration, lightweight asset download) | `robocasa_kitchen_env_reset.json` | `byof-robocasa.yaml` |
| OpenPI | `Physical-Intelligence/openpi` `15a9616a…` | `pi05_droid_checkpoint_infer` (+ config materialization, checkpoint download) | `openpi_pi05_droid_config.json` | `byof-openpi.yaml` |
| DROID policy learning | `droid-dataset/droid_policy_learning` `9a29c832…` | `droid_100_config_gen` (+ RLDS contract, `droid_100` download) | `droid_rlds_config_generator.json` | `byof-droid-policy-learning.yaml` |

## Native Capabilities Per Container

### ManiSkill

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `gymnasium_pickcube_registration` | required in smoke | Gymnasium env id listing |
| `pickcube_cpu_step` | required in smoke | Quickstart create/reset/step with `render_backend="none"` |
| `pickcube_parallel_envs` | required in smoke | Quickstart `num_envs>1` |
| `pickcube_gpu_rgb_render` | probed (may defer on Vulkan) | `render_mode="rgb_array"` + GPU render backend |
| RL/IL baselines / demos | deferred follow-up | `mani_skill.examples.*` |

### MuJoCo Playground

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `mjx_cartpole_step` | required in smoke | `registry.load("CartpoleBalance")` reset/step |
| `mjx_cheetah_run_step` | required in smoke | Additional registered env beyond Cartpole |
| `train_jax_ppo_cartpole_smoke` | required in smoke | `train-jax-ppo --env_name CartpoleBalance --num_timesteps 256` |

### RoboCasa

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `kitchen_task_registration` | required in smoke | Gymnasium `robocasa/PickPlaceCounterToCabinet` |
| `download_kitchen_assets_lw` | required in smoke | `download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw` |
| `kitchen_egl_env_reset` | required in smoke | `MUJOCO_GL=egl` gym.make + reset |
| `kitchen_random_rollout` | best-effort in smoke | `run_random_rollouts` |

### OpenPI

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `policy_config_materialization` | required in smoke | `get_config("pi05_droid")` |
| `pi05_droid_checkpoint_download` | required in smoke | `download.maybe_download(gs://openpi-assets/checkpoints/pi05_droid)` |
| `pi05_droid_checkpoint_infer` | required in smoke | `create_trained_policy` + `policy.infer` |
| LoRA / fine-tune recipes | deferred follow-up | Upstream fine-tune docs |

### DROID policy learning

| Capability | Status target | Upstream basis |
| --- | --- | --- |
| `rlds_config_generator_contract` | required in smoke | `droid_runs_language_conditioned_rlds` module contract |
| `droid_100_download` | required in smoke | `gsutil cp gs://gresearch/robotics/droid_100` |
| `droid_100_config_gen` | required in smoke | Documented `EXP_NAMES` debug subset wiring |
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
