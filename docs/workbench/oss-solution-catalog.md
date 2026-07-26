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
| Open Dreamer (world model, **2-GPU min**) | `next-state/open-dreamer` `2b10640` | `dreamer4_tokenizer_train_two_gpu` | `open_dreamer_world_model_2gpu.json` | `byof-open-dreamer.yaml` |

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
| Open Dreamer | `jax_two_gpu_data_parallel_mesh` | **accepted** | `byof-open-dreamer-mc-20260726T013512Z` (real Minecraft/VPT, jax 0.10.1, 2×RTX PRO 6000 Blackwell, mesh `{data:2, model:1}`) |
| Open Dreamer | `minecraft_vpt_video_dataloader` | **accepted** | Same run (`dreamer.data.build_iterator` minecraft_vpt batch `[48,24,128,128,3]` sharded across 2 devices) |
| Open Dreamer | `dreamer4_tokenizer_train_two_gpu` | **accepted** | Same run (`scripts/train_tokenizer.py` exit 0, 15000 steps on real Minecraft; reconstruction closely tracks gameplay — sky/grass/trees/hotbar, see `gt_decoded`) |
| Open Dreamer | `dreamer4_latent_tokenization` | **accepted** | Same run (`scripts/tokenize_minecraft_dataset.py`, real latents + `latent_stats`, real 27/121 VPT actions) |
| Open Dreamer | `dreamer4_dynamics_train_two_gpu` | **accepted** | Same run (`scripts/train_dynamics.py` exit 0, 15000 steps on the Minecraft latents) |
| Open Dreamer | `dreamer4_action_conditioned_dream_rollout` | **accepted** | Same run (`sample_video` context→dream; dream maintains coherent Minecraft scenery across the 32-frame horizon; dream PSNR 17.3 dB) |
| Open Dreamer | `world_model_rerun_visualization` | **accepted** | Same run (21 MB `.rrd` = 64 frames × observation/dream/gt_decoded + 10 reconstruction grids, `rerun-sdk==0.31.4`, loaded live into the agent Rerun viewer) |

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
| `kitchen_random_rollout` | accepted (live) | `run_random_rollouts` with video (`gymnasium==0.29.1` + `env.sim` bind) |

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

### Open Dreamer (world model, 2-GPU minimum)

JAX/Flax Dreamer 4 world-model training pipeline. This is the reference
multi-GPU BYOF candidate: its accepted capability requires a real `>=2` GPU
device mesh, so it uses `byof-solution-smoke-rtxpro-2gpu.yaml`
(`RTXPRO-6000-BLACKWELL-SERVER-EDITION:2`), not the single-GPU profile.

The smoke is the full Dreamer 4 loop end to end on a **real Minecraft/VPT**
gameplay subset (128x128), headlined by an action-conditioned **dream rollout**
(context frames -> predicted future frames vs ground truth).

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `jax_two_gpu_data_parallel_mesh` | accepted hard gate (live) | `dreamer.parallel.build_parallel("data")` `{data:2, model:1}` over 2 `jax.devices()` |
| `minecraft_vpt_video_dataloader` | accepted (live) | `dreamer.data.build_iterator` minecraft_vpt MP4 path (decord decode + VPT action parse) + device sharding (2 devices) |
| `dreamer4_tokenizer_train_two_gpu` | accepted hard gate (live) | `scripts/train_tokenizer.py` causal video tokenizer trained on real Minecraft frames, data-parallel across the mesh |
| `dreamer4_latent_tokenization` | accepted (live) | `scripts/tokenize_minecraft_dataset.py` encodes the episodes into latent ArrayRecords + `latent_stats` with real 27-binary/121-categorical VPT actions |
| `dreamer4_dynamics_train_two_gpu` | accepted (live) | `scripts/train_dynamics.py` action-conditioned latent dynamics trained on the real Minecraft latents (core world-model loop) |
| `dreamer4_action_conditioned_dream_rollout` | accepted (live) | `dreamer.sampler.sample_video` rolls out predicted future gameplay frames from context + future actions; reports dream PSNR |
| `world_model_rerun_visualization` | accepted (live) | Rerun `.rrd` with synchronized `world/observation` (GT) + `world/dream` (predicted) + `world/gt_decoded` (tokenizer ceiling) + `world/tokenizer_reconstruction` streams, loaded into the agent viewer |

Data: a real **Minecraft/VPT** contractor-gameplay subset (OpenAI VPT `.mp4` +
`.jsonl`), center-cropped and resized to 128x128, staged as `minecraft_vpt`
ArrayRecords (pickled `{video: mp4_bytes, video_shape, actions: [VPT dicts],
source}`) to the run bucket under `datasets/minecraft_vpt_128_64/` and pulled at
run time. Actions parse to the real 27-binary / 121-categorical VPT layout that
`train_dynamics.py` asserts. Dream fidelity scales with the tokenizer/dynamics
training budget (`OD_TOK_STEPS`/`OD_DYN_STEPS`; upstream trains ~200k). LPIPS is
left off (no HF download); FVD/I3D scoring (`eval_fvd.py`) remains a follow-up.

## First-class Workbench tools (not BYOF)

LeRobot is already a first-class `npa workbench lerobot` tool. Supported
package/image tags:

| Version | Status | Notes |
| --- | --- | --- |
| `0.5.1` | default | `npa-lerobot:0.5.1` |
| `0.6.0` | additional | `npa-lerobot:0.6.0`; lean extras + `env_eval_freq` |

Select with `--lerobot-version`. Manifest:
`npa/src/npa/deploy/lerobot_version_manifest.json`. Upstream:
https://huggingface.co/blog/lerobot-release-v060

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
