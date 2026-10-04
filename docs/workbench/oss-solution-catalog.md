# OSS Physical AI Solution Candidates

[Workbench docs](README.md)

This catalog tracks open-source Physical AI projects that are being onboarded as
Workbench registry candidates through BYOF, including entries subsequently
promoted to native tools or public images as noted below. Promotion requires the pushed registry image
to run in a real NPA/SkyPilot/Kubernetes E2E workflow and produce declared
artifacts.

Authoring skill: `skills/workflows/oss-solution-registry-onboard/SKILL.md`.

Capabilities are **solution-specific** (upstream env ids, configs, scripts). Do
not collapse them into a shared cross-solution taxonomy — every solution is
unique and must be tested with its own upstream-named capabilities.

## Candidate Matrix

| Candidate | Pinned source | Primary (hard-gate) capability | Artifact | NPA workflow |
| --- | --- | --- | --- | --- |
| Habitat-Sim (**neutral runtime-fetch; development proof only**) | `facebookresearch/habitat-sim` `57ee4941…` | `skokloster_castle_rgb_depth_bullet_traversal` | `habitat-sim-smoke.json` + saved RGB/depth observations | `habitat-sim-smoke.yaml` |
| LIBERO | `Lifelong-Robot-Learning/LIBERO` `8f1084e3…` | `libero_spatial_bc_rnn_train_reload_heldout` | Canonical `libero-smoke.json` + checkpoint digest/reload evidence (checkpoint remains local and is never uploaded) | `byof-libero.yaml` |
| DM05-Lerobot-LIBERO checkpoint (**runtime compatibility pending**) | `Dexmal/DM05-Lerobot-LIBERO` `c22df98a…`, predecessor `Dexmal/DM05-Lerobot` `716afe31…` | `dm05_lerobot_libero_matched_rollout_comparison` | protocol-bound native `eval_info.json`, suite metrics, decoded comparison MP4 and RRD | `dm05-lerobot-libero-comparison.yaml` |
| ManiSkill | `mani-skill/ManiSkill` `v3.0.1` | `gymnasium_pickcube_registration` | `maniskill_pickcube_step.json` | `byof-maniskill.yaml` |
| MuJoCo Playground | `google-deepmind/mujoco_playground` `v0.2.0` | `mjx_cartpole_step` (+ CheetahRun) | `mujoco_playground_cartpole_step.json` | `byof-mujoco-playground.yaml` |
| Gymnasium-Robotics | `Farama-Foundation/Gymnasium-Robotics` `4d1ebecb…` | `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` | `gymnasium-robotics-smoke.json` | `byof-gymnasium-robotics.yaml` |
| RoboCasa | `robocasa/robocasa` `v1.0` | `kitchen_task_registration` | `robocasa_kitchen_env_reset.json` | `byof-robocasa.yaml` |
| Enactic OpenArm (**accepted public image; Isaac runtime fetch**) | `enactic/openarm_mujoco` `2.2.0` + `enactic/openarm_isaac_lab` `bad82e…` | `openarm_mujoco_bimanual_rollout` + `Isaac-Reach-OpenArm-v0` | MuJoCo/Isaac trajectories and RSL-RL checkpoint | `openarm-simulators.yaml` |
| RoboTwin 2.0 (**operator BYOF candidate; normal submit blocked**) | `RoboTwin-Platform/RoboTwin` `96c1feab…` | `beat_block_hammer_successful_seed_replay_collection` | operator evidence: `robotwin-smoke.json` + native HDF5 + MP4; registry admission deferred | `byof-robotwin.yaml` |
| OpenPI | `Physical-Intelligence/openpi` `15a9616a…` | connected direct / cross-pod serve / LoRA optimizer smoke / held-out evaluation, plus the upstream full-DROID fine-tuning recipe | `openpi_pi05_droid_jointpos_polaris_inference.json` plus connected mode reports; full-DROID emits preparation and 100-update qualification RRDs, then immutable run-derived progress RRDs/manifests through the 100,000-update checkpoint | `byof-openpi.yaml` → `openpi-pi05-four-mode.yaml`; trusted public-image build → `openpi-pi05-full-droid-finetune.yaml` |
| flex-pi (**accepted public image**) | `geyan21/flex-pi` `20c1b2b…` | strict released-checkpoint action-only inference | `actions.json` + input/result provenance | `flex-pi-b200-inference.yaml` / `flex-pi-rtxpro-inference.yaml` |
| DROID policy learning | `droid-dataset/droid_policy_learning` `9a29c832…` | `rlds_config_generator_contract` | `droid_rlds_config_generator.json` | `byof-droid-policy-learning.yaml` |
| evo trajectory evaluation | `MichaelGrupp/evo` `8dd6cfe0…` (`v1.35.1`) | `evo_ape_rpe_trajectory_evaluation` | metric archives + matched PNGs + `evo_trajectory_evaluation.json` | `byof-evo.yaml` |
| AprilTag 3 | `AprilRobotics/apriltag` `94be7839…` (`v3.4.5`) | `apriltag_real_image_fiducial_detection` | labeled corner metrics + camera-consumer records + annotated PNGs + `apriltag_fiducial_evaluation.json` | `byof-apriltag.yaml` |
| robomimic | `ARISE-Initiative/robomimic` `d309eae…` | `lift_ph_lowdim_checkpoint_reload_action` (deferred) | `robomimic-smoke.json` (not produced) | `byof-robomimic.yaml` |
| Open Dreamer (world model, **2-GPU min**) | `next-state/open-dreamer` `2b10640` | `dreamer4_tokenizer_train_two_gpu` | `open_dreamer_world_model_2gpu.json` | `byof-open-dreamer.yaml` |
| Alibaba Wan 2.2 TI2V-5B | `Wan-Video/Wan2.2` `42bf4cf…` | `wan2.2_ti2v_5b_text_to_video` | capability JSON + runtime inventory + MP4 | `byof-wan2.2.yaml` |
| Lightricks LTX-2.5 (**accepted public image; entitled runtime fetch**) | `Lightricks/LTX-2` `fd4ded7f…` | `ltx2_5_text_to_video` | `ltx2_5_text_to_video.json` + provenance manifest + MP4 | `byof-ltx2.yaml` |
| Alibaba Wan 2.2 TI2V-5B (**4-GPU distributed**) | same pinned source/checkpoint | `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | multi-GPU capability JSON + rank topology + runtime inventory + MP4 | `byof-wan2.2-multigpu.yaml` |

## Live capability results

| Solution | Capability | Live status | Run / evidence |
| --- | --- | --- | --- |
| Habitat-Sim | `skokloster_castle_rgb_depth_bullet_traversal` / `headless_nvidia_egl_rgb_depth_render` / `bullet_physics_world_step` / `greedy_geodesic_agent_traversal` | **development evidence; supported release quarantined** | [Exact-digest development manifest](validation/habitat-sim-development-image-manifest.json): 19 RGB/depth frame pairs, 19 Bullet steps and 2.2466 metres of navigation on one RTX PRO 6000 Blackwell. Historical producer evidence; no policy-training or long-benchmark claim. |
| LIBERO | `libero_spatial_bc_rnn_train_reload_heldout` | **qualification pending; payload-free public-development staging permitted; not released** | Requires complete-byte and anonymous-pull proof followed by one STRICT-bound B200 run of the exact candidate digest: eight upstream BC-RNN/Adam steps on the official LIBERO-Spatial demonstration, checkpoint reload, and full trajectory-disjoint held-out evaluation |
| DM05-Lerobot-LIBERO | `dm05_lerobot_libero_matched_rollout_comparison` | **operator-private qualification in progress; not released** | The exact Apache-2.0 checkpoint-compatible implementation is pinned to `hbzfeng/lerobot@6eede4f7d2efe6b4f6a58ddb7b13ed55e2346b9c` (closed/superseded upstream PR #4051) in a private validation derivative. No public image or benchmark claim exists until an immutable digest passes native 40-task evaluation and independently decoded MP4/RRD artifact inspection. |
| ManiSkill | `gymnasium_pickcube_registration` | **accepted** | `defcap-maniskill-20260708-230227` (81 `-v1` envs) |
| ManiSkill | `pickcube_cpu_step` / `pickcube_parallel_envs` / `pickcube_gpu_rgb_render` | **accepted** | `defcap11-maniskill-20260709-043408` (sapien 3.0.3 on CUDA Ubuntu22.04/py3.10; Blackwell render OK) |
| MuJoCo Playground | `mjx_cartpole_step` | **accepted** | `defcap8-mujoco-playground-20260709-024455` (+ prior `…-005745`) |
| MuJoCo Playground | `mjx_cheetah_run_step` | **accepted** | Same runs; CheetahRun reward≈0.0019 |
| MuJoCo Playground | `train_jax_ppo_cartpole_smoke` | **accepted** | `defcap9-mujoco-playground-20260709-034059` (`brax_ppo_train_api`, jax 0.8.0) |
| Gymnasium-Robotics | `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` | **historical private proof; neutral development build release-quarantined** | The accepted `c308945a` result remains bound to its private digest. The redesigned source has a payload-free development-build path but no accepted release or public image proof; any new private receipt remains owner-only. |
| RoboCasa | `kitchen_task_registration` | **accepted** | `defcap8-robocasa-20260709-024455` (+ prior `…-011138`) |
| RoboCasa | `download_kitchen_assets_lw` | **accepted** | `defcap17-robocasa-20260709-060243` (IIFAN fixtures+objects; restored git accessories) |
| RoboCasa | `kitchen_egl_env_reset` | **accepted** | `defcap17-robocasa-20260709-060243` (post-download subprocess; 58 lightwheel cats; obs dict) |
| RoboCasa | `kitchen_random_rollout` | **accepted** | `defcap20-robocasa-20260710-032142` (`run_random_rollouts` + mp4 `22150` bytes; `gymnasium==0.29.1` + `env.sim` bind) |
| Enactic OpenArm | `openarm_mujoco_bimanual_rollout` | **accepted** | exact public development digest: 500 real `mj_step` calls, finite joint/command/energy trace, and fully decoded 100-frame H.264 render |
| Enactic OpenArm | `Isaac-Reach-OpenArm-v0` rollout | **accepted** | same digest on RTX PRO 6000: 64 environments × 100 real PhysX/CUDA steps with finite rewards and policy observations |
| Enactic OpenArm | `Isaac-Reach-OpenArm-v0` RSL-RL training | **accepted** | same digest: upstream trainer completed one iteration and emitted an independently validated serialized Torch checkpoint |
| RoboTwin 2.0 | `beat_block_hammer_successful_seed_replay_collection` | **operator evidence retained; registry admission deferred** | One RTX PRO 6000 operator run produced 123 state/action pairs and 124 decoded frames. Public CLI execution and the normal-submit worker bridge remain blocked; see [scope and digest](byof-robotwin.md#retained-operator-evidence-and-readiness). |
| OpenPI | `pi05_droid_jointpos_polaris_checkpoint_download` | **accepted** | Canonical isolated B200 gate: image build/push/digest verification, then 12,434,530,837 runtime-only GCS bytes with 27-object generation-manifest provenance; exact scoped `NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` is runtime-only |
| OpenPI | `pi05_droid_jointpos_polaris_direct_infer` | **accepted** | Same digest-pinned B200 `sm_100` gate; deterministic Franka input produced finite `float64[15,8]` joint-position targets |
| OpenPI | `pi05_droid_jointpos_polaris_served_infer` | **accepted builder regression** | Same gate; upstream WebSocket health + same-pod client round trip produced finite `float64[15,8]` |
| OpenPI | `pi05_droid_jointpos_polaris_cross_pod_serve` | **accepted** | Isolated single-B200 connected gate: private ClusterIP, ready digest-pinned server Deployment, and a distinct CPU client pod completed two finite `float64[15,8]` requests; exact service cleanup passed |
| OpenPI | `pi05_droid_jointpos_polaris_lora_optimizer_smoke` | **accepted** | Same connected gate: upstream pi0.5 LoRA forward/backward/AdamW step, finite loss, changed trainable-state hash, and independently reloadable private Orbax checkpoint |
| OpenPI | `pi05_droid_jointpos_polaris_heldout_evaluate` | **accepted** | Same connected gate: exact trained-checkpoint reload, two samples excluded from the four-sample training split, finite upstream loss and action MAE/MSE, and finite `float64[15,8]` trajectory |
| flex-pi | `robotwin_action_only_infer` | **accepted** | Current r2 digest independently on one B200 and one RTX PRO 6000: complete checkpoint-state load, compiled four-step inference, finite `float32[32,14]` actions, and three read-back-verified JSON artifacts per target. The historical `0.1.0-cu128` B200 capacity run completed 24 independent one-GPU replicas; that scaling record does not qualify r2. |
| DROID | `rlds_config_generator_contract` | **accepted** | `defcap8-droid-policy-learning-20260709-024455` (+ prior) |
| DROID | `droid_100_download` | **accepted** | Same run (`https_meta` `dataset_info.json`) |
| DROID | `droid_100_config_gen` | **accepted** | Same run (`EXP_NAMES` droid_100 wiring) |
| evo | `evo_ape` / `evo_rpe` / `evo_traj` | **live-qualified candidate** | `evo-live-20260920t0111z`: digest-pinned CPU Kubernetes execution passed 2 positive and 2 negative controls, decoded 10 native result archives, preserved native KITTI plots plus hash-linked labeled review copies, and passed identify-first hosted-VLM review after exact-prompt cross-swap controls; registry admission remains a maintainer decision |
| AprilTag 3 | `apriltag_real_image_fiducial_detection` | **live-qualified private-image candidate** | Standard SkyPilot CPU execution matched all 47 upstream labels across three photographs, rejected blank/noise controls, and reproduced the complete overlay pixels covered by retained calibrated visual review; registry admission and public image publication are not claimed |
| Open Dreamer | `jax_two_gpu_data_parallel_mesh` | **accepted** | `byof-open-dreamer-mc-20260726T013512Z` (real Minecraft/VPT, jax 0.10.1, 2×RTX PRO 6000 Blackwell, mesh `{data:2, model:1}`) |
| Open Dreamer | `minecraft_vpt_video_dataloader` | **accepted** | Same run (`dreamer.data.build_iterator` minecraft_vpt batch `[48,24,128,128,3]` sharded across 2 devices) |
| Open Dreamer | `dreamer4_tokenizer_train_two_gpu` | **accepted** | Same run (`scripts/train_tokenizer.py` exit 0, 15000 steps on real Minecraft; reconstruction closely tracks gameplay — sky/grass/trees/hotbar, see `gt_decoded`) |
| Open Dreamer | `dreamer4_latent_tokenization` | **accepted** | Same run (`scripts/tokenize_minecraft_dataset.py`, real latents + `latent_stats`, real 27/121 VPT actions) |
| Open Dreamer | `dreamer4_dynamics_train_two_gpu` | **accepted** | Same run (`scripts/train_dynamics.py` exit 0, 15000 steps on the Minecraft latents) |
| Open Dreamer | `dreamer4_action_conditioned_dream_rollout` | **accepted** | Same run (`sample_video` context→dream; dream maintains coherent Minecraft scenery across the 32-frame horizon; dream PSNR 17.3 dB) |
| Open Dreamer | `world_model_rerun_visualization` | **accepted** | Same run (21 MB `.rrd` = 64 frames × observation/dream/gt_decoded + 10 reconstruction grids, `rerun-sdk==0.31.4`, loaded live into the agent Rerun viewer) |
| Wan 2.2 TI2V-5B | `wan2.2_ti2v_5b_text_to_video` | **accepted** | exact zero-payload public-dev digest with Torch 2.13.0/CUDA 13.0: native TI2V-5B generation on RTX PRO 6000 Blackwell (`sm_120`) |
| Wan 2.2 TI2V-5B | `wan2.2_decoded_mp4_validation` | **accepted** | same exact-digest run: 2,807,385-byte H.264 MP4, 1280x704, 17 frames at 24 fps; full decode and non-uniform-content gates passed |
| Wan 2.2 TI2V-5B | `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | **accepted historical evidence** | prior Torch 2.7.1/CUDA 12.8/NCCL 2.27.7 runtime: one node, 4×B200 (`sm_100`), world size 4, T5/DiT FULL_SHARD FSDP, Ulysses size 4, and official `generate.py`; current NCCL 2.29.7 gate is not yet live-qualified |
| Wan 2.2 TI2V-5B | `wan2.2_distributed_rank_topology_validation` | **accepted historical evidence** | same prior run: four unique GPU hashes/ranks 0–3, NCCL sum 10/10 per rank, 480 distributed-attention and 1,920 all-to-all calls per rank, three barriers, final barrier, and process-group teardown |
| Wan 2.2 TI2V-5B | `wan2.2_decoded_mp4_validation` (distributed run) | **accepted historical evidence** | same prior run: 2,809,770-byte H.264 MP4, 1280x704, 17 frames at 24 fps; spatial stddev 71.9485, pixel range 255, temporal delta 9.714725, SHA-256 `9574f79c…94865` |

## Native Capabilities Per Container

### Habitat-Sim

Pinned MIT source:
`facebookresearch/habitat-sim@57ee4941dc4765240f0f91f70b2c97a919bf9038`.
Upstream warns that beyond v0.3.4, Meta internal teams do not officially
maintain releases or provide active development. Its public build target is a
neutral Ubuntu/Python bootstrap with accompanying exact Ubuntu source. The
simulator and scientific/native dependencies are fetched only at runtime.
Supported release selection remains quarantined. The [development image manifest](validation/habitat-sim-development-image-manifest.json)
binds the retained exact-digest 19-step managed-workflow result to its image
producer and evidence hashes.

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `skokloster_castle_rgb_depth_bullet_traversal` | development proof; release quarantined | Upstream `Simulator`, `make_cfg`, pathfinder, greedy follower, RGB/depth sensors, and saved observations on the exact Skokloster scene |
| `headless_nvidia_egl_rgb_depth_render` | development proof; release quarantined | Headless NVIDIA OpenGL strings plus NVIDIA EGL libraries loaded into the renderer process |
| `bullet_physics_world_step` | development proof; release quarantined | Bullet-enabled build and advancing world time through `Simulator.step(dt=1/60)` |
| `greedy_geodesic_agent_traversal` | development proof; release quarantined | Navmesh path, upstream follower action sequence, and nonzero start-to-end displacement |

The hard gate fetches the official Meta test-scene archive referenced by the
pinned Habitat-Sim tree, verifies its 94,590,970-byte SHA-256
`1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06`,
extracts only the exact hash-pinned `skokloster-castle.glb` and `.navmesh`, and
deletes the archive. The runtime installer uses the existing immutable Ubuntu
snapshot and hash-locked Python/source inputs. The public image contains those
locks and launchers, without the simulator or scientific runtime payload. Habitat's pinned README and the original asset identify the
demo as CC BY 4.0; the proof carries attribution, license/original links, and
modification provenance. Matterport3D, HM3D, Replica, other proprietary or gated
datasets, semantic annotations, and distributed Habitat-Lab training are
deferred. The renderer targets exactly one RTX PRO 6000 Blackwell and never B200. See
[`byof-habitat-sim.md`](byof-habitat-sim.md).
### LIBERO

Pinned runtime source: `Lifelong-Robot-Learning/LIBERO`
`8f1084e3132a39270c3a13ebe37270a43ece2a01` (MIT). The narrow admission
candidate runtime-fetches one official `libero_spatial` demonstration from
`yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`
and verifies its 508,779,600 bytes against SHA-256
`ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`.
The upstream LIBERO publisher declares its datasets CC BY 4.0; the mirror card's
conflicting Apache-2.0 tag is not used to broaden rights.
Task conditioning runtime-fetches the independently pinned Apache-2.0
`google-bert/bert-base-cased@cd5ef92a9fb2f889e972770a36d4ed042daf221e`
files and runs the pinned upstream LIBERO `AutoTokenizer`/`AutoModel`
`pooler_output` path. Neither those model bytes nor the demonstration is baked.
The public-neutral candidate also bakes no LIBERO, robomimic, MuJoCo, PyTorch,
CUDA/NVIDIA runtime, task/render asset, populated cache, checkpoint, credential,
or output byte. It remains absent from the release manifest and public image
table, and unqualified until byte, provenance, anonymous-pull, and live
acceptance. Historical private r15 bytes are old-head evidence only.
The trusted public workflow permits payload-free development publication after
local image-byte/security gates. The local scan records the complete ordered
layer and flattened-rootfs inventory; the pushed digest must match it and pass
anonymous pull. Runtime qualification and customer acknowledgement remain
separate requirements for execution and supported release promotion.

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `libero_official_demo_sha256` | qualification pending | Exact official HDF5 mirror revision, byte size, SHA-256, 50 trajectories, 5,068 samples, task language, BDDL, and initial-state hashes |
| `libero_upstream_bert_task_conditioning` | qualification pending | Exact Apache-2.0 BERT revision and file hashes, pinned LIBERO embedding-source hash, and finite 768-dimensional upstream `pooler_output` |
| `libero_trajectory_disjoint_heldout_split` | qualification pending | Deterministic 40-train / 10-held-out trajectory split; no trajectory may appear in both partitions |
| `libero_spatial_bc_rnn_train_reload_heldout` | hard gate, qualification pending | Upstream BERT task conditioning plus `Sequential.observe` + `BCRNNPolicy` + the upstream-configured `torch.optim.Adam` for exactly eight nonzero optimizer steps, strict upstream checkpoint reload, held-out NLL, and finite reloaded 7-DoF action predictions on exactly one B200 (`sm_100`) |

An authorized runtime sparse-fetch retains the hash-bound BDDL and initial
states but never fetches the unused render-asset tree. The official
demonstration is fetched into a manifest-addressed cache outside the artifact
directory and is never baked or uploaded. A missing, denied, expired, or
mismatched customer/run authorization refuses before cache or network mutation;
credentials establish upstream access only, and runtime fetch is delivery, not
permission. Acceptance
requires the Pod-observed immutable image digest in `libero-smoke.json`; imports,
BDDL parsing, dataset inventory, or zero-step training do not pass. Rendered
closed-loop sweeps, all 130 tasks, lifelong-algorithm comparison, and physical
robots remain deferred. See [`byof-libero.md`](byof-libero.md).

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

### Gymnasium-Robotics

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `registered_shadow_hand_environment` | historical private proof only | Upstream `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1` registration at exact source commit |
| `mujoco_physics_steps` / `mujoco_contacts` | historical private proof only | 120 upstream `env.step` calls, 2,400 MuJoCo substeps, finite rewards, contacts, and quantitative state/orientation change |
| `continuous_touch_sensor_response` | historical private proof only | Official 92-site Shadow Hand `sensordata` vector with nonzero live readings |
| `egl_rgb_rendering` | historical private proof only | Actual 240×320 RGB frames, distinct hashes, measured rate, and loaded NVIDIA EGL library |
| `rtx_pro_6000_blackwell_execution` | historical private proof only | One strict RTX PRO 6000 Blackwell (`sm_120`) plus private pushed/pod-observed immutable digest equality |

This remains a minimal BYOF candidate. It has no model, external dataset,
gated asset, terms acceptance, RL training claim, expert score, other
environment-family claim, or physical-robot transfer claim. See
[`byof-gymnasium-robotics.md`](byof-gymnasium-robotics.md).
The neutral zero-Shadow-payload bootstrap has a development-build path but is
release-quarantined. An
owner-only reference build supplied the scanner's exact config and ordered
20-DiffID anchors, but did not complete the product scan, SBOM, push, or
immutable-digest gates and left no accepted artifact. Pinned source, Shadow
assets, MuJoCo/Python runtime, and populated caches are runtime-only; historical
private evidence does not transfer to redesigned image or executable bytes.

### RoboCasa

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `kitchen_task_registration` | accepted hard gate (live) | Gymnasium `robocasa/PickPlaceCounterToCabinet` |
| `download_kitchen_assets_lw` | accepted (live) | `download_kitchen_assets --type tex tex_generative fixtures_lw` |
| `kitchen_egl_env_reset` | accepted (live) | `MUJOCO_GL=egl` gym.make + reset after asset download |
| `kitchen_random_rollout` | accepted (live) | `run_random_rollouts` with video (`gymnasium==0.29.1` + `env.sim` bind) |

> **Promoted to a first-class workbench tool.** RoboCasa is now a native
> `npa workbench robocasa` tool with a dedicated `npa-robocasa` container, a
> FastAPI service, CLI, SDK, and workflow toolRefs. The BYOF candidate
> (`byof-robocasa.yaml`) is preserved for compatibility, but the native tool is
> the maintained surface. See `skills/tools/robocasa/SKILL.md` and
> `workflows/testing/robocasa-smoke.yaml`.

### Enactic OpenArm

OpenArm is split upstream by capability. NPA pins `enactic/openarm_mujoco`
release 2.2.0 (`a8c979629f2591ad035d99d338ce114969e6cddc`) and the untagged
`enactic/openarm_isaac_lab` repository at
`bad82e23716e6941c2de78ccb978f57c78b37734`. Both simulator repositories and
their included MJCF/USD robot assets are Apache-2.0. The separate hardware/CAD
repository is not an image input and no rights for it are inferred.

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `openarm_mujoco_bimanual_rollout` | accepted (live) | `openarm_mujoco.v2.openarm_demo_xml`, `JointResolver`, 500 real `mj_step` calls, finite joint/command/energy NPZ, and 100-frame H.264 render |
| `Isaac-Reach-OpenArm-v0` rollout | accepted (live) | upstream registered Isaac Lab environment, 64 vectorized environments × 100 real PhysX/CUDA steps, finite reward and policy-observation trace on RTX PRO 6000 |
| `Isaac-Reach-OpenArm-v0` RSL-RL training | accepted (live) | pinned upstream `scripts/reinforcement_learning/rsl_rl/train.py`, one completed iteration, and independently validated serialized Torch checkpoint |

The public `npa-openarm` image contains the Apache-2.0 OpenArm sources and
MuJoCo closure, but no Isaac Sim, Isaac Lab, or Omniverse Kit bytes. Isaac is
hash-pinned and fetched into the operator's runtime cache through the shared
acceptance/refusal bootstrap. See [OpenArm](openarm.md) and
`workflows/testing/openarm-simulators.yaml`.

### RoboTwin 2.0

Pinned bimanual SAPIEN simulation and native data-collection candidate. The
source is `RoboTwin-Platform/RoboTwin`
`96c1feab536306b50c26af200044fcdf126e8904`; required runtime assets come from
`TianxingChen/RoboTwin2.0`
`785feb15aa4a4f532395ad2b1d2be5f28cb561ad`. The operator workload
fetches and hash-checks only the aggregate objects and embodiments archives,
then uses the
ALOHA-AgileX embodiment and custom `020_hammer` object. No asset bytes are baked
into the image: the live harness scans the exact image digest's rootfs
and every layer before it may submit the GPU run.

The workflow describes a CPU-only outer launcher and a fixed one-RTX inner
profile, but normal `npa workbench workflow submit` remains blocked on
remote-source and worker identity proof. The public BYOF CLI also refuses
RoboTwin execution. The standalone operator script can invoke the guarded inner
launcher after authorization and input checks. Its retained GPU result is
documented in [the operator guide](byof-robotwin.md#retained-operator-evidence-and-readiness);
it does not establish agent or normal-submit readiness. Plans and rendered YAML
retain sanitized placeholders, and registry admission remains deferred.

| Capability | Status | Upstream basis / required evidence |
| --- | --- | --- |
| `sapien_vulkan_rt_renderer` | qualification pending | `SapienRenderer`, `rt` camera shader, successful `vulkaninfo`, and SAPIEN device summary from one RTX PRO 6000 (`sm_120`) |
| `beat_block_hammer_successful_seed_search` | qualification pending | official `scripts/collect_data.py beat_block_hammer demo_clean` seed-search phase, reduced only to one episode |
| `beat_block_hammer_successful_seed_replay` | qualification pending hard gate | official replay must finish with `check_success()` true; imports, registration, simulator startup, or a planned trajectory do not pass |
| `robotwin_native_hdf5_collection` | qualification pending hard gate | non-empty native HDF5 with RoboTwin provenance, action/state/vision groups, positive action count, size, and SHA-256 |
| `robotwin_rendered_mp4` | qualification pending hard gate | fully decoded MP4 with positive dimensions and exactly `action_count + 1` frames, size, and SHA-256 |

The public bootstrap contains none of these bytes. Its implemented customer
runtime compiles pinned CuRobo v0.7.8 for `sm_120`. CuRobo's NVIDIA license
limits use to noncommercial research/evaluation. The operator's exact
`noncommercial` statement for this bounded run is compatible with that field of
use for containerization and technical workload validation/evaluation; it
expires with the run and does not authorize hosted service or broader outputs.
CuRobo's field-of-use limit continues to bind use and service claims and every
generated workload. No additional generated-output restriction was found in
the inspected authoritative terms for the five declared classes: native HDF5
action/state data, decoded MP4, rendered frames, smoke JSON, and summary JSON.
Hosted-service use remains unapproved even though a
zero-vendor-payload bootstrap may be eligible for public redistribution after
exact-byte review. Public artifacts require exact-revision payload probes;
gated artifacts additionally require the customer's own runtime-only credential
and an exact provider/artifact/revision/terms access result before provisioning.
An explicit customer-terminal decision or the existing hosted authorization
boundary gates runtime delivery. Supported release promotion remains
quarantined pending validation of the supported submission path, and the candidate
is absent from the supported public image table. The Hugging Face
asset repository card classifies the exact locked `embodiments.zip` and
`objects.zip` members at revision
`785feb15aa4a4f532395ad2b1d2be5f28cb561ad` as MIT. Other embodiments,
unselected future task assets, training/evaluation, physical deployment, and
their outputs remain independently deferred. Registry credentials, runtime
fetch, and the byte-absence scan do not grant permission. Customer-issued
entitlement for CUDA, cuDNN, and CuRobo plus the exact applicable payload probes
remain pre-fetch/pre-run gates. See
[`byof-robotwin.md`](byof-robotwin.md) for the exact license, GPU, workflow, and
artifact contract.

Deferred: the 50-task sweep, randomized-background coverage, policy training or
evaluation, other embodiments and object-license review, and physical-robot
deployment.

### flex-pi

| Capability | Status | Upstream basis / NPA evidence |
| --- | --- | --- |
| `robotwin_action_only_infer` | accepted (live) | Released 6B RoboTwin checkpoint; three RGB cameras, 14D state, and language input produced finite 32-step bimanual action chunks independently on B200 and RTX PRO 6000 |
| `strict_checkpoint_state` | accepted (live) | Maintained fail-closed loader rejected missing/unexpected MoT keys and required proprio, DINO, and pointmap state before the success marker |
| `action_artifact_provenance` | accepted (live) | Source, checkpoint, data, runtime-asset, input, action, and image identities persisted in three hash-verified JSON artifacts |
| `blackwell_compiled_infer` | accepted (live) | Current r2 digest passed real compiled inference independently on one B200 and one RTX PRO 6000. Historical `0.1.0-cu128` paired benchmarks measured 2.645× and 2.278× median speedup; those values apply only to the old digest |

> **Promoted to a first-class workbench tool.** flex-pi is available through
> `npa workbench flex-pi`, `npa.sdk.workbench.flex_pi`, the
> `workbench.flex_pi.infer` toolRef, and maintained B200 and RTX PRO 6000 workflows.
> The public image contains no weights, observation media, or populated cache.
> See `skills/tools/flex-pi/SKILL.md` and `docs/workbench/flex-pi.md`.

### OpenPI

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `pi05_droid_jointpos_polaris_checkpoint_download` | accepted (live) | canonical build/push/digest gate; anonymous runtime `download.maybe_download(gs://openpi-assets/checkpoints/polaris/…)`; 27 objects / 12,434,530,837 bytes; weights and the exact scoped terms acceptance are never baked |
| `pi05_droid_jointpos_polaris_direct_infer` | accepted (live) | digest-pinned B200 `sm_100` `get_config("pi05_droid_jointpos_polaris")` + direct `policy.infer`; finite `float64[15,8]` joint-position targets |
| `pi05_droid_jointpos_polaris_served_infer` | accepted builder regression (live) | upstream `WebsocketPolicyServer` + same-pod `WebsocketClientPolicy`; served finite `float64[15,8]` |
| `pi05_droid_jointpos_polaris_cross_pod_serve` | accepted (live) | upstream server Deployment + private ClusterIP + distinct client Job; two finite `float64[15,8]` requests (39.350 s cold, 50.2 ms warm); exact cleanup |
| `pi05_droid_jointpos_polaris_lora_optimizer_smoke` | accepted (live) | supported upstream pi0.5 LoRA config; one real forward/backward/AdamW update (loss 0.145676, update L2 0.0957375), changed trainable state, and reloadable 29-file Orbax checkpoint |
| `pi05_droid_jointpos_polaris_heldout_evaluate` | accepted (live) | exact trained-checkpoint reload; two disjoint held-out samples; finite mean upstream loss 0.182892, action MAE 0.0111408 and MSE 0.000200538, plus valid `float64[15,8]` trajectory |
| `pi05_full_droid_finetune_rtxpro8` | qualification pending | pinned upstream non-LoRA recipe: DROID RLDS 1.0.1, 10,000,000-frame normalization, batch 256, 100,000 updates, and FSDP 8 across eight nodes with one RTX PRO 6000 each; acceptance requires the private immutable checkpoint/report plus preparation, qualification, and staged progress RRD manifests from the dedicated live run |

The Polaris request/response schema, upstream terms, licensing boundary, B200 stack,
and 15 Hz re-query guidance are documented in
[`openpi-pi05-polaris.md`](openpi-pi05-polaris.md). Historical validation of
the older generic `pi05_droid` checkpoint is not treated as Polaris/B200 proof.
The connected four-mode gate is the only surface that may establish cross-pod
ClusterIP serving and live optimizer/evaluation acceptance. It does not claim
physical Franka success, external Ingress, convergence, or robot success from
offline evaluation.

### DROID policy learning

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `rlds_config_generator_contract` | accepted hard gate (live) | `droid_runs_language_conditioned_rlds` module contract |
| `droid_100_download` | accepted (live) | HTTPS metadata pull of `droid_100/1.0.0/dataset_info.json` |
| `droid_100_config_gen` | accepted (live) | Documented `EXP_NAMES` debug subset wiring |

### evo trajectory evaluation

NPA pins the maintained upstream `MichaelGrupp/evo` source at
`8dd6cfe0ec1747f9e1b5b569edd82c54d1a3f422` (`v1.35.1`). The source and
Python package are GPL-3.0-or-later. This is an operator-built BYOF candidate,
not a published NPA image; any future conveyance must preserve the GPL license
and corresponding-source obligations. The
[independently audited CPU workload proof](https://github.com/nebius/nebius-physical-ai/blob/899caeab007bd2d55221ff3bc7b195f02a828c11/docs/testing/evidence/evo-proof/README.md)
and its [payload-scan scope correction](https://github.com/nebius/nebius-physical-ai/pull/584#issuecomment-5747248494)
publish sanitized metrics, controls, hardware applicability, visual-review
failures, and current merge blockers without live infrastructure identifiers.

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `evo_ape` | live-qualified candidate | Native absolute pose error over translation, saved as a decoded result archive and matched error-map plot |
| `evo_rpe` | live-qualified candidate | Native relative pose error at a declared frame or distance delta, with the complete finite error distribution retained |
| `evo_traj` | live-qualified candidate | Native matched trajectory plots for the pinned KITTI ground truth, ORB, and S-PTAM examples |
| `trajectory_acceptance_controls` | live-qualified candidate | The digest-pinned Kubernetes run accepted both 120-pose low-error controls and rejected nonlinear drift plus malformed input with zero false positives/negatives |
| `decoded_plot_validation` | live-qualified auxiliary check | Six review-facing plots and eight additional `evo_ape` raw/control plots are decoded, dimension/content-checked, and hash-retained; only the six review-facing plots enter the capture/review set. Five auxiliary `evo_traj` plots are outside this check, including the 651x491 synthetic speed plot below its 800x600 floor |

The integration gate is about reliable evaluation, not improving the upstream
ORB or S-PTAM estimates. Representative KITTI metrics are reported as observed
and are not tuned to the synthetic control threshold. Passing the gate does not
establish navigation success, robot safety, sensor accuracy, or generalization
to another trajectory format. The live qualification retained ten decoded
archives and exact plot hashes and matched the local generated-image metrics.
An exact-prompt ORB/S-PTAM cross-swap probe invalidated the first unlabeled
per-estimate visual review with two false positives. That failed evidence is
preserved. The corrected run keeps each native map and emits a hash-linked
review copy with the estimate and reference visibly named; identify-first
cross-swap calibration then had zero false positives, and both final live plots
scored 0.95 with no critical defect. A complete filesystem/layer-history
restricted-payload scan found no hits across 22,826 image entries; complete
archive-byte accounting is not claimed. This remains an operator-built
candidate, not a published NPA image; registry admission and future conveyance
remain maintainer decisions.

In `npa.evo.capture-manifest.v1`, the unsuffixed `kitti_*_ape` entries are the
identity-labeled review copies, while the corresponding `*_native` entries are
evo's untouched maps. Review entries link back through `source_path` and
`source_sha256`; consumers must follow those fields rather than infer provenance
from the key suffix.

### AprilTag 3 fiducial detection

NPA pins `AprilRobotics/apriltag` release `v3.4.5`
(`94be783968e5091bcc9972c72c84fd63efce2935`) under BSD-2-Clause. The
representative upstream test photographs depict NASA Swarmathon and are
separately identified upstream as CC-BY-SA-2.0. This is an operator-built BYOF
candidate; no container-image publication is claimed. The derived
[CPU workload proof](evidence/apriltag-fiducial/README.md) retains the required
photo attribution and share-alike license.

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `native_apriltag_ctest` | accepted (live CPU) | Pinned native detector regression suite passed 3/3 in the digest-pinned Kubernetes workload |
| `apriltag_real_image_fiducial_detection` | accepted upstream-parity gate (live CPU) | Reproduced all 47 recorded `tag36h11` IDs/corners across three photographs; precision/recall 1.0 and the 0.000050 px maximum residual describe parity with four-decimal upstream records, not localization accuracy (upstream tolerance: 0.1 px) |
| `blank_and_noise_false_positive_controls` | accepted objective gate (live CPU) | Blank and fixed-seed random-noise images both produced zero detections |
| `camera_consumer_observation_export` | accepted (live CPU) | 47 source-hashed records with IDs, centers, ordered corners, margins, and hamming distances |
| `source_linked_annotation_capture` | accepted objective bytes; retained calibrated presentation review | Three source-bound annotated PNGs match the complete overlay pixels in the later qualified Kimi-K3 review sheets; no fresh VLM call was made |

The 2026-10-02 standard SkyPilot CPU rerun used client source
`3301aa36eb9f386dcca067fc709de3118149c504` and the qualified private image index
`sha256:7bb4606386f4487eeeb6953ee5843a1e4698a485eac4cde3cbdd7f3a69d6612e`,
built from `d0057c80bd19563c2484bd793d9f8ef0a443ce4f`. Its recipe and smoke
commands match the executing client. All 24 output objects were downloaded and
rehashed. Independent checks matched 47 detections and 376 corner coordinates,
with aggregate residual RMSE `0.0000273889 px`, zero false positives/negatives,
and three passing native CTests. The actual task image and CPU requests were
observed; owned task, controller, and pull-secret cleanup was verified. The
pod observer reported a connection error on shutdown, retained separately from
the successful workload and independent resource-absence checks.

Earlier hosted-judge calibration failed, and that attempt's final review was
correctly gated off; the [original CPU proof](evidence/apriltag-fiducial/README.md)
retains that history. The later
[immutable calibrated visual proof](https://github.com/nebius/nebius-physical-ai/blob/e5f4210090dac2d1a1880bdffdc33a9e042c102b/docs/testing/evidence/apriltag-kimi-k3-production-census/README.md)
records four disclosed controls and four committed holdout cases passing before
all three production sheets scored `0.95`. The fresh overlays match the entire
overlay pixels embedded in those retained sheets. Reusing that presentation
evidence makes no new model-quality claim: the production census overlaps
calibration inputs and is not an independent holdout or generalization estimate.

The detector produces image-space fiducial observations, not a camera-pose or
navigation-success claim. Pose accuracy additionally requires calibrated
intrinsics and known tag size/layout. The three real photographs do not prove
generalization to arbitrary cameras, lighting, motion blur, occlusion, or tag
families. The exact image was qualified for private delivery and this CPU
workflow, including complete-byte review with recorded finding dispositions.
Registry admission remains a maintainer decision; public image publication and
current-head CI readiness are not claimed by these workload results.

### robomimic

Pinned: `ARISE-Initiative/robomimic`
`d309eaecc18acf4152a830a895a6984b8ac71b05`. The runtime-only input is the
official Lift PH low-dimensional HDF5 at immutable dataset revision
`robomimic/robomimic_datasets@74fa018461f479cd9fd15b924a16103012096203`.

Phase A provides an unbuilt, quarantined neutral bootstrap candidate: pinned
source and Debian/bootstrap packages only. The historical 40-entry hash lock
is retained as verifier evidence and is not baked into this public bootstrap.
CUDA/PyTorch and other restricted dependencies are fetched at runtime into a
customer-owned, exact-inventory environment using the customer's vendor
entitlement. The official HDF5 is an immutable runtime fetch; pretrained
weights are unnecessary. No runtime/data bytes were fetched, no image was
built or published, and no B200 result is claimed. The public, anonymously
retrievable exact dataset carries its MIT notice and needs no separate NPA
acceptance. The external CUDA/cuDNN runtime requires a customer-facing notice
and a customer-created, unexpired authorization bound to the customer, run,
exact runtime lock and inventory, exact terms set, and expiry. A manager or
generic human signature is not a substitute. If gated Hugging Face or NGC
assets are selected later, use the customer's real vendor entitlement probe
without a duplicate NPA terms boolean. The runtime fetch verifies the fetched
environment and installed RECORD before use; `NPA_ROBOMIMIC_CUSTOMER_DENYLIST`
is an explicit customer runtime input, and when unset the built-in safe denylist
applies. It is not a publication prerequisite. Separate image, infrastructure,
dataset, and B200 transaction gates still apply.

| Capability | Status | Upstream basis |
| --- | --- | --- |
| `lift_ph_lowdim_bc_train` | pending live B200 acceptance | `robomimic/scripts/train.py`, BC, four serialized Adam steps |
| `lift_ph_lowdim_heldout_validate` | pending live B200 acceptance | upstream deterministic 90/10 HDF5 masks, disjoint validation loss |
| `lift_ph_lowdim_checkpoint_reload_action` | pending live B200 acceptance | `policy_from_checkpoint` plus finite held-out seven-dimensional action |

The candidate is deliberately low-dimensional and headless on exactly one B200.
It does not claim simulator success or convergence. The sole hard-gate artifact
is `robomimic-smoke.json`; see [the operator contract](byof-robomimic.md).

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

### Alibaba Wan 2.2 TI2V-5B

Official Alibaba generative-video baseline, pinned to
`Wan-Video/Wan2.2@42bf4cfaa384bc21833865abc2f9e6c0e67233dc` with the
official `Wan-AI/Wan2.2-TI2V-5B` checkpoint pinned to
`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`. Checkpoint and tokenizer files are
fetched at run time; they are not baked into the canonical `npa-wan2-2` image.
CUDA-enabled PyTorch and its `nvidia-*` closure are likewise installed only in
an operator-owned volume after explicit terms acceptance; the image contains
the pinned source and OSS CPU dependency base. The checked-in
single-GPU profile targets one RTX PRO 6000 Blackwell (`sm_120`) and the
upstream PyTorch SDPA fallback. The separate distributed profile requests four
B200s in one pod. `torch.distributed.run` launches an instrumentation wrapper
on the four ranks, and the wrapper executes pinned official `generate.py` as
`__main__` with `--dit_fsdp --t5_fsdp --ulysses_size 4`; the 24 attention heads divide evenly
across the four Ulysses ranks. The current distributed smoke fails unless the
Torch 2.13.0/CUDA 13.0 wheel contains `sm_100`, every observed device is compute
capability 10.0, NCCL 2.29.7 connects all four unique devices, both T5 and WanModel use
FULL_SHARD FSDP, and Ulysses performs real distributed attention/all-to-all
collectives during the shared generation.

The exact public-dev digest has current single-GPU evidence for the Torch
2.13.0/CUDA 13.0 closure. The prior Torch 2.7.1/CUDA 12.8/NCCL 2.27.7
distributed records remain historical only; they cannot qualify the current
four-GPU path or add a B200 claim to the release.

| Capability | Status | Upstream basis / NPA evidence |
| --- | --- | --- |
| `wan2.2_ti2v_5b_text_to_video` | accepted current evidence | exact digest: native `wan.WanTI2V.generate` at 1280x704 on RTX PRO 6000 Blackwell (`sm_120`) with Torch 2.13.0/CUDA 13.0 |
| `wan2.2_decoded_mp4_validation` | accepted current evidence | same exact-digest run: all 17 frames decoded at 24 fps; 2,807,385 bytes, spatial stddev 76.1516, pixel range 255, mean temporal delta 11.8930 |
| `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | accepted historical evidence | prior runtime: `torch.distributed.run` launched four wrapper ranks on one 4×B200 node; each wrapper executed pinned official `generate.py` as `__main__`; loaded NCCL 2.27.7, T5/DiT FULL_SHARD FSDP, Ulysses size 4 |
| `wan2.2_distributed_rank_topology_validation` | accepted historical evidence | same prior run: ranks/local ranks 0–3 mapped to four unique GPU hashes; each rank recorded NCCL sum 10/10, 480 Ulysses attention calls, 1,920 all-to-all calls, three barriers, the observed final barrier, and teardown |
| `wan2.2_decoded_mp4_validation` (distributed run) | accepted historical evidence | same prior run: all 17 H.264 frames decoded at 24 fps; 2,809,770 bytes, spatial stddev 71.9485, pixel range 255, mean temporal delta 9.714725, SHA-256 `9574f79c…94865` |
| `wan2.2_verified_rerun_recording` | accepted current single-GPU evidence | the current exact-digest output produced an uploaded RRD with local/remote parse and exact embedded-video identity; distributed RRD evidence remains historical |
| `wan2.2_ti2v_5b_image_to_video` | deferred | official unified-model capability and a real optional S3-image code path exist, but no separate live input/output evidence |
| `wan2.2_t2v_a14b` / `wan2.2_i2v_a14b` | deferred | separate MoE checkpoints and materially different GPU contract; not in this image gate |
| `wan2.2_s2v_14b` | deferred | separate speech/audio inputs and checkpoint |
| `wan2.2_animate_14b` | deferred | separate character-animation inputs and checkpoint |
| `wan2.2_fine_tuning` | deferred | pinned official source does not expose a TI2V training entrypoint |
| stock Wan action prediction | rejected | action prediction is not an upstream Wan 2.2 capability |

The single-GPU primary JSON is `wan2_2_ti2v_5b_text_to_video.json`. The
distributed workflow emits `wan2_2_ti2v_5b_multigpu.json`,
`wan2_2_multigpu_topology.json`, four per-rank JSON files,
`wan2_2_multigpu_runtime_inventory.json`, and
`wan2_2_ti2v_5b_multigpu.mp4`. The successful BYOF path then publishes
`wan2_2_ti2v_5b_multigpu.rrd` and its verified manifest. The recording embeds
the exact MP4 and exposes the real run evidence in the NPA agent's Rerun viewer.
The accepted current single-GPU run used the immutable zero-payload public-dev
digest; CUDA Python distributions and model/tokenizer bytes remained in the
operator-owned runtime volume. Historical distributed evidence does not qualify
the current release. A historical private image that baked CUDA Python
distributions remains excluded from publication. See
[`wan2.2.md`](wan2.2.md) for the workflow, RRD, licensing, and validation
contracts.

### Lightricks LTX-2.5

Audio-video DiT foundation model, pinned to
`Lightricks/LTX-2@fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` with the gated
`Lightricks/LTX-2.5` checkpoint set. The accepted public image is
`npa-ltx2:2.5-rtfetch-20260817`; its exact digest and prior real RTX PRO 6000
results are recorded in `npa/src/npa/deploy/ltx2_image_manifest.json`.
The 2026-09-05 registry audit matched those released bytes anonymously; it did
not rerun generation or establish additional capabilities.

LTX-2.5 differs from every other candidate in this catalog in what it licenses.
The LTX-2.x Community License Agreement (2026-08-11, not OSI) covers the
`ltx-core` / `ltx-pipelines` source as well as the weights, so the habitual
"bake the code, fetch the weights" split is not available: the image bakes
neither, and both fetches run under the operator's own `HF_TOKEN`. Acceptance
happens on Lightricks' gated Hugging Face repository, not here, and compliance
with the Agreement — including Attachment A(18), which forbids using Outputs to
train another machine learning model for commercial use, and a robot policy is
another machine learning model — is the operator's own responsibility.

| Capability | Status | Upstream basis / NPA evidence |
| --- | --- | --- |
| `ltx2_5_text_to_video` | accepted exact-digest evidence | pinned upstream distilled pipeline generated a real clip on one RTX PRO 6000; source and weights remained runtime fetches |
| `ltx2_5_decoded_mp4_validation` | accepted exact-digest evidence | independently decoded H.264 MP4: 1536×1024, 121 frames, 1,994,625 bytes; digest and refusal evidence in the accepted image manifest |
| `ltx2_5_image_to_video` | not claimed | upstream pipeline exists; no code path or evidence here |
| `ltx2_5_audio_to_video` | not claimed | separate `A2VidPipelineTwoStage` inputs |
| `ltx2_5_lora_fine_tuning` | not claimed | `ltx-trainer` is licensed material and training on Outputs is what Attachment A(18) restricts |

The primary JSON is `ltx2_5_text_to_video.json`. The run itself still proves the
refusal first (`ltx-runtime assert-refusal`: exit 78 and empty caches before any
fetch), but that is a property of the image rather than a graded capability. See
[`ltx2.md`](ltx2.md) for the dev-VM runbook, the entitlement the run requires,
and the exact-digest validation record. A future image must earn fresh evidence.

## First-class Workbench tools (not BYOF)

LeRobot is already a first-class `npa workbench lerobot` tool. Supported
package/image tags:

| Version | Status | Notes |
| --- | --- | --- |
| `0.5.1` | default | Accepted CUDA 13 timestamped image pin in `lerobot_version_manifest.json`; plain `0.5.1` is historical |
| `0.6.0` | validated additional package support | Accepted immutable `0.6.0-d6-extras-20260912` image and digest; lean diffusion + SmolVLA extras; real DiffusionPolicy construction and Blackwell validation passed on B200; `env_eval_freq` |

Select the package with `--lerobot-version`; serverless training accepts a
custom container through `train --image`, while VM deployment installs the
package. The failed 2026-09-05 anonymous lookup was resolved by the 2026-09-12
exact-digest publication. The immutable resolver tag and compatibility
`0.6.0` alias now resolve to the same accepted bytes; the optional version
remains separate from the default public release row.
Manifest:
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
  workflows/testing/byof-maniskill.yaml \
  --run-id byof-maniskill-smoke --json
```

The registry-ready gate is not satisfied until the live run pulls the pushed
image, executes the smoke command, and writes `npa_byof_summary.json`, smoke
logs, and the named capability artifact to object storage.


## GPU video BYOF workflows

These Tier 1 BYOF recipes use immutable Diffusers source and model revisions. They do not claim action-conditioned simulation, training, or real-time serving. See [build and run instructions](video-generation-byof.md).

| Model | Native API | Capability | Artifact | Workflow | Status |
| --- | --- | --- | --- | --- | --- |
| mochi-1 | `MochiPipeline` on CUDA | `mochi-1_text_to_video` | `mochi_1_text_to_video.json` + `video.mp4` | `byof-mochi-1.yaml` | public GHCR image; B200 inference verified (2026-09-16) |
| cogvideox-2b | `CogVideoXPipeline` on CUDA | `cogvideox-2b_text_to_video` | `cogvideox_2b_text_to_video.json` + `video.mp4` | `byof-cogvideox-2b.yaml` | public GHCR image; B200 inference verified (2026-09-16) |
| wan2.1-14b | `WanPipeline` on CUDA | `wan2.1-14b_text_to_video` | `wan2_1_14b_text_to_video.json` + `video.mp4` | `byof-wan2.1-14b.yaml` | public GHCR image; B200 inference verified (2026-09-16) |


### LingBot World v1 camera workflow

`byof-lingbot-world.yaml` exercises `lingbot_world_camera_conditioned_video` using pinned upstream `generate.py`, FSDP, and Ulysses on four B200 GPUs. It emits `lingbot_world_camera_conditioned_video.json`, per-rank execution evidence, authored camera controls, the source image, and a fully decoded MP4. Exact public GHCR image inference verified on B200 on 2026-09-16. Camera trajectories and approximate intrinsics are authored, not measured. This v1 recipe does not claim robot action input, training, real-time performance, or support for the separate World Infinity successor.

- `byof-depth-anything-v2.yaml`: `relative_depth_video` emits `depth_anything_v2_relative_depth.json`, raw prediction arrays and a fully decoded GPU-derived video. Exact public GHCR image inference verified on B200 on 2026-09-16. Model checkpoints are immutable runtime fetches; video inputs require a complete SHA256. No metric-depth, ground-truth-mask, or robot-success claim.

- `byof-sam2.1.yaml`: `prompted_video_mask_propagation` emits `sam2_1_video_mask_propagation.json`, raw prediction arrays and a fully decoded GPU-derived video. Exact public GHCR image inference verified on B200 on 2026-09-16. Model checkpoints are immutable runtime fetches; video inputs require a complete SHA256. No metric-depth, ground-truth-mask, or robot-success claim.

Qualification above used the exact public GHCR images pulled by GPU workers, native capability execution, successful worker summaries, complete artifact hash verification and independent full video decode. The recipes default to the accepted `npa-diffusers`, `npa-lingbot-world` and `npa-sam2` digests; they are Tier 1 workflows, without standalone serving APIs. [Public capability evidence](validation/studio-public-models-20260916.json) records the bounded results. Concrete infrastructure records remain private.
