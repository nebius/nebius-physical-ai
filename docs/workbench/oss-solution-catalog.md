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
| OpenPI | `Physical-Intelligence/openpi` `15a9616a…` | connected direct / cross-pod serve / LoRA optimizer / held-out evaluation gate | builder `openpi_pi05_droid_jointpos_polaris_inference.json`, then four mode-specific JSON reports plus exact checkpoint manifest | `byof-openpi.yaml` → `openpi-pi05-four-mode.yaml` |
| DROID policy learning | `droid-dataset/droid_policy_learning` `9a29c832…` | `rlds_config_generator_contract` | `droid_rlds_config_generator.json` | `byof-droid-policy-learning.yaml` |
| Open Dreamer (world model, **2-GPU min**) | `next-state/open-dreamer` `2b10640` | `dreamer4_tokenizer_train_two_gpu` | `open_dreamer_world_model_2gpu.json` | `byof-open-dreamer.yaml` |
| Alibaba Wan 2.2 TI2V-5B | `Wan-Video/Wan2.2` `42bf4cf…` | `wan2.2_ti2v_5b_text_to_video` | capability JSON + runtime inventory + MP4 | `byof-wan2.2.yaml` |
| Lightricks LTX-2.5 (**not built; licence-gated**) | `Lightricks/LTX-2` `fd4ded7f…` | `ltx2_5_text_to_video` | `ltx2_5_text_to_video.json` + provenance manifest + MP4 | `byof-ltx2.yaml` |
| Alibaba Wan 2.2 TI2V-5B (**4-GPU distributed**) | same pinned source/checkpoint | `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | multi-GPU capability JSON + rank topology + runtime inventory + MP4 | `byof-wan2.2-multigpu.yaml` |

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
| OpenPI | `pi05_droid_jointpos_polaris_checkpoint_download` | **accepted** | Canonical isolated B200 gate: image build/push/digest verification, then 12,434,530,837 runtime-only GCS bytes with 27-object generation-manifest provenance; exact scoped `NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` is runtime-only |
| OpenPI | `pi05_droid_jointpos_polaris_direct_infer` | **accepted** | Same digest-pinned B200 `sm_100` gate; deterministic Franka input produced finite `float64[15,8]` joint-position targets |
| OpenPI | `pi05_droid_jointpos_polaris_served_infer` | **accepted builder regression** | Same gate; upstream WebSocket health + same-pod client round trip produced finite `float64[15,8]` |
| OpenPI | `pi05_droid_jointpos_polaris_cross_pod_serve` | **accepted** | Isolated single-B200 connected gate: private ClusterIP, ready digest-pinned server Deployment, and a distinct CPU client pod completed two finite `float64[15,8]` requests; exact service cleanup passed |
| OpenPI | `pi05_droid_jointpos_polaris_lora_optimizer_smoke` | **accepted** | Same connected gate: upstream pi0.5 LoRA forward/backward/AdamW step, finite loss, changed trainable-state hash, and independently reloadable private Orbax checkpoint |
| OpenPI | `pi05_droid_jointpos_polaris_heldout_evaluate` | **accepted** | Same connected gate: exact trained-checkpoint reload, two samples excluded from the four-sample training split, finite upstream loss and action MAE/MSE, and finite `float64[15,8]` trajectory |
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
| Wan 2.2 TI2V-5B | `wan2.2_ti2v_5b_text_to_video` | **accepted historical evidence** | prior Torch 2.7.1/CUDA 12.8 runtime: native TI2V-5B generation on RTX PRO 6000 Blackwell (`sm_120`); current Torch 2.13.0/CUDA 13.0 gate is not yet live-qualified |
| Wan 2.2 TI2V-5B | `wan2.2_decoded_mp4_validation` | **accepted historical evidence** | same prior run: 2,923,858-byte H.264 MP4, 1280x704, 17 frames at 24 fps; full decode and non-uniform-content gates passed |
| Wan 2.2 TI2V-5B | `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | **accepted historical evidence** | prior Torch 2.7.1/CUDA 12.8/NCCL 2.27.7 runtime: one node, 4×B200 (`sm_100`), world size 4, T5/DiT FULL_SHARD FSDP, Ulysses size 4, and official `generate.py`; current NCCL 2.29.7 gate is not yet live-qualified |
| Wan 2.2 TI2V-5B | `wan2.2_distributed_rank_topology_validation` | **accepted historical evidence** | same prior run: four unique GPU hashes/ranks 0–3, NCCL sum 10/10 per rank, 480 distributed-attention and 1,920 all-to-all calls per rank, three barriers, final barrier, and process-group teardown |
| Wan 2.2 TI2V-5B | `wan2.2_decoded_mp4_validation` (distributed run) | **accepted historical evidence** | same prior run: 2,809,770-byte H.264 MP4, 1280x704, 17 frames at 24 fps; spatial stddev 71.9485, pixel range 255, temporal delta 9.714725, SHA-256 `9574f79c…94865` |

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
| `pi05_droid_jointpos_polaris_checkpoint_download` | accepted (live) | canonical build/push/digest gate; anonymous runtime `download.maybe_download(gs://openpi-assets/checkpoints/polaris/…)`; 27 objects / 12,434,530,837 bytes; weights and the exact scoped terms acceptance are never baked |
| `pi05_droid_jointpos_polaris_direct_infer` | accepted (live) | digest-pinned B200 `sm_100` `get_config("pi05_droid_jointpos_polaris")` + direct `policy.infer`; finite `float64[15,8]` joint-position targets |
| `pi05_droid_jointpos_polaris_served_infer` | accepted builder regression (live) | upstream `WebsocketPolicyServer` + same-pod `WebsocketClientPolicy`; served finite `float64[15,8]` |
| `pi05_droid_jointpos_polaris_cross_pod_serve` | accepted (live) | upstream server Deployment + private ClusterIP + distinct client Job; two finite `float64[15,8]` requests (39.350 s cold, 50.2 ms warm); exact cleanup |
| `pi05_droid_jointpos_polaris_lora_optimizer_smoke` | accepted (live) | supported upstream pi0.5 LoRA config; one real forward/backward/AdamW update (loss 0.145676, update L2 0.0957375), changed trainable state, and reloadable 29-file Orbax checkpoint |
| `pi05_droid_jointpos_polaris_heldout_evaluate` | accepted (live) | exact trained-checkpoint reload; two disjoint held-out samples; finite mean upstream loss 0.182892, action MAE 0.0111408 and MSE 0.000200538, plus valid `float64[15,8]` trajectory |

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

The accepted live records below are historical evidence from the prior Torch
2.7.1/CUDA 12.8/NCCL 2.27.7 runtime. They remain evidence that the capabilities
ran, but the current postprocessor intentionally rejects them as acceptance
inputs for the Torch 2.13.0/CUDA 13.0/NCCL 2.29.7 closure. That closure requires
a new operator-accepted single- and four-GPU qualification before publication.

| Capability | Status | Upstream basis / NPA evidence |
| --- | --- | --- |
| `wan2.2_ti2v_5b_text_to_video` | accepted historical evidence | prior runtime: native `wan.WanTI2V.generate` at 1280x704 on RTX PRO 6000 Blackwell (`sm_120`) |
| `wan2.2_decoded_mp4_validation` | accepted historical evidence | same prior run: all 17 frames decoded at 24 fps; 2,923,858 bytes, spatial stddev 78.0124, pixel range 255, mean temporal delta 11.7294 |
| `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | accepted historical evidence | prior runtime: `torch.distributed.run` launched four wrapper ranks on one 4×B200 node; each wrapper executed pinned official `generate.py` as `__main__`; loaded NCCL 2.27.7, T5/DiT FULL_SHARD FSDP, Ulysses size 4 |
| `wan2.2_distributed_rank_topology_validation` | accepted historical evidence | same prior run: ranks/local ranks 0–3 mapped to four unique GPU hashes; each rank recorded NCCL sum 10/10, 480 Ulysses attention calls, 1,920 all-to-all calls, three barriers, the observed final barrier, and teardown |
| `wan2.2_decoded_mp4_validation` (distributed run) | accepted historical evidence | same prior run: all 17 H.264 frames decoded at 24 fps; 2,809,770 bytes, spatial stddev 71.9485, pixel range 255, mean temporal delta 9.714725, SHA-256 `9574f79c…94865` |
| `wan2.2_verified_rerun_recording` | accepted historical evidence | prior four-GPU evidence produced a 2,948,508-byte RRD (`5a4f7746…0606`) with local/remote parse and identity checks; the 3,045,269-byte single-GPU RRD (`49a57f5b…fb10`) was also served byte-identically and visibly rendered by the live agent |
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
The historical accepted single- and distributed runs used the same immutable
runtime-fetch candidate; CUDA Python distributions and model/tokenizer bytes remain in
operator-owned runtime volumes. A historical private image that baked CUDA
Python distributions remains excluded from publication. Live capability results
do not by themselves authorize public image publication. See
[`wan2.2.md`](wan2.2.md) for the workflow, RRD, licensing, and validation
contracts.

### Lightricks LTX-2.5

Audio-video DiT foundation model, pinned to
`Lightricks/LTX-2@fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` with the gated
`Lightricks/LTX-2.5` checkpoint set. **No status here is live**: the `npa-ltx2`
image has not been built and nothing has run on a GPU. Every row below is
therefore a declared contract awaiting evidence, not a result.

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
| `ltx2_5_text_to_video` | declared; no image built | `python -m ltx_pipelines.distilled` at the pinned ref, per upstream's own quick start |
| `ltx2_5_decoded_mp4_validation` | declared; no image built | `npa/src/npa/workbench/ltx2/video_check.py`, unit-tested against real ffmpeg clips and copied into the image |
| `ltx2_5_image_to_video` | not claimed | upstream pipeline exists; no code path or evidence here |
| `ltx2_5_audio_to_video` | not claimed | separate `A2VidPipelineTwoStage` inputs |
| `ltx2_5_lora_fine_tuning` | not claimed | `ltx-trainer` is licensed material and training on Outputs is what Attachment A(18) restricts |

The primary JSON is `ltx2_5_text_to_video.json`. The run itself still proves the
refusal first (`ltx-runtime assert-refusal`: exit 78 and empty caches before any
fetch), but that is a property of the image rather than a graded capability. See
[`ltx2.md`](ltx2.md) for the dev-VM runbook, the entitlement the run requires,
and what has to happen before any of the above may be marked live.

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
