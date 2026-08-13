---
name: open-dreamer
description: Use when onboarding or running the Open Dreamer (next-state/open-dreamer) JAX/Flax Dreamer 4 world-model pipeline as a multi-GPU BYOF registry candidate — causal video tokenizer, action-conditioned latent dynamics, and a real >=2 GPU data-parallel smoke.
---

# Open Dreamer (Dreamer 4 world model)

Open Dreamer (`next-state/open-dreamer`) is an open JAX/Flax NNX implementation
of the **Dreamer 4 world-model training pipeline**: a causal video tokenizer, an
action-conditioned latent dynamics model, rollout generation, and FVD scoring.
It is onboarded into NPA as a **multi-GPU BYOF registry candidate** and is the
reference example for the `>=2` GPU BYOF path.

Load `skills/workflows/byof-onboard/SKILL.md` for packaging/run mechanics and
`skills/workflows/oss-solution-registry-onboard/SKILL.md` for the registry
admission contract. For onboarding a **new** world model, load
`skills/workflows/onboard-world-model/SKILL.md` — the generic playbook for which
this skill is the worked reference example. This skill records the
solution-specific facts.

## When To Use

- Build/push and run the Open Dreamer registry image via BYOF `solution-smoke`.
- Author or debug a **multi-GPU** (2-GPU minimum) world-model BYOF workflow.
- Understand how to exercise the Dreamer 4 tokenizer/dynamics on synthetic data
  without staging the full Minecraft/VPT dataset.

## Upstream Facts

- Repo: `https://github.com/next-state/open-dreamer` · pinned ref `2b10640`.
- Stack: Python **3.11**, `uv`, CUDA-12 `jax[cuda12]` + Flax NNX, Hydra/OmegaConf
  configs, Grain/ArrayRecord data. Install via `uv sync`.
- Documented entrypoints (`scripts/`): `train_tokenizer.py`,
  `tokenize_minecraft_dataset.py`, `train_dynamics.py`, `eval_fvd.py`.
- Multi-GPU: `dreamer.parallel.build_parallel(strategy)` builds a JAX device
  mesh over `jax.devices()` for `data` / `fsdp` / `tp` / `sp`. `train_tokenizer`
  forces `data` parallelism; `train_dynamics` uses `parallel_strategy`.
- Datasets: raw Minecraft/VPT MP4 ArrayRecords (`minecraft_vpt`), a self-
  contained procgen `coinrun` path, and pre-tokenized `latent` ArrayRecords.

## NPA Mapping

| Item | Path |
| --- | --- |
| Workflow spec | `npa/workflows/workbench/npa-workflows/byof-open-dreamer.yaml` |
| 2-GPU resource profile | `npa/src/npa/workflows/byof/profiles/byof-solution-smoke-rtxpro-2gpu.yaml` |
| BYOF toolRef | `workbench.byof.repo` (`--workload solution-smoke`) |
| Catalog entry | `docs/workbench/oss-solution-catalog.md` |

The workflow uses `RTXPRO-6000-BLACKWELL-SERVER-EDITION:2`, not the single-GPU
profile, because the accepted capability requires a genuine 2-GPU mesh.

## Capabilities

Hard gates (all must pass; the driver `raise SystemExit`s if any is missing from
`capabilities_exercised`, so a green smoke means the dream actually ran):

- `jax_two_gpu_data_parallel_mesh` — `build_parallel("data")` yields
  `{data: 2, model: 1}` over `>=2` `jax.devices()`. Fails on a single GPU.
- `dreamer4_tokenizer_train_two_gpu` — real `scripts/train_tokenizer.py` trains
  the causal video tokenizer sharded data-parallel across the mesh to
  **legibility** (lower `mae_p_max`, more steps, `lpips_weight=0`).
- `dreamer4_action_conditioned_dream_rollout` — the marquee payoff:
  `dreamer.sampler.sample_video` gives the dynamics model context frames + future
  actions and dreams future gameplay; reports dream PSNR against ground truth.
  Gating it transitively gates the whole loop (dataloader → tokenizer → latent →
  dynamics → dream).

Also exercised (the rest of the Dreamer 4 loop):

- `minecraft_vpt_video_dataloader` — real `dreamer.data.build_iterator`
  `minecraft_vpt` MP4 path (decord decode + VPT action parse) with device
  sharding.
- `dreamer4_latent_tokenization` — `scripts/tokenize_minecraft_dataset.py`
  encodes the episodes with the trained tokenizer into **real** latent
  ArrayRecords + `latent_stats` (mean/std overrides), carrying the real
  27-binary / 121-categorical VPT actions.
- `dreamer4_dynamics_train_two_gpu` — `scripts/train_dynamics.py`
  action-conditioned latent dynamics trained on those Minecraft latents (the
  core Dreamer world-model loop).
- `world_model_rerun_visualization` — emits `open_dreamer_world_model.rrd` with
  synchronized `world/observation` (GT), `world/dream` (predicted),
  `world/gt_decoded` (tokenizer ceiling), and `world/tokenizer_reconstruction`
  streams, loadable into the NPA agent's Rerun viewer.

The data is a **real Minecraft/VPT** contractor-gameplay subset (OpenAI VPT
`.mp4` + `.jsonl`), center-cropped and resized to 128x128, staged as
`minecraft_vpt` ArrayRecords to the run bucket under
`datasets/minecraft_vpt_128_64/` and pulled at run time. This is a real,
multi-stage GPU run on real gameplay — not an import-only or synthetic smoke.

### View / share the visualization in the agent Rerun

The `.rrd` is uploaded to the run's S3 output prefix. Load it into a live NPA
agent's Rerun viewer:

```bash
curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -X POST https://<agent-ip>/api/sim-viz/load-artifact \
  -H 'content-type: application/json' \
  -d '{"run_id":"<run-id>","s3_uri": "s3://<bucket>/byof/<run-id>/open_dreamer_world_model.rrd"}'
```

Then open `https://<agent-ip>/rerun/` (see `skills/tools/npa-agent/SKILL.md`).
The agent also discovers it via the artifact browser (`what can I view?`).

## Data Contract (real Minecraft/VPT)

- The smoke trains on a real **Minecraft/VPT** subset. Stage it once (dev VM has
  egress): read a VPT index (`.../snapshots/all_8xx_Jun_29.json`, etc.), download
  each `{basedir}{relpath}.mp4` + `.jsonl`, center-crop to square, resize to
  128x128, slice fixed-length clips, and write **minecraft_vpt** records —
  pickled `{"video": mp4_bytes, "video_shape": (T,128,128,3), "actions": [VPT
  action dicts], "source": relpath}` — as `shard-*.array_record`. Upload to the
  run bucket under `datasets/minecraft_vpt_128_64/`; the smoke derives the
  bucket from `S3_OUTPUT_PREFIX` and pulls it at run time (system `python3` has
  boto3; the uv venv does not, so download in the bash prelude).
- `ProcessMinecraftEpisodeAndSlice` decodes the MP4 with **decord** and
  `parse_action_dicts` turns the VPT dicts into the 27-binary / 121-categorical
  action layout `train_dynamics.py` asserts (`NUM_BINARY_ACTIONS`,
  `NUM_CAMERA_CLASSES`). `tokenize_minecraft_dataset.py` writes the **latent**
  records (`serialize_msgpack_record`, `{"latents": float32(T, n_latents,
  d_bottleneck), "actions": {...}}`) + `metadata/latent_stats.npz`.
- Override `dataset.H=128 dataset.W=128 dataset.padding_H=[0,0]
  dataset.padding_W=[0,0] dataset.patch_size=16` for the 128x128 subset.

## Run It

```bash
export NPA_E2E_PROJECT=rtxpro
npa/.venv/bin/python npa/scripts/run_byof_repo.py \
  --repo-url https://github.com/next-state/open-dreamer.git \
  --repo-ref 2b10640 \
  --base-profile ubuntu \
  --project rtxpro \
  --workload solution-smoke \
  --yaml npa/src/npa/workflows/byof/profiles/byof-solution-smoke-rtxpro-2gpu.yaml \
  --build-command '<from byof-open-dreamer.yaml config.build_command>' \
  --smoke-command '<from byof-open-dreamer.yaml config.smoke_command>' \
  --solution-name open-dreamer \
  --capability-name dreamer4_tokenizer_train_two_gpu \
  --smoke-artifact-name open_dreamer_world_model_2gpu.json \
  --run-id byof-open-dreamer-2gpu-<stamp> \
  --cleanup
```

`build_command` and `smoke_command` are the source of truth in
`byof-open-dreamer.yaml`; pass them verbatim (the SkyPilot 2-GPU profile is
selected with `--yaml`).

## Validate

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-open-dreamer.yaml --json
npa/.venv/bin/python -m pytest npa/tests/workflows/test_byof_solution_smokes.py -q
```

### Live real-GPU e2e (the spec is the tested artifact)

`npa/tests/e2e/test_byof_open_dreamer_live_e2e.py` drives the real
`byof-open-dreamer.yaml` (its `smoke_command` + 2-GPU resource profile, read
straight from the YAML) through the canonical BYOF runner on live GPUs, then
verifies via S3 that all 7 capabilities were exercised with 0 deferred and that
the dream `.rrd` was produced. Reuse a prebuilt image so the test does not
rebuild:

```bash
NPA_INTEGRATION_E2E=1 NPA_BYOF_OPEN_DREAMER_LIVE_GPU=1 \
NPA_BYOF_TEST_IMAGE=<registry>/npa-byof:open-dreamer-<ref> \
NPA_E2E_PROJECT=rtxpro NPA_NEBIUS_PROFILE=npa-mk8s \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_byof_open_dreamer_live_e2e.py -q
```

Without `NPA_BYOF_OPEN_DREAMER_LIVE_GPU=1` the multi-hour 2-GPU case skips; the
non-GPU render check (`test_open_dreamer_spec_renders_via_workflow_machinery`)
still asserts the spec plans/renders through the real npa.workflow machinery.
The BYOF *submit* path is intentionally plan-only (its outer K8s pod cannot build
the image); the runner path is the real GPU path.

## Gotchas

- Use a **uv-managed** Python 3.11 installed to a world-readable dir
  (`UV_PYTHON_INSTALL_DIR=/opt/uv/python`), create the venv there, and
  `chmod -R a+rX` it — the default uv dir under `/root` is unreadable by the
  non-root runtime `ubuntu` user. Use `/opt/byof/.venv/bin/python` directly at
  run time, not `uv run` (it would try to re-sync/write the root-owned venv).
- `decord` (MP4 decode) and `imageio[ffmpeg]` are in the image via `uv sync`;
  `jaxlpips` is a hard import in `scripts/train_tokenizer.py` (present via
  `uv sync`) but keep `lpips_weight=0` so no Hugging Face LPIPS weights download
  at run time. `boto3` is only in the system `python3` (profile setup), not the
  uv venv, so pull the dataset in the bash prelude, not the venv driver.
- Dynamics asserts `num_binary_actions==27` and `categorical_action_dim==121`;
  the `minecraft_vpt` path satisfies this via `parse_action_dicts` on the VPT
  action dicts. Keep the dream rollout context `<=` `dynamics.context_length`
  (trim frames to `min(T, context_length)`) or the KV cache overflows at
  prefill. `packing_factor` must divide the tokenizer `n_latents`.
- The smoke ships a tiny CPU profile (`OD_MODE=cpu`, `OD_NUM_WORKERS=0`,
  `OD_LOCAL_DATASET=<dir>`) that runs the whole chain on one CPU device for
  offline validation against a small local `minecraft_vpt` sample; the container
  uses the full 2-GPU profile and the S3-staged dataset by default.
- Dream fidelity scales with the tokenizer/dynamics budget
  (`OD_TOK_STEPS`/`OD_DYN_STEPS`; upstream trains ~200k). FVD/I3D scoring
  (`eval_fvd.py`, needs I3D weights) remains a follow-up.
- Batch size `B` must be divisible by the number of devices (mesh `data` axis)
  and by `jax.process_count()`.
- `rerun-sdk==0.31.4` (matches the agent viewer) is installed at build time so
  the `.rrd` visualization can be produced and loaded live.
