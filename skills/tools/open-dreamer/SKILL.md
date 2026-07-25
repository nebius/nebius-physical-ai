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
admission contract. This skill records the solution-specific facts.

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

Hard gates (both must pass for registry admission):

- `jax_two_gpu_data_parallel_mesh` — `build_parallel("data")` yields
  `{data: 2, model: 1}` over `>=2` `jax.devices()`. Fails on a single GPU.
- `dreamer4_tokenizer_train_two_gpu` — real `scripts/train_tokenizer.py` trains
  the causal video tokenizer sharded data-parallel across the mesh to
  **legibility** (lower `mae_p_max`, more steps, `lpips_weight=0`).

Also exercised (the full Dreamer 4 loop, headlined by the dream rollout):

- `coinrun_video_dataloader` — real `dreamer.data.build_iterator` CoinRun path
  with device sharding.
- `dreamer4_latent_tokenization` — encode the SAME procedural videos with the
  trained tokenizer into **real** latent ArrayRecords + `latent_stats` (mean/std
  overrides), attaching 27-binary / 121-categorical actions derived from the
  known agent motion. This replaces random latents so dynamics can learn.
- `dreamer4_dynamics_train_two_gpu` — `scripts/train_dynamics.py`
  action-conditioned latent dynamics trained on those real latents (the core
  Dreamer world-model loop).
- `dreamer4_action_conditioned_dream_rollout` — `dreamer.sampler.sample_video`
  gives the dynamics model context frames + future actions and dreams the
  future; reports dream PSNR against the ground truth.
- `world_model_rerun_visualization` — emits `open_dreamer_world_model.rrd` with
  synchronized `world/observation` (GT), `world/dream` (predicted),
  `world/gt_decoded` (tokenizer ceiling), and `world/tokenizer_reconstruction`
  streams, loadable into the NPA agent's Rerun viewer.

The data is an **action-conditioned sprite-navigation** environment: the
per-frame action selects the agent's velocity, so future frames are genuinely
determined by future actions (a distractor sprite + textured background add
occlusion/richness). This is a real, multi-stage GPU run — not an import-only
or under-trained smoke.

### View / share the visualization in the agent Rerun

The `.rrd` is uploaded to the run's S3 output prefix. Load it into a live NPA
agent's Rerun viewer:

```bash
curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -X POST https://<agent-ip>/api/sim-viz/load-artifact \
  -H 'content-type: application/json' \
  -d '{"s3_uri": "s3://<bucket>/byof/<run-id>/open_dreamer_world_model.rrd"}'
```

Then open `https://<agent-ip>/rerun/` (see `skills/tools/npa-agent/SKILL.md`).
The agent also discovers it via the artifact browser (`what can I view?`).

## Synthetic Data Contract (no external dataset needed)

- **CoinRun** records are written as raw `pickle` bytes — the format
  `EpisodeLengthFilter(format_hint="coinrun")` and `ProcessEpisodeAndSlice`
  read. Do **not** use `ShardWriter` here: it serializes msgpack, which the
  CoinRun reader path does not decode. Each record:
  `{"raw_video": uint8(T,64,64,3).tobytes(), "sequence_length": T,
  "actions": int(T,), "rewards": float(T,)}`.
- **Latent** records use `dreamer.data.serialization.serialize_msgpack_record`
  with `{"latents": float32(T, n_latents, d_bottleneck), "actions": {"binary":
  int32(T,27), "categorical": int32(T,), "continuous": None}}`. The 27-binary /
  121-categorical action layout is asserted by `train_dynamics.py`
  (`NUM_BINARY_ACTIONS`, `NUM_CAMERA_CLASSES`).

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

## Gotchas

- Install a **system** `python3.11` (deadsnakes) and `uv venv --python
  /usr/bin/python3.11`. A uv-managed interpreter lives under `/root` and is not
  readable by the non-root runtime `ubuntu` user, so `.venv/bin/python` would
  fail at run time.
- Use `/opt/byof/.venv/bin/python` directly at run time, not `uv run` — the venv
  is root-owned from build and `uv run` would try to re-sync/write it.
- `jaxlpips` is a hard import in `scripts/train_tokenizer.py`, so it must be in
  the image (it is, via `uv sync`), but keep `lpips_weight=0` so no Hugging Face
  LPIPS weights are downloaded at runtime. Tokenizer legibility then comes from
  a lower `mae_p_max` (~0.3) plus more steps, not LPIPS.
- Dynamics asserts `num_binary_actions==27` and `categorical_action_dim==121`,
  so every latent record (training and rollout) must carry that action layout —
  derive it from the known agent motion, don't feed CoinRun's 16-way action.
- `packing_factor` must divide the tokenizer `n_latents`; the dream `horizon` +
  context must be `<=` the episode length `T`.
- The smoke ships a tiny CPU profile (`OD_MODE=cpu`, `OD_NUM_WORKERS=0`) that
  runs the whole chain on one CPU device in ~2 min for offline validation; the
  container uses the full 2-GPU profile by default.
- Batch size `B` must be divisible by the number of devices (mesh `data` axis)
  and by `jax.process_count()`.
- Full CoinRun generation needs `procgen`+`gym3` (no Python 3.11 wheels); the
  run synthesizes correctly-formatted **coherent, action-conditioned** records
  instead. Real Minecraft/VPT data, LPIPS finetuning, and FVD/I3D scoring
  (`eval_fvd.py`) remain follow-ups.
- `rerun-sdk` is installed at build time (and a `pip install --user` fallback at
  run time for reused images) so the `.rrd` visualization can be produced.
