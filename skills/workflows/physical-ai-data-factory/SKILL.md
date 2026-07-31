---
name: physical-ai-data-factory
description: Use when authoring, running, submitting, or viewing the NVIDIA Physical AI Data Factory blueprint on Nebius + SkyPilot (no OSMO) — annotate → Cosmos Transfer augment → evaluate/validate gate → re-label → FiftyOne curate → Rerun visualize — implemented as an npa.workflow that composes existing workbench tools.
---

# Physical AI Data Factory (NPA-native, no OSMO)

## Source And Attribution

NPA-native re-implementation of the NVIDIA Physical AI Data Factory / Video Data
Augmentation workflow. Design adapted from NVIDIA agent skills
(https://github.com/NVIDIA/skills), primarily `physical-ai-video-data-augmentation`.
Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. Upstream licenses:
Apache-2.0 and CC-BY-4.0. See `skills/NOTICE-NVIDIA-SKILLS`. NPA orchestrates on
SkyPilot (not OSMO) and composes existing workbench tools.

## When To Use

Load this skill when the user wants to author, validate, submit, run, or view the
`physical-ai-data-factory.yaml` blueprint, adapt it to a new dataset, run it on
GPUs, or troubleshoot why a run's Rerun panel / augmented output looks wrong.

Do NOT invent an `npa workbench data-factory` tool — there is none. The blueprint
is pure composition of existing toolRefs; only add real tools with tests.

## What It Is

`npa/workflows/physical-ai-data-factory.yaml` — one
`npa.workflow/v0.0.1` spec. Blueprint → NPA stage mapping:

| NVIDIA stage | NPA state | Tool (all REAL — no stubs) | Runtime |
| --- | --- | --- | --- |
| Config Generation | `generate-configs` | `data_factory_stages.generate_configs` (run.shell) | CPU |
| Understand & Annotate | `annotate-original` | `workbench.token_factory.caption` | Token Factory (zero-GPU) |
| Augment & Multiply | `augment` | `workbench.cosmos2.transfer_execute` (real Cosmos Transfer 2.5 `--execute`; uploads video+frames to S3) | GPU |
| Evaluate & Validate | `grade` loop (`attribute-verify` + `quality-gate`) | `workbench.vlm_eval.run` + `data_factory_stages.grade_gate` (reads the real VLM score) | Token Factory + CPU |
| Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Curation | `curate` | `data_factory_stages.curate` (real dataset report) | CPU |
| Visualize | `visualize` | `data_factory_viz.build_run_rrd` → `reports/sim2real.rrd` | CPU |
| Finalize | `finalize` | `data_factory_stages.finalize` (real aggregate report) | CPU |

Every stage invokes a real component (enforced by `test_real_components.py` and
the `real-components` skill). The `augment` stage runs the real Cosmos Transfer
2.5 model on GPU via `--execute` and publishes the generated video + extracted
frames to `augment_uri`, which the grade / re-label / visualize stages consume.

**Config → augment MULTIPLY.** `generate-configs` samples N appearance combos
(from `config.n_augmentations`); the `augment` toolRef passes `--configs-uri`, and
the augment stage runs **one real Cosmos Transfer 2.5 inference per sampled combo**
— each combo's prompt drives a distinct appearance, and each is published as its
own per-clip dir under `cosmos_augmented/<clip>/` with its own `metadata.json`
`variables` (which drives that clip's Rerun label). So an N-augmentation config
yields **N scenario variants**, not one image. The fan-out is surfaced in the
machine-readable artifacts: `variant_count` / `multiply_mode` / `variant_parallelism`
in the augment run-level `manifest.json`, `multiply` (mode + `variant_count`) in the
curation report, and `multiply_mode` / `variant_count` in the finalize report. (A
config with a single combo still emits one variant — `multiply_mode: single-variant`.)

**Multi-GPU fan-out (use ≥4 GPUs).** The multiply loop fans the N GPU-bound
diffusions **across the augment pod's GPUs**, one variant per GPU (pinned via
`CUDA_VISIBLE_DEVICES`), then publishes sequentially in combo order. Concurrency =
`NPA_COSMOS_VARIANT_PARALLELISM` if set, else the auto-detected visible-GPU count,
capped at the variant count (so it is safe on 1 GPU and never pins a variant to a
GPU the pod lacks). Request the GPUs in the spec: `resources.gpu.accelerators:
RTXPRO6000:4` runs 4 variants at once (~one variant's wall-clock instead of 4×).
Verified live: a 4-variant run on `RTXPRO6000:4` drove all 4 GPUs to 100%
(4 distinct compute PIDs) and finished in ~14 min end-to-end; the manifest recorded
`variant_parallelism: 4`.

**Authoring from chat (agent).** The NPA chat agent can WRITE this blueprint. Ask
it e.g. *"write me a paidf workflow: augment my robot clips and fan out 4 scenarios
on at least 4 RTX 6000 PRO GPUs"* — the deterministic router classifies
`create_data_factory_workflow`, `agent_workflow.choose_workflow_template` selects the
`physical-ai-data-factory` template, and `extract_data_factory_params` parses the
fan-out count → `config.n_augmentations`, the GPU count → `resources.gpu.accelerators`
+ `config.variant_parallelism` (capped to the GPU count), and the free-form
augmentation subject → `config.augment_subject`. The generated YAML is validated +
planned before it is returned (chat only emits runnable specs). `generate_data_factory_yaml(user_text=...)`
is the direct entry point; `generate_workflow_draft(intent="create_data_factory_workflow", user_text=...)`
is the chat path.

**Input conditioning (real augmentation of the caller's clip).** By default the
augment renders the bundled, self-contained control example (`robot_depth_spec.json`),
which keeps the golden eval hermetic but is NOT an augmentation of the run's own
input. To make the output a genuine transform of the run's real footage, opt in:
set `NPA_COSMOS_CONDITION_ON_INPUT=1` at submit (or pass `--condition-on-input` /
`--input-video <path|s3://>` to `npa workbench cosmos2 transfer`). The augment then
downloads the first clip under `--input-uri` (the run's `input/`), builds a
controlnet spec with `video_path` = that clip and an **`edge`** (or `vis`) control
computed on-the-fly, and the sampled appearance prompt drives the new look — so the
output preserves the input's structure/motion with a new appearance. `edge`/`vis`
need no precomputed control asset; `depth`/`seg` would need one, so input-only
conditioning falls back to `edge`. Conditioned runs record `mode:
cosmos_transfer2.5_gpu` + `input_conditioned: true` + `conditioned_input` in the
augment `metadata.json` / `manifest.json`, which the agent's provenance panel surfaces.

**Naming caveat:** the `attribute-verify` stage runs the REAL `vlm_eval` tool
with `--backend api`; its output file is `vlm_eval_stub.json`, a LEGACY filename
of the vlm_eval tool (`RESULT_FILENAME`), not a stubbed stage. `grade_gate`
imports that constant instead of hardcoding the string, so it stays in sync.

Verified Token Factory model roles: `Qwen/Qwen2.5-VL-72B-Instruct` (VLM),
`meta-llama/Llama-3.3-70B-Instruct` (LLM), `nvidia/Cosmos3-Super-Reasoner`
(Cosmos-family critic). Cosmos Transfer 2.5 is the GPU augment engine, not a
Token Factory model.

## Commands

```bash
SPEC=npa/workflows/physical-ai-data-factory.yaml
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec   "$SPEC" --run-id demo --assume-decision promote_checkpoint --json
# Render/submit on GPUs (needs NPA_SRC_S3_URI or --image, and secret-envs):
npa workbench workflow submit "$SPEC" --run-id "$(date -u +paidf-%Y%m%dt%H%M%sz)" \
  --assume-decision promote_checkpoint --var bucket=<bucket> \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN
```

## Key Operational Notes

- **GPU accelerator name is cluster-specific.** The spec uses canonical
  `RTXPRO6000:1`; some clusters advertise `RTXPRO-6000-BLACKWELL-SERVER-EDITION`.
  If `sky` reports `FAILED_PRECHECKS` / no matching resources, check
  `sky gpus list` and resubmit with the cluster's accelerator name.
- **Reproducible deploy/submit:** unset a stale `NEBIUS_IAM_TOKEN` before
  `sky`/`terraform` (the Nebius provider prefers the ambient token over the
  fresh one). `provisioner._run` scrubs it for agent deploys.
- **Augment output contract (per-clip layout):** `cosmos2.transfer` writes a
  contract manifest by default; with `--execute` on S3 output,
  `publish_transfer_to_s3` uploads the real Cosmos Transfer 2.5 result in the
  **per-clip** layout the consumers require:
  `cosmos_augmented/<clip>/{augmented_video.mp4, frame-*.png, metadata.json}`
  plus a run-level `cosmos_augmented/manifest.json`.   `curate` counts clip
  subdirs (not top-level files) and `build_run_rrd` reads each clip's
  `metadata.json` for its Rerun label. Producer and consumers share this shape;
- **Real FiftyOne curation (Voxel51):** the `curate` stage runs *real* FiftyOne
  Brain curation over the augmented variants when FiftyOne is importable (i.e. the
  stage runs in the `npa-fiftyone` image): `data_factory_stages.curate` delegates
  to `data_factory_curate.run_curation`, which builds a `fiftyone.Dataset`,
  computes a GPU-free per-variant embedding (downsampled RGB + color histogram),
  and runs `compute_uniqueness` + `compute_similarity().find_duplicates()` +
  `compute_visualization(method="pca")`. The report gains `curation_engine:
  fiftyone-brain`, per-variant `uniqueness`, near-duplicate clusters, and a
  kept/dropped `selection` (schema stays `npa.fiftyone.curation.v1` — new fields
  are additive). Outside the image (unit tests, dev-VM worktree python) it
  degrades to the report-only counts path (`curation_engine: report-only`). Run it
  standalone with `npa workbench fiftyone curate-augmented --augment-uri ...
  --report-uri ...`. The agent's Voxel51 tab surfaces uniqueness + kept/dropped
  per card and curation stats in the summary (`build_fiftyone_dataset`).
  `test_publish_transfer_layout_interoperates_with_curate_and_viz` guards it. The
  augment stage "multiplies": it runs one inference per sampled combo and emits
  one clip dir per variant (`publish_transfer_clip` per clip + a single
  `write_run_manifest`), so N sampled combos → N clip dirs.
- **Full pipeline in the Rerun panel:** `build_run_rrd` logs the run's input
  frames and each augmented clip's frames/label PLUS a static text doc per stage
  — sampled scenarios (config), the attribute-verify / hallucination grade + gate
  decision, the curation report, the finalize aggregate, and a stage log — under
  the `pipeline/*` entities, so the whole pipeline (logs, hallucination check,
  input + output images) is viewable in one embedded Rerun recording.
- **Viewing in the NPA agent:** every stage lands under one S3 run prefix
  (`input/ configs/ labeled_original/ cosmos_augmented/ grade/ labeled_augmented/
  curation/ reports/`). The `visualize` stage writes `reports/sim2real.rrd`,
  which the agent's embedded Rerun viewer renders. Browse via
  `/api/artifacts/runs?prefix=physical-ai-data-factory`. See
  `docs/workbench/guides/physical-ai-data-factory.md`.

## Testing (live-infra is a priority)

Follow `skills/atomic/testing-conventions/SKILL.md` ("Live-Infra Testing Is A
Priority"). The blueprint is registered in `SUBMIT_LIVE_MATRIX` and
`DYNAMIC_SPECS`; smoke via `test_all_workflow_yamls.py`; the Rerun builder via
`npa/tests/workflows/test_data_factory_viz.py`.
