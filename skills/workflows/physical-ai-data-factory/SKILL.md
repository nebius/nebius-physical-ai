---
name: physical-ai-data-factory
description: Use when authoring, running, submitting, or viewing the NVIDIA Physical AI Data Factory blueprint on Nebius + SkyPilot (no OSMO) — annotate → Cosmos Transfer augment → Cosmos Evaluator gate → re-label → Cosmos Curator + FiftyOne curate → Rerun visualize — implemented as an npa.workflow that composes existing workbench tools.
---

# Physical AI Data Factory (NPA-native, no OSMO)

## Source And Attribution

NPA-native re-implementation of the NVIDIA Physical AI Data Factory / Video Data
Augmentation workflow. Design adapted from NVIDIA agent skills
(https://github.com/NVIDIA/skills), primarily `physical-ai-video-data-augmentation`.
Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. Upstream licenses:
Apache-2.0 and CC-BY-4.0. See `skills/NOTICE-NVIDIA-SKILLS`. NPA orchestrates on
SkyPilot (not OSMO) and composes existing workbench tools.

Three NVIDIA components in the pipeline are the real open-source projects, not
NPA look-alikes: **Cosmos Transfer 2.5** augments, **Cosmos Evaluator**
(https://github.com/nvidia-cosmos/cosmos-evaluator, Apache-2.0) grades, and
**Cosmos Curator** (https://github.com/nvidia-cosmos/cosmos-curate, Apache-2.0)
curates. See `skills/NOTICE-NVIDIA-COSMOS-OSS` for exactly which upstream code
runs and where NPA substitutes its own endpoint.

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
| Evaluate & Validate | `grade` loop (`evaluate` + `quality-gate`) | `workbench.cosmos_evaluator.evaluate` (real Cosmos Evaluator: hallucination + attribute verification) + `data_factory_stages.grade_gate` | Token Factory + CPU |
| Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Curation | `cosmos-curate` | `workbench.cosmos_curate.curate` (real Cosmos Curator stages → `clips/` + `metas/v0/`) | CPU |
| Curation review | `curate` | `workbench.fiftyone.curate_augmented` → `npa workbench fiftyone curate-augmented` (real FiftyOne Brain, merges the curator report) | CPU |
| Visualize | `visualize` | `workbench.nurec.visualize` → `data_factory_viz.build_run_rrd` → `reports/sim2real.rrd` | CPU, prebuilt `npa-rerun-viewer` image |
| Finalize | `finalize` | `data_factory_stages.finalize` (real aggregate report) | CPU |

Every stage invokes a real component (enforced by `test_real_components.py` and
the `real-components` skill). The `augment` stage runs the real Cosmos Transfer
2.5 model on GPU via `--execute` and publishes the generated video + extracted
frames to `augment_uri`, which the grade / re-label / curate / visualize stages
consume.

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

Chat authoring fails closed above 64 augmented variants or 8 GPUs. These are
generation ceilings, not workflow/job-count budgets: an operator can still edit
and validate a larger hand-authored spec after confirming real cluster capacity.

**Input conditioning (real augmentation of the caller's clip).** The managed
`workbench.cosmos2.transfer_execute` path always conditions on the caller's real
footage. Its `config.trigger_uri` must contain at least one supported video (`.mp4`,
`.mov`, `.webm`, `.mkv`, or `.avi`); an empty, inaccessible, or video-free input
fails closed before inference. Bundled upstream media was removed for
redistribution reasons and is not a fallback. The augment downloads the first clip
under `--input-uri` (the run's `input/`), builds a controlnet spec with `video_path`
= that clip and an **`edge`** (or `vis`) control computed on-the-fly, and the sampled
appearance prompt drives the new look — so the output preserves the input's
structure/motion with a new appearance. Generic direct CLI callers can opt into the
same behavior with `--condition-on-input` or `--input-video <path|s3://>`. `edge`/
`vis` need no precomputed control asset; `depth`/`seg` would need one, so input-only
conditioning falls back to `edge`. Conditioned runs record `mode:
cosmos_transfer2.5_gpu` + `input_conditioned: true` + `conditioned_input` in the
augment `metadata.json` / `manifest.json`, which the agent's provenance panel surfaces.

**Cosmos Evaluator grading (`evaluate` stage).** `npa workbench cosmos-evaluator
evaluate` runs two of upstream's checks per augmented variant and writes
`grade/cosmos_evaluator.json` (schema `npa.cosmos_evaluator.report.v1`):

- *attribute verification* — upstream's protocol: an LLM writes one
  multiple-choice question per sampled appearance attribute (guided JSON schema,
  with upstream's tolerant text fallback), then a VLM answers it from a frame of
  the variant. Both hops run on Token Factory, because upstream drives them
  through a configurable OpenAI-compatible endpoint. The sampled combo is
  upstream's `selected_variables` and `APPEARANCE_VARIABLES` is its
  `variable_options`, so a variant that ignored its prompt fails.
- *hallucination* — per-frame dynamic-mask comparison of the source clip against
  the variant. CPU only. It delegates to upstream's own `HallucinationProcessor`
  when a checkout is importable (`NPA_COSMOS_EVALUATOR_SRC`, else
  `/opt/cosmos-evaluator`) and otherwise runs the in-repo port of the same
  algorithm; the result's `engine` field says which ran, and the two agree to
  ~1e-3. Managed variants are input-conditioned, so this comparison contributes
  to their score. For a generic unconditioned transfer the source and output are
  different scenes, so it remains informational and the score is the attribute
  pass rate.

`grade_gate` thresholds on that report's `score`. It also still accepts the older
`vlm_eval` report (`vlm_eval_stub.json`, a LEGACY filename of the vlm_eval tool's
`RESULT_FILENAME`, never a stubbed stage), so runs started before the `evaluate`
stage existed keep grading. Both filenames come from the producing tool's own
constant, so the gate cannot drift from its producer.

**Cosmos Curator curation (`cosmos-curate` stage).** `npa workbench cosmos-curate
curate-augmented` drives upstream's real stage classes in-process — no Ray
scheduler, no GPU: `VideoDownloader` → `FixedStrideExtractorStage` →
`ClipTranscodingStage` → `MotionVectorDecodeStage` + `MotionFilterStage` →
`ClipWriterStage`. It writes upstream's canonical tree under `curated_clips_uri`
(`clips/<clip-uuid>.mp4`, `metas/v0/<clip-uuid>.json` with real per-clip motion
scores, `processed_videos/`) plus a summary at `curator_report_uri`, which the
`curate` review stage merges into its report under `cosmos_curator`.

Runs in the `npa-cosmos-curate` image, which bakes a pinned upstream checkout and
a conda-forge ffmpeg carrying `libopenh264` — upstream's transcoding stage accepts
only `libopenh264` or `h264_nvenc`, and Debian/Ubuntu ffmpeg builds have neither.
Outside that image the stage records `engine: unavailable` with the exact reason
(`npa workbench cosmos-curate engine` prints the same diagnosis) rather than
pretending to curate, and the FiftyOne review stage still runs. Operators with the
full curator container can instead run upstream's GPU pipeline — `npa workbench
cosmos-curate plan-pipeline` prints the documented `video-pipeline split` command,
which adds TransNetV2 shot detection, aesthetic filtering, embeddings, and VLM
captioning; both paths write the same layout, so the ingest reads either.

## Containers And Model Weights

Both NVIDIA tools are containerized as workbench images, and **neither image
carries model weights**. That is a licensing boundary, not an optimization:
upstream's *code* is Apache-2.0 and redistributable, its *weights* are not ours to
ship. A build-time check in each Dockerfile fails if a weight file is present, a
guardrail test (`npa/tests/docker/test_cosmos_oss_images.py`) fails if a Dockerfile
grows a build-time download, and both checkouts are fetched with
`GIT_LFS_SKIP_SMUDGE=1` so upstream's Git-LFS payloads never enter a layer.

| Image | Tier | Weights | Credential at run time |
| --- | --- | --- | --- |
| `npa-cosmos-evaluator` | job, CPU | none needed | `NEBIUS_TOKEN_FACTORY_KEY`, for attribute verification only |
| `npa-cosmos-curate` | job, CPU | none baked; GPU stages fetch on demand | `HF_TOKEN`, for `fetch-models` |

**The evaluator needs no weights at all.** The hallucination check is classical
computer vision, and attribute verification calls a hosted VLM instead of loading
one — so its golden eval runs with `--network none`. Upstream's *objects/obstacle*
check would need an EULA-gated SegFormer ONNX and a CWIP checkpoint; those stay as
LFS pointers and that check is not wired. Anyone who needs it must accept
upstream's EULA and fetch the weights themselves.

**The curator's GPU stages do need weights**, so the image fetches them at run time
with the operator's own Hugging Face token, into a `/config/models` volume that
survives across runs:

```bash
docker run --rm -e HF_TOKEN=... -v curator-weights:/config/models \
  <registry>/npa-cosmos-curate:0.1.0 fetch-models --models split-annotate
docker run --rm -v curator-weights:/config/models \
  <registry>/npa-cosmos-curate:0.1.0 models --output text
```

Model sets name a capability, and each set's membership mirrors the
`model_id_names` of the upstream model class that stage instantiates:
`split-transnetv2`, `embed-internvideo2`, `embed-cosmos-embed1`,
`filter-aesthetic`, `caption-qwen`, `dataset-t5`, and `split-annotate` (what
upstream's `video-pipeline split` needs with default flags). The model ids **and
their pinned revisions come from upstream's own registry**
(`cosmos_curator/configs/all_models.json`), and the download is upstream's own
`huggingface_hub` call, so a pin moves only when the checkout does — never because
NPA hardcoded one. `fetch-models` skips anything already complete, records
per-model failures instead of aborting the batch (a gated repo shows up as its own
403), and refuses outright with an actionable message when no token is set.

NGC versus HF: every curator model is a Hugging Face repo, so `HF_TOKEN` is the
credential that fetches weights. `NGC_API_KEY` is what pulls NVIDIA *containers*
(and what upstream's own NVCF path uses); `models` reports whether each is visible
so an operator can tell which one is missing.

Both images are mode-based (`engine`, `smoke`, plus each tool's commands), so one
image serves the workflow stage, the golden eval, and interactive debugging.

**Custom images must satisfy SkyPilot's Kubernetes setup.** Its provisioner runs, in
the image and as the image's own user, `$(prefix_cmd) apt install openssh-server
rsync -y` then `service ssh restart`, where `prefix_cmd` is `sudo` for a non-root
user. An image missing `sudo` or those packages fails that script, the container
exits, and SkyPilot reports `container not found ("ray-node")` — which reads like a
scheduling fault and is the reason operators historically reached for unpinned
submits. All three Cosmos images install
`openssh-server`, `rsync`, and `sudo` and grant their user passwordless sudo;
`npa/tests/docker/test_cosmos_oss_images.py` fails if that regresses. The same test
covers the entrypoint contract: a bare `ENTRYPOINT ["/bin/bash"]` swallows the args
Kubernetes passes, so an entrypoint must exec its arguments.

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

Run either NVIDIA component on its own, against a run prefix or local files:

```bash
# Which engine will each resolve to here, and why?
npa workbench cosmos-evaluator engine --output text
npa workbench cosmos-curate    engine --output text

# Grade one run's variants (writes grade/cosmos_evaluator.json).
npa workbench cosmos-evaluator evaluate \
  --augment-uri s3://<bucket>/physical-ai-data-factory/<run>/cosmos_augmented/ \
  --output-uri  s3://<bucket>/physical-ai-data-factory/<run>/grade/ \
  --input-uri   s3://<bucket>/physical-ai-data-factory/<run>/input/ \
  --configs-uri s3://<bucket>/physical-ai-data-factory/<run>/configs/

# One check at a time (both take local paths).
npa workbench cosmos-evaluator hallucination --original-video a.mp4 --augmented-video b.mp4
npa workbench cosmos-evaluator attribute-verify --video b.mp4 \
  --variables '{"cloth_color": "blue"}' --options '{"cloth_color": ["blue","red","white","green"]}'

# Curate one run's variants (writes the curator tree + cosmos_curator.json).
npa workbench cosmos-curate curate-augmented \
  --augment-uri s3://<bucket>/physical-ai-data-factory/<run>/cosmos_augmented/ \
  --curated-uri s3://<bucket>/physical-ai-data-factory/<run>/curation/cosmos_curator/

# Curate a local directory of clips.
npa workbench cosmos-curate curate-videos --input-dir ./clips --output-dir ./curated
```

## Key Operational Notes

- **Neither NVIDIA component fetches code at run time.** The evaluator looks for
  a checkout at `NPA_COSMOS_EVALUATOR_SRC` / `/opt/cosmos-evaluator` and falls
  back to the in-repo port; the curator requires `NPA_COSMOS_CURATE_SRC` /
  `/opt/cosmos-curate` (baked into `npa-cosmos-curate`) and reports
  `engine: unavailable` without one. If a report says `unavailable`, read its
  `reason` — it names the missing piece (checkout, importability, or ffmpeg
  encoder) instead of guessing.
- **Curator encoder gotcha.** `ffmpeg -encoders | grep -E 'libopenh264|h264_nvenc'`
  must match something, or upstream's `ClipTranscodingStage` cannot write clips.
  Debian/Ubuntu ffmpeg has neither; conda-forge's build carries `libopenh264`, and
  any GPU node's ffmpeg carries `h264_nvenc`.
- **Curator needs Python >= 3.12** (upstream declares
  `requires-python >=3.12,<3.13`). On an older interpreter the failure otherwise
  surfaces as `cannot import name 'Self' from 'typing'` from deep inside an
  upstream import, so the availability probe checks the version first and says so.
  The `npa-cosmos-curate` image is 3.12; a dev box on 3.10/3.11 needs its own
  3.12 environment (`uv venv --python 3.12`) to run the curator locally.
- **A VLM can answer outside the options it was given.** Token Factory's VLM does
  return `D` for a three-option question. That reads as `UNKNOWN` (a failed check
  with no verdict about the pixels), not as a concrete wrong answer, so check
  `vlm_answer` before drawing conclusions from a failed attribute.
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
  --report-uri ... --curator-report-uri ...`. The agent's Voxel51 tab surfaces uniqueness + kept/dropped
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
- **Restricted-egress visualization:** the `visualize` state uses
  `workbench.nurec.visualize`, which the renderer pins to the prebuilt
  `npa-rerun-viewer` image. It never performs `pip install` at task runtime.

## Testing (live-infra is a priority)

Follow `skills/atomic/testing-conventions/SKILL.md` ("Live-Infra Testing Is A
Priority"). The blueprint is registered in `SUBMIT_LIVE_MATRIX` and
`DYNAMIC_SPECS`; smoke via `test_all_workflow_yamls.py`; the Rerun builder via
`npa/tests/workflows/test_data_factory_viz.py`.
