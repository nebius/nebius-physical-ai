# NVIDIA Physical AI Data Factory on NPA (no OSMO)

This guide runs **NVIDIA Physical AI Data Factory** workflows natively on
Nebius + SkyPilot. Five `npa.workflow` specs cover the direct VDA, scoped DIG,
IAA, and EVG translations plus one clearly labeled NPA-specific Cosmos3 VDA
alternative. They compose registered Workbench tools and narrow protocol
adapters; neither OSMO nor Airflow is embedded. SkyPilot is the sole
orchestrator, and stages hand off durable artifacts through S3-compatible
storage.

Two official repositories have intentionally different roles. NVIDIA's
[Physical AI Data Factory](https://github.com/NVIDIA/physical-ai-data-factory)
repository is the ecosystem and agent-skill entry point; its Video Data
Augmentation workflow runs on OSMO upstream. NVIDIA's
[PAIDF Orchestration](https://github.com/NVIDIA/paidf-orchestration) repository
is an Apache Airflow-on-Kubernetes scaler and currently carries Image Attribute
Augmentation and Event Video Generation DAGs. NPA does not embed either
orchestrator and does not describe those Airflow DAGs as this VDA workflow.
Every NPA run records the reviewed revisions, licenses, and this execution
boundary in `reports/upstream.json`; see `skills/NOTICE-NVIDIA-PAIDF`.

## Authoritative YAML mapping

| NPA YAML | Upstream repository / workflow | Relationship | NPA orchestrator / runtime |
| --- | --- | --- | --- |
| `physical-ai-data-factory.yaml` | `NVIDIA/physical-ai-data-factory` / Video Data Augmentation | Direct VDA translation using Cosmos Transfer 2.5 | `npa.workflow/v0.0.1` on SkyPilot; Workbench GPU/CPU stages + Token Factory |
| `paidf-defect-image-generation.yaml` | `NVIDIA/physical-ai-data-factory` / DIG Day-1 manual-ROI default fresh-finetune branch | Direct, deliberately scoped translation; not the USD Day-0 or real-alignment branch | `npa.workflow/v0.0.1` on SkyPilot; operator-built restricted AnomalyGen compatibility image on B200 |
| `paidf-image-attribute-augmentation.yaml` | `NVIDIA/paidf-orchestration` / `image_attribute_augmentation_dag` | Direct Airflow-DAG translation | `npa.workflow/v0.0.1` on SkyPilot; operator-built Qwen Image Edit worker + pinned PAIDF augmentation/auto-label protocols |
| `paidf-event-video-generation.yaml` | `NVIDIA/paidf-orchestration` / `event_video_generation_dag` | Direct Airflow-DAG translation | `npa.workflow/v0.0.1` on SkyPilot; operator-built Cosmos3 Super worker + pinned PAIDF augmentation and auto-label services |
| `paidf-cosmos3.yaml` | NPA composition informed by the PAIDF VDA contract; no upstream Airflow DAG | NPA-specific Cosmos3 video2video VDA alternative, not IAA or EVG | `npa.workflow/v0.0.1` on SkyPilot; NPA Cosmos3/Curator/FiftyOne/Rerun images |

All five specs write `npa.paidf.upstream.v1`. Direct translations name the
exact upstream workflow and revision; the Cosmos3 alternative records
`translation: npa-specific-variant`. Vendor images are digest-pinned. Gated
weights and operator inputs remain runtime-only and are never published in NPA
image layers. IAA pins `Qwen/Qwen-Image-Edit-2511` at
`6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`; EVG pins
`nvidia/Cosmos3-Super-Image2Video` at
`4f847566f3d3388fbf0ac07b99dd1a6432db9ecd`.
The direct IAA and EVG translations reject model/revision overrides outside
these reviewed pairs. Their VLM/LLM endpoints must use the approved Token
Factory HTTPS origin before the operator's credential is forwarded. Every
native artifact handoff verifies its schema and run identity; branch-specific
handoffs also verify the workflow identity. An enabled attribute evaluator must
return an explicit passing verdict for an output to enter the final dataset.

### Native DIG, IAA, and EVG execution contracts

The three new translations are ordinary generic workflow submissions; they do
not add one-off top-level CLI commands. Before any GPU or image work, validate
the exact spec and run its workflow-specific access checks:

```bash
npa workbench workflow validate-spec workflows/testing/paidf-defect-image-generation.yaml --json
npa workbench health access --capability paidf-dig
npa workbench health access --capability paidf-iaa
npa workbench health access --capability paidf-evg
npa workbench health access --capability paidf-label-detection
npa workbench health access --capability paidf-label-captioning
npa workbench health access --capability paidf-label-visual-qa
npa workbench health access --capability paidf-label-attribute-search
```

`validate-spec` derives only human-gated workflow blockers. DIG therefore names
the exact Cosmos Guardrail payload; IAA names only the attribute-search NGC
image; EVG names its four exact NGC service digests and gated
Cosmos-1.0-Guardrail snapshot. The broader
`health access` calls also probe public, revision-pinned model payloads used at
runtime. None of these checks accepts terms for the operator.

DIG is the official Day-1 manual-ROI fresh-finetune path: it validates the clean
images, ROI masks, and defect specification, runtime-fetches and verifies the
pinned AnomalyGen checkpoint closure, fine-tunes the selected upstream recipe,
then runs the upstream generator and native label export. It intentionally does
not claim Day-0 USD scene preparation or the separate PCBA real-alignment path.

DIG's pinned OpenMDW-1.1 Cosmos Framework Qwen guard catches inference and
parsing errors and returns an allowed result upstream. NPA uses an exact-source
private overlay to raise on those errors and require one complete published
safety verdict. Safe and Controversial keep their upstream allowed decisions;
Unsafe is rejected. The vendor interpreter must import the reviewed module,
and the completed run must report enforced text screening before publication.
The result binds the source/license, original and adapted hashes, package trees,
and actual timing summary. Installed vendor files and NLTK behavior are
preserved. The upstream image preset runs RetinaFace blur but contains no image
content classifier; its enforcement flag remains false.

The upstream IAA/EVG generation images lack required SkyPilot worker commands.
The accepted EVG image also needs an exact-source tokenizer adaptation for its
Transformers 5.13 runtime: the published Cosmos3 tokenizer specifies Qwen2 but
does not ship the model config that automatic dispatch requests. NPA selects the
published `qwen2` tokenizer type in a private copy of the verified vLLM-Omni
pipeline. The seven pinned tokenizer files, prompt method, special tokens and
installed vendor package remain unchanged. `generation_runtime` requires the
exact `tokenizer_source_adaptation` alongside its guardrail adaptation. EVG
serving and DIG offline children receive no Hugging Face token after staging.

Build `paidf-image-edit-sky` and `paidf-event-video-sky` from their exact pinned
parents, scan the actual bytes, publish privately, and supply their immutable
digests with `--var generation_image=<operator-image@sha256:digest>`. DIG uses
`--var anomalygen_image=<operator-image@sha256:digest>`. The checked-in defaults
are deliberately non-runnable placeholders. The [container catalog](../container-image-catalog.md)
records verified wrapper publication separately from native workload acceptance.

IAA selects the official aligned vLLM/Omni 0.22.0 parent to fix the
[authentication bypass in the blueprint's 0.20 runtime](https://github.com/vllm-project/vllm/security/advisories/GHSA-94f4-hr76-p5j6).
Its Qwen model revision and image-edit protocol are preserved. Upstream
provenance records both the original blueprint image and this security update;
the executed report records the actual worker digest. EVG upgrades NLTK to the
hash-pinned 3.10.3 wheel to fix its inherited
[JVM argument injection vulnerability](https://github.com/nltk/nltk/security/advisories/GHSA-m4rf-3fr8-xwx3).
Its actual service also needs `nvidia/Cosmos-1.0-Guardrail` at
`cf03c0395fac8c4de386c0bdab12cc4fc8d66362` and
`Qwen/Qwen3Guard-Gen-0.6B` at `fada3b2f655b89601929198343c94cd2f64d93cc`.
NPA checks access, fetches the exact snapshots, verifies each file's path, size,
and hash against the pinned revision's Hub manifest, and
binds vendor default references inside an isolated cache consumed offline.
Only the small NLTK data subtree is copied to verified regular files; NLTK's
symlink protection and the upstream content guardrails stay enabled. The EVG
`generation_runtime` report binds model inventories and the NLTK tree hash
through output validation and terminal lineage.
The pinned upstream EVG template sets per-request `guardrails: false`. NPA
explicitly sets it to `true` and rejects missing, false, or malformed values
before execution; this deliberate adaptation keeps the declared guardrail
contract consistent with actual generation requests.
The installed Qwen guardrail also catches inference/parser errors and returns
allow. An exact-source-bound runtime code copy changes those errors to failures
and requires one complete `Safety: Safe`, `Safety: Unsafe`, or
`Safety: Controversial` verdict line. Real model inference is preserved;
`Controversial` retains the upstream allow policy. The installed package remains
unchanged, and EVG provenance records the original and adapted source hashes.
Both generation recipes update the distribution kernel headers before scanning.
Every native handoff retains hashed producer reports, exact scene sets, and
content manifests; terminal validation reopens the full generation and labeling
chain, including each executed worker image.

The labeling stages likewise require the exact private digests built from
`paidf-attribute-search-sky` (IAA and EVG), `paidf-detection-sky`,
`paidf-captioning-sky`, and `paidf-visual-qa-sky` (EVG). Set
`attribute_search_image`, `detection_image`, `captioning_image`, and
`visual_qa_image` as applicable. Their placeholders deliberately fail before
launch. These restricted recipes retain the pinned NGC service environment and
real `/app/.venv/bin/main` CLI, adding worker bootstrap and command forwarding.
NPA uses an independent `/opt/npa-venv` so its setup leaves vendor dependencies
unchanged.
Configure pull secrets for the operator registry. Build-byte scanning, registry
pullability, bootstrap proof and real labeling artifacts remain required.

EVG's detector runtime-fetches the public RF-DETR Base checkpoint with the
SHA-256 published in the pinned auto-labeling source. NPA requires that same
digest for custom cache filenames too; a different model hash fails before
labeling. The source URL and expected hash appear in both upstream provenance
and the completed detection report. The selected BoostTrack path uses no
additional ReID model.

The live matrix takes these digests from
`NPA_E2E_PAIDF_ATTRIBUTE_SEARCH_IMAGE`, `NPA_E2E_PAIDF_DETECTION_IMAGE`,
`NPA_E2E_PAIDF_CAPTIONING_IMAGE`, and `NPA_E2E_PAIDF_VISUAL_QA_IMAGE`; it does not
substitute the incompatible raw NGC parents when an override is missing.
The EVG live test fetches Lav Varshney's CC0 camera photograph from a pinned
scikit-image revision, checks its SHA-256 before staging, and records source
provenance. It exercises real person detection and labeling; IAA and DIG use
repository-authored inputs. See the [starter-media notice](../../../skills/NOTICE-PAIDF-STARTER-MEDIA)
for exact source and license boundaries.

IAA and EVG preserve the upstream service/batch boundary by starting the pinned
vLLM-Omni generation service inside the same SkyPilot state that consumes it.
Frozen client setup and execution select the worker's Python interpreter.
IAA postprocessing reuses its generation image on a CPU-only resource because
the upstream client requires Python 3.11 or newer. Video validation decodes
actual frames with NPA's pinned PyAV dependency; it requires no host `ffprobe`.
Input preparation writes real RGB JPEG files and retains both source and
prepared hashes. The pinned augmentation client otherwise writes PNG response
bytes under its published `.jpg` output path and declares them as JPEG to its
verifier. NPA applies an exact-source-bound writer correction before execution:
JPEG outputs are encoded as JPEG, while video bytes remain unchanged. The
augmentation report records the original and patched source hashes, and output
and terminal validation require that exact adaptation. The configured MiniCPM
model remains unchanged. Runs with previously completed PNG preparation need a
fresh run identity so completed input artifacts are preserved.
Each real PAIDF component gets the upstream three retries with a 30-second retry
delay. EVG then preserves the serial detection → captioning → anomaly Visual QA
→ per-person Visual QA → person-attribute-search chain. Component adapters
require the published contextual and sidecar files immediately after each
service exits. Dataset assembly performs the upstream track-aware completeness
rules, and a separate terminal state re-opens every media, caption, metadata,
and labeling handoff before writing `reports/terminal-validation.json`.
Captioning and anomaly Visual QA need one B200 because the pinned labeling
stack decodes H.264 through NVIDIA CUVID; hosted VLM calls do not remove that
local decoder requirement. Per-person Visual QA reads detector JPEG crops and
shares the same B200 profile. Person attribute search consumes the resulting
labels and remains a CPU stage. NPA preserves the
[upstream codec policy](https://github.com/NVIDIA/paidf-auto-labeling/blob/36dc1114dea00d9986df97325a664520993964de/scripts/ffmpeg_codec_policy.py).

The selected Token Factory VLM accepts at most ten images in one request.
Anomaly Visual QA uses the upstream CLI's `--max-frames 10` (DAG default: 16),
and per-person Visual QA uses `--max-crops-per-track 10` (DAG default: 12).
The pinned service [evenly subsamples the candidates](https://github.com/NVIDIA/paidf-auto-labeling/blob/36dc1114dea00d9986df97325a664520993964de/packages/tasks/visual_qa/src/visual_qa/media.py),
including the first and last candidate in each sequence; a candidate endpoint
is not necessarily the video's final frame. Question banks, models, resolution,
sampling rate and retries are preserved. Each VQA result records the exact
original/effective sampling controls and source hash in `request_media_contract`.
Downstream stages and terminal validation reject missing or changed contracts.
This uses the published CLI without patching vendor code or rebuilding an image.

The optional Airflow-only YAML/HTML performance dashboard is not copied: it
queries Airflow's task-instance REST metadata, which does not exist in the NPA
runtime. SkyPilot/NPA retain stage timing and terminal state, while the terminal
JSON records output counts, manifest digest, validated-artifact count, and
trackless-scene count. This is an explicit orchestration-reporting substitution,
not a model or data-path substitution.

### Native live validation evidence

IAA completed all nine states on reserved B200 capacity from source
`39120bc9b567d6400d4fe955988132ba1f6ce682`. The upstream attribute evaluator
accepted one 896×1184 JPEG (81,149 bytes); CPU postprocessing and the real
attribute-search service produced one person with structured clothing attributes
and three queries in each of the easy, medium, and hard tiers. The final dataset
has one entry and 13 assembled files. Independent live assertions and a fresh
terminal-validation replay verified all four producer handoffs, the executed
image digests, copied artifacts, and both required terminal artifact groups.

The generated JPEG SHA-256 is
`d47fb223aa0e4d2354beb94af68b2ae7417a04549c92e336ae3eb3a63c9bb9d0`;
the dataset manifest SHA-256 is
`fc245007886b6372a10afd082e156ed2bedf1af7eb7b3ad4ecaef0e63e7b858a`.
Submission to terminal completion took 1,664.3 seconds; the GPU state took
269.0 seconds. Ten GPU samples observed up to 61,496 MiB resident memory but
missed active kernels, so they do not establish compute utilization.
Visual review confirmed the requested black hoodie, black shorts, and brown
boots. The result is a three-view illustration from a repository-authored
silhouette; this acceptance does not establish photographic fidelity or
preservation of a single-person composition.

EVG completed all twelve states on reserved B200 capacity. The first seven
states ran from `27700b94612d7f8297a4f879c1c3f550bff467f1`; an explicit durable
resume retained their verified artifacts and completed the remaining five from
`f466e119f1249de81e2e752ff3092098c5839964`. The retained first attempt records the
hosted VLM's rejection of twelve images. The resume used the published ten-image
controls described above. Independent end-to-end assertions and terminal replay
verified one assembled scene, 47 files, twelve validated artifacts, no trackless
scenes, and the complete generation, detection, captioning, VQA and search
handoffs. All 47 assembled files (22,191,075 bytes) and seven lineage documents
were reopened; sixteen JPEG detector crops were decoded. PAS produced one
person with six nonempty, distinct primary queries. The canonical dataset JSON
SHA-256 is
`ddc54f5424d5819312bd13678716621ca4c8983b818e0a94ee192367519b9d37`.

The final H.264 video contains 93 fully decoded frames at 1280×720 and 24 fps
(3.875 seconds, 5,020,456 bytes), with SHA-256
`d98203dba3798514b1b20dcef0a830aa26e4b1b75b8803082ec2f0895b799a0b`.
Its bytes match the generated video reviewed visually. Anomaly VQA returned
21 option-valid answers for 21 questions. Person VQA returned 29 valid answers
from a 33-question bank: the upstream normalizer skipped two empty answers with
warnings, and the model omitted two headwear-detail answers. The published
protocol accepts this partial response; PAS retains its optional-field behavior.
This result establishes protocol acceptance with recorded missing answers, not
complete person-attribute coverage.

Four reviewed frames show the source camera and tripod, with a person moving
beside them and ending low to the ground. Face pixelation is visible in three
reviewed frames; the final face is angled down and partly obscured. These stills
do not establish physical accuracy, continuous-motion fidelity or perfect blur
recall. The resume took 1,081.748 seconds and reused generation. Its nine GPU
samples observed zero compute utilization and resident memory; the earlier
generation attempt separately recorded a 100% utilization peak and 86,774 MiB
peak resident memory. Exact run locations remain in owner-only evidence.

DIG image and full native acceptance remain pending; image and CPU protocol
checks do not replace its fine-tuning and generation workload.

> **Want the from-zero runbook?** See
> [physical-ai-data-factory-deploy.md](physical-ai-data-factory-deploy.md) for a
> copy-paste deploy guide: install `npa`, set credentials/config, deploy the NPA
> agent, submit this blueprint (including multi-GPU fan-out and real FiftyOne
> curation), and view results in the agent.

## Blueprint mapping

NVIDIA blueprint (OSMO) → NPA stage (toolRef / run):

| NVIDIA stage | NPA state | Tool | Runtime |
| --- | --- | --- | --- |
| Source boundary | `record-upstream` | `paidf_upstream.write_upstream_contract` | CPU |
| Stage 1 Config Generation | `generate-configs` | `run.shell` (sample appearance-only variables) | CPU |
| Stage 2a Understand & Annotate | `annotate-original` | `workbench.token_factory.caption` | Token Factory (zero-GPU) |
| Stage 2b Augment & Multiply | `augment` | `workbench.cosmos2.transfer_execute` | GPU (Cosmos Transfer 2.5; optional upstream Meta SAM2 masks) |
| Evaluate & Validate | `grade` loop + `quality-disposition` | Cosmos Evaluator + NPA source-relative temporal/appearance fidelity + fail-closed disposition | Token Factory + CPU |
| Stage 3 Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Stage 4a Curation | `cosmos-curate` | `workbench.cosmos_curate.curate` | CPU |
| Stage 4b Curation review | `curate` | `workbench.fiftyone.curate_augmented` (required FiftyOne Brain) | CPU |
| Visualize | `visualize` / `visualize-rejected` | `data_factory_viz.build_run_rrd` (full accepted run or partial rejection evidence) | CPU |
| Finalize | `finalize` | `data_factory_stages.finalize` | CPU |

Three NVIDIA components in that table are the real open-source projects:

- **Cosmos Transfer 2.5** augments (GPU diffusion, `npa-cosmos2-transfer` image).
  It is **not** a Token Factory model.
- **[Cosmos Evaluator](https://github.com/nvidia-cosmos/cosmos-evaluator)**
  (Apache-2.0) grades. The `evaluate` stage runs two of upstream's checks per
  variant: *attribute verification* (an LLM writes one multiple-choice question
  per sampled attribute, a VLM answers it from a truthful beginning/middle/end
  contact sheet — both on Token Factory,
  since upstream drives them through a configurable OpenAI-compatible endpoint)
  and *hallucination* (per-frame dynamic-mask comparison against the source clip,
  CPU only). The hallucination score only feeds the run score for
  input-conditioned variants; otherwise the clips are different scenes and it stays
  informational. NPA adds a separately attributed source-relative temporal
  consistency diagnostic over the full frame and localized regions. It Gaussian-
  filters decoded frames, compares signed temporal acceleration, and scores the
  two-sided residual against a fixed noise floor. This exposes both added
  instability and collapsed source motion without weakening as source motion rises.
  It is `advisory` by default because encoded capture paths need calibration; set
  `temporal_consistency_mode: required` only after tuning `temporal_noise_floor`
  on representative clips. NPA also reports protected-appearance fidelity in
  CIELAB: p95 luminance/chroma drift, localized chroma residual after subtracting
  the scene-wide shift, and frame-to-frame chroma-shift instability. This catches
  excessive global colour casts and localized material recolouring that stable
  motion checks cannot see. It is also `advisory` by default because generic data
  factories may intentionally recolour materials. Set
  `appearance_fidelity_mode: required` when base colours/material identity are
  invariants, and optionally supply normalized `appearance_regions_json` for the
  protected areas. Empty regions use the generic full frame plus a 2x2 grid.
- **[Cosmos Curator](https://github.com/nvidia-cosmos/cosmos-curate)**
  (Apache-2.0) curates. The `cosmos-curate` stage drives upstream's own stages —
  `VideoDownloader` → `FixedStrideExtractorStage` → `ClipTranscodingStage` →
  `MotionVectorDecodeStage` + `MotionFilterStage` → `ClipWriterStage` — and writes
  upstream's canonical `clips/` + `metas/v0/` tree with real per-clip motion
  scores. FiftyOne then reviews that set.

See `skills/NOTICE-NVIDIA-COSMOS-OSS` for exactly which upstream code runs and
where NPA substitutes its own endpoint.

**Model roles** (verify against the current key-scoped Token Factory catalog):

- VLM captioning + the evaluator's attribute answering: `MiniMaxAI/MiniMax-M3`
- Hosted reasoning critic: `MiniMaxAI/MiniMax-M3`
- Prompt / MCQ LLM: `nvidia/Nemotron-3_5-Lightning`

Model availability can change independently of the workflow. Run
`npa workbench token-factory models` immediately before execution and override
`caption_model` if the configured multimodal model is not reachable.

> **Config → augment MULTIPLY.** The `augment` stage receives the Config-Gen
> manifest via `--configs-uri` and runs **one Cosmos Transfer 2.5 inference per
> sampled combo**: each combo's prompt drives a distinct appearance, published as
> its own per-clip dir under the current scheduler-owned
> `cosmos_augmented/_attempts/<attempt-id>/<clip>/` prefix with its own `metadata.json`
> `variables` (which drives that clip's Rerun label). So a config with N
> augmentations produces **N scenario variants**, recorded via `variant_count` /
> `multiply_mode` in the augment manifest, curation, and finalize reports.
> Consumers follow only the executed canonical manifest, so artifacts
> retained from a delayed or recovered prior attempt are never counted.
> The managed transfer conditions every variant on a supported video under the
> run's `config.trigger_uri` (`input/`), preserving geometry/motion while changing
> appearance. `config.augment_control` chooses which structure is preserved:
> `edge` (default), `vis`, or `seg` may be derived from the clip; `depth` requires
> an operator-owned precomputed weight-free control. Video Depth Anything weights
> are not downloaded or executed by this workflow.

### Input conditioning and its evaluation source

Before image checks or automatic provisioning, `workflow submit` validates and
stages a source, normalizes it to an H.264/yuv420p 1280×720 clip of exactly 93
frames at 16 fps, and extracts eight caption frames. The catalog always invokes
`workbench.cosmos2.transfer_execute` with `--condition-on-input` (equivalent to
`NPA_COSMOS_CONDITION_ON_INPUT=1`), and the resolver explicitly prefers
`input/conditioning.mp4` over `source.mp4`. Cosmos Evaluator uses that same clip
for hallucination scoring. Missing or invalid input therefore cannot turn into a
decorative staged object or a fixed control example.

Edge, visibility-blur, and segmentation controls are derived from the staged clip.
Depth is precomputed-only and must be supplied as `augment_control_asset_uri`:
`--var augment_control=seg` conditions on a GroundingDINO+SAM2 segmentation
(`config.augment_control_prompt` names the classes) instead of Canny edges, which
lets a prompt change what a region is made of while keeping its shape and motion.
`config.augment_mask_prompt` (or a precomputed
`config.augment_mask_asset_uri`) restricts the control to one region so the rest of
the frame follows the prompt freely. An unsupported modality fails the stage rather
than silently rendering an edge-conditioned variant. The control map and mask that
conditioned each variant are published under `config.augment_control_uri`
(`cosmos_control/`, a sibling of `cosmos_augmented/`) and logged into the Rerun
recording, so the conditioning signal is reviewable. See
[section 6c of the deploy guide](physical-ai-data-factory-deploy.md#6c-choose-what-the-augmentation-preserves---var-augment_controlseg).

### Starter input: authenticity, licensing, and replacement

With no selector, PAIDF uses this pinned replaceable starter:

| Field | Verified value |
| --- | --- |
| Dataset | [Hoshipu/RoboPro](https://huggingface.co/datasets/Hoshipu/RoboPro), “RoboPro: 80-Task Bimanual Manipulation Demonstrations on Aloha-Agilex,” Zhiyuan Li (2026) |
| Immutable revision | `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0` |
| Asset | `lerobot/roboreal_all_80tasks/videos/chunk-000/observation.images.cam_high/episode_000000.mp4` ([pinned object](https://huggingface.co/datasets/Hoshipu/RoboPro/resolve/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/lerobot/roboreal_all_80tasks/videos/chunk-000/observation.images.cam_high/episode_000000.mp4)) |
| Authenticity | Actual physical capture: episode 000000 of expert teleoperation recorded from the high RGB camera on an Aloha-Agilex robot; [pinned episode metadata, line 1](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/lerobot/roboreal_all_80tasks/meta/episodes.jsonl#L1) says “reposition the bottle” |
| Integrity | SHA-256 `caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198`; 607,681 bytes |
| Media | MP4, H.264 High, 640×480, 50 fps, 169 frames, 3.38 s |
| Media/dataset license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode): sharing, adaptation, commercial redistribution, and hosted/service use are permitted with credit, license link, and modification notice; no field-of-use restriction, token, click-through, or special acceptance |
| Delivery | Operator-side pinned runtime fetch; NPA does not commit or bake the binary |

The machine-readable source of truth is
`npa/src/npa/assets/paidf_starter_video.json`, with attribution in
`skills/NOTICE-PAIDF-STARTER-MEDIA`. This media grant is separate from the NPA
code license, Cosmos Transfer's Apache-2.0 **source-code** license, the NVIDIA
Open Model License for runtime-fetched **weights**, and the operator's runtime
use. We reviewed the official Cosmos Transfer repository at NPA's pinned revision
`67d56b7d550a3911024a32dc23ae0bae5258e633`; its code license does not separately
license every Git LFS media asset or establish that every sample is a real
capture. Those media files are neither bundled nor silently fetched. See
`npa/docker/workbench/cosmos2-transfer/REDISTRIBUTION.md`.

Selection precedence is strict: one explicit local file, S3 object, or S3
LeRobotDataset prefix wins; conflicting selectors fail; no selector chooses the
starter; `--seed-fixture` is the only synthetic geometry path. Operator input is
labeled without inventing an authenticity or license claim for it.

```bash
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"

# Default: verified upstream real sample
npa workbench workflow submit "$SPEC" --run-id "$RUN_ID" --runtime \
  --var bucket="$BUCKET" --assume-decision promote_checkpoint

# Replace with local media
npa workbench workflow submit "$SPEC" --run-id "$RUN_ID" --runtime \
  --var bucket="$BUCKET" --input-video ./capture.mp4 \
  --assume-decision promote_checkpoint

# Replace with one S3 object
npa workbench workflow submit "$SPEC" --run-id "$RUN_ID" --runtime \
  --var bucket="$BUCKET" --input-uri s3://source-bucket/path/capture.mp4 \
  --assume-decision promote_checkpoint

# Replace with one operator-reviewed episode/camera video from an S3
# LeRobotDataset prefix. The strict flag prevents compatibility defaults from
# silently substituting another visual stream.
npa workbench workflow submit "$SPEC" --run-id "$RUN_ID" --runtime \
  --var bucket="$BUCKET" --lerobot-uri s3://source-bucket/datasets/robot-run/ \
  --lerobot-camera observation.images.front --lerobot-episode 0 \
  --require-explicit-lerobot-selection \
  --assume-decision promote_checkpoint

# Developers/tests only — explicitly synthetic
npa workbench workflow submit "$SPEC" --run-id "$RUN_ID" --runtime \
  --var bucket="$BUCKET" --seed-fixture --assume-decision promote_checkpoint
```

Those are alternatives: run one with the prepared fresh ID. A previous run is
resumed only with an explicit `--resume-run "$RUN_ID"`; an unattended command
never reads or silently reuses the legacy global `~/.npa/paidf-first-run-id`.
For subject-sensitive LeRobot runs, inspect beginning, middle, and end frames
privately, then pass `--require-explicit-lerobot-selection` with both reviewed
selectors. The gate runs before object-store access and provisioning. Omitting it
retains the compatibility behavior (episode 0 and a lexically first camera when
not otherwise specified); schema validity alone is not semantic confirmation.

The default cache is `~/.cache/npa/physical-ai-data-factory/`; set
`NPA_PAIDF_CACHE_DIR` to move it. Every hit is rechecked. Set
`NPA_PAIDF_OFFLINE=1` to forbid a fetch: an absent or corrupt cache then fails
with the expected path. Downloads use three bounded per-request attempts, an
atomic cache write, and an inter-process lock. Checksum mismatch, unsupported
container/codec/shape, inaccessible S3, or invalid local media fails closed—no
fixture fallback.

The source is committed last via `input/provenance.json`; `source.mp4`,
`conditioning.mp4`, and `conditioning-frame-*.png` carry digests. A retry with
the same run ID verifies and reuses the source, repairs deterministic derived
objects if needed, and refuses a different explicit source. Provenance records
`input_origin`, `source_kind`, authoritative URL/revision, license/attribution,
SHA-256, canonical S3 URI, and source→conditioning→frame derivation. The durable
workflow manifest, config/final reports, Rerun panel, and FiftyOne-backed dataset
view surface the same labels: “Upstream real sample,” “User-supplied input,”
“Synthetic seeded fixture,” “Derived conditioning clip,” and Cosmos variants.

Cosmos Transfer's upstream prompt and generated-video content guardrails remain
enabled by default. Some validated benign domains can fall outside the generic
video classifier's calibration set and produce no publishable output after a
successful diffusion. After an operator reviews that false positive, a single
run may explicitly set `NPA_COSMOS_DISABLE_CONTENT_GUARDRAILS=1`. NPA then passes
upstream's documented `--disable-guardrails` setup option and records
`content_guardrails_enabled: false` in Transfer metadata. This opt-out does not
weaken the downstream attribute, hallucination, temporal, protected-appearance,
or quality-disposition checks; it should not be made a shared default.

## Runtime placement

- **Token Factory (zero-GPU, hosted):** captioning, and the Cosmos Evaluator
  attribute-verification check's LLM + VLM calls.
- **GPU (Nebius Managed K8s):** Cosmos Transfer 2.5 augmentation only.
- **CPU:** config sampling, hallucination, temporal-consistency, and protected-
  appearance checks, Cosmos Curator curation, FiftyOne review, visualize, finalize.

### Live validation scope

The complete workflow was validated on reserved RTX PRO 6000 capacity with the
pinned RoboPro physical capture: eight real segmentation-conditioned Cosmos
Transfer 2.5 variants ran across four GPU nodes with upstream content guardrails
enabled. One candidate passed all four attribute checks plus hallucination and a
disjoint decoded-frame holdout at `0.907986` against the default `0.75`
threshold. The accepted branch then completed real Cosmos Curator and FiftyOne
Brain curation, finalized 320 artifacts, and wrote a 46,332,150-byte Rerun
recording. The workflow completed successfully after durable resume replayed the
already committed stages; no on-demand capacity or synthetic fallback was used.
Concrete project, cluster, and object-store identifiers remain private runtime
evidence.

The quality gate is fail closed. PAIDF first ranks every generated candidate,
then copies only candidates with explicit independently passing 4/4 attribute,
hallucination, threshold, and configured hard-check evidence into an additive
`selection/iteration-N/` batch. It re-evaluates that selection on deterministic
decoded holdout frames that are disjoint from the ranking beginning/middle/end
sample. Temporal consistency joins the hard checks only in calibrated `required`
mode; protected-appearance fidelity does the same when its mode is `required`.
The complete ranking pool remains unchanged. If refinement is exhausted,
`quality-disposition`
writes `grade/quality_disposition.json` with `quality_status: rejected` and stops
the workflow before labeling or curation. Workflow execution status and dataset
quality status therefore remain separate and auditable.

Refinement is adaptive by default. `prepare-refinement` writes
`configs/refinement.json` before each render and keeps immutable
`refinement-attempt-NN.json` copies plus commit markers. The established first-pass
control weight remains `1.0`; `augment_guidance` defaults to `3.0`. After a failed
evaluation the default retry lowers prompt guidance, while a custom baseline below
its configured ceiling may also
raise edge-control strength. Each retry selects a different in-bounds pair; once
the declared monotonic schedule is exhausted, refinement fails closed instead of
toggling back to a previous policy. A configuration with no possible first retry
fails before GPU work. Transfer metadata records the effective values and failed
check names and exact failed attribute names. Retry prompts emphasize only those
failed attributes with their candidate-specific requested values, so a retry is
not an unauditable replay of identical inference settings.
The planner validates Cosmos Transfer's native constraints before reserving a GPU:
edge-control weights stay within `0..1`, and guidance remains a non-negative
integer.

This baseline is a compatibility choice, not a claim that stronger structural
conditioning always improves evaluator score. Prior live refinement evidence was
non-monotonic, so PAIDF retains `1.0` for the first pass and changes guidance on
retry; operators should tune only from comparable A/B evidence. Cosmos precedence
is explicit CLI values, then `NPA_COSMOS_*` environment overrides, then the
validated run-scoped refinement artifact. The final artifact wins so ambient
worker settings cannot mutate a committed retry.

Edge control preserves structure and motion, not source color. Deployments that
must protect identity-bearing material colors can set
`protected_chroma_mode: source-chroma` and provide normalized rectangles through
`protected_chroma_regions_json`. Cosmos still generates the video, but the transfer
stage restores source Cb/Cr per pixel inside feathered protected regions and
limits generated luma to `protected_luma_max_delta` from the source pixel. Mild
exposure and illumination changes remain without copying source RGB pixels, while
extreme darkening or brightening is suppressed. Use
`protected_feather_pixels` to soften rectangle boundaries. The mode is off by
default: rectangles are a coarse MP4-only protection surface, and semantic masks
or simulator passes are preferable when available. A decode or frame-count mismatch
fails closed rather than publishing partially protected output.

For object-shaped protection instead of coarse rectangles, set
`segmentation_mode: sam2-auto`. It invokes the real upstream Meta SAM2 runtime
once per source clip, emits one binary PNG mask per frame under the versioned
`segmentation_uri`, and reuses those masks across all Cosmos variants and bounded
refinement retries. Automatic mode discovers stable first-frame foreground
proposals; raw prompt coordinates
are intentionally not accepted through workflow config or rendered argv. The
mask drives the same source-chroma/luma-bounded fidelity policy at pixel precision
with feathered boundaries. Missing frames, empty eligible proposals, checkpoint
mismatch, decode failure, empty/all-frame masks, invalid coverage, or upload
failure stops the augment stage.

The default is `segmentation_mode: off`, which neither downloads nor invokes
SAM2 and preserves the original PAIDF/Cosmos behavior. The official
`facebook/sam2.1-hiera-tiny` checkpoint is pinned by immutable revision for the
default speed/quality tradeoff; tune proposal thresholds and `max_objects` from
measured coverage and downstream evaluator results rather than assuming that
more masks improve a domain. The image replaces the Cosmos lock's unaffiliated
PyPI repackaging with Meta's immutable official source; its Apache-2.0 source,
BSD helper notice, and checkpoint boundary are recorded in the Cosmos Transfer
redistribution notice.

For an A/B comparison, give both fresh runs the same non-sensitive
`augmentation_seed`. Config generation will then order the same coherent,
nonconflicting appearance profiles and assign the same distinct per-candidate
diffusion seeds even though the run IDs differ. The first eight candidates cover
eight concrete lighting/backdrop/palette/finish combinations without replacement,
making evaluator and throughput deltas attributable to the optional component
rather than a different workload. An empty value retains deterministic
run-ID-derived sampling behavior. The maintained search table now provides eight
conservative, unambiguous appearance profiles before it repeats a profile with a
different diffusion seed. `quality_anchor_uri` can point at a preserved canonical
run; PAIDF derives the highest-scoring independently hard-passing candidate and
uses its recorded prompt variables, seed, control weight, and guidance as the
first search anchor without embedding scene-specific values in the workflow.

`protected_chroma_regions_json` is deliberately separate from
`appearance_regions_json`: the former changes transfer pixels, while the latter
only selects evaluator measurements. A deployment may use the same rectangles for
both, but PAIDF does not couple those policies implicitly.

Attribute verification remains an all-attributes hard check; its score still
contributes to the aggregate. A bare summary `passed` bit without its per-check
evidence is not selectable. An unavailable VLM marks evaluation `degraded` and
fails closed instead of falling back to a motion-only promotion.

Each curation/evaluation tool has its own CPU-only workbench image:

| Image | Stage | Needs |
| --- | --- | --- |
| `npa-cosmos-evaluator` | `evaluate` | `NEBIUS_TOKEN_FACTORY_KEY` (attribute verification only) |
| `npa-cosmos-curate` | `cosmos-curate` | conda-forge ffmpeg with `libopenh264`, baked in |
| `npa-fiftyone` | `curate` | bundled `mongod`; real Brain uniqueness/similarity/PCA |

Both curation stages fail closed: the Cosmos Curator command uses
`--require-curator`, and the following `workbench.fiftyone.curate_augmented`
toolRef must execute real FiftyOne Brain curation in the `npa-fiftyone` image.
Missing engines, unpublished output, or an invalid preceding report stop the
pipeline. Check what an environment resolves to with:

```bash
npa workbench cosmos-evaluator engine --output text
npa workbench cosmos-curate    engine --output text
```

## Model weights are never baked into the images

Upstream's *code* is Apache-2.0 and redistributable; its *weights* are not ours to
ship. So neither image contains any, a build-time check in each Dockerfile fails if
one appears, and both upstream checkouts are fetched with `GIT_LFS_SKIP_SMUDGE=1`
so Git-LFS payloads never enter a layer.

- **The evaluator needs no weights at all.** Its hallucination check is classical
  computer vision and its attribute verification calls a hosted VLM, so the image's
  golden eval runs with `--network none`. Upstream's objects/obstacle check would
  need an EULA-gated SegFormer ONNX plus a CWIP checkpoint; those stay as LFS
  pointers and the check is not wired. Using it means accepting upstream's EULA and
  fetching the weights yourself.
- **The curator's GPU stages do need weights**, so they are downloaded at run time
  with your own Hugging Face token into a volume that persists across runs:

```bash
docker run --rm -e HF_TOKEN=... -v curator-weights:/config/models \
  <registry>/npa-cosmos-curate:0.1.0 fetch-models --models split-annotate
docker run --rm -v curator-weights:/config/models \
  <registry>/npa-cosmos-curate:0.1.0 models --output text
```

`--models` takes a capability set (`split-transnetv2`, `embed-internvideo2`,
`embed-cosmos-embed1`, `filter-aesthetic`, `caption-qwen`, `dataset-t5`, or
`split-annotate` for upstream's `video-pipeline split` defaults) or a raw upstream
model key. The model ids and their pinned revisions come from upstream's own
registry (`cosmos_curator/configs/all_models.json`) and the download is upstream's
own `huggingface_hub` call, so a pin moves only when the pinned checkout does.

Every curator model is a Hugging Face repo, so `HF_TOKEN` is what fetches weights;
`NGC_API_KEY` is what pulls NVIDIA *containers*. `npa workbench cosmos-curate
models` reports which of the two is visible along with what is already on disk.

## Validate / plan / render

```bash
SPEC=workflows/testing/physical-ai-data-factory.yaml
npa workbench workflow validate-spec "$SPEC" --json
# --var bucket= is what ties the plan to your storage; without it the spec's
# `example-bucket` placeholder is planned (plan-spec warns when that happens).
npa workbench workflow plan-spec "$SPEC" --run-id demo \
  --assume-decision promote_checkpoint --var bucket=<your-bucket> --json
# Render the serial SkyPilot YAML without launching (needs NPA_SRC_S3_URI or --image
# for the CPU tool steps, same as the other Token Factory specs):
NPA_SRC_S3_URI=s3://<your-bucket>/npa-src/npa/ \
  npa workbench workflow submit "$SPEC" --run-id demo --var bucket=<your-bucket> \
    --assume-decision promote_checkpoint --plan-only
```

## Submit (real run)

```bash
SPEC=workflows/testing/physical-ai-data-factory.yaml
npa workbench health access --capability paidf
npa workbench workflow preflight-images "$SPEC" \
  --project <alias> --registry <registry>
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project <alias>)"
npa workbench workflow submit "$SPEC" \
  --project <alias> --run-id "$RUN_ID" \
  --var bucket=<your-bucket> \
  --runtime --auto-load \
  --assume-decision promote_checkpoint \
  --infra k8s/<your-kube-context> \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN
```

`--var bucket=` points `config.bucket` at the artifact bucket your NPA agent
reads; without it the run uses the spec's `example-bucket` placeholder.
Submit automatically fingerprints and publishes the local `npa` package to a
content-addressed prefix below `s3://<your-bucket>/npa-src/npa/`, verifies the
commit manifest, and persists the exact URI for later shells. Retries reuse it;
`stage-src` remains an advanced explicit command, not a happy-path requirement.
After successful PAIDF completion, `--auto-load` verifies the exact final Rerun
URI through the configured agent; `workflow load-artifact <run-id>` retries only
that optional handoff. Submit checks prerequisites up front and prints anything
still missing in a single list; the
[deploy runbook](physical-ai-data-factory-deploy.md) has the full ordered
quickstart (`configure` → `skypilot bootstrap` → `provision-if-absent` → submit).
The PAIDF submit preflight also requires one Ready, schedulable, appropriately
untainted node with 6 CPU and 24 GiB allocatable for the SkyPilot controller and
one CPU stage. A qualifying GPU node is valid because the CPU profile has no
GPU exclusion. The check uses the exact submission context before S3/input/source
mutation, verifies gated Cosmos Transfer access with the forwarded `HF_TOKEN`,
and reuses the generic image-pull check/secret refresh.

Monitor an exact run using NPA alone:

```bash
npa workbench workflow status <run-id> --project <alias>
# Explicit fallback; the final manifest itself may still be pending:
npa workbench workflow status <run-id> --project <alias> \
  --workflow-s3-uri s3://<bucket>/physical-ai-data-factory/<run-id>/npa-workflow

# Intentional offline inspection (exit 0 but never live-verified):
npa workbench workflow status <run-id> --project <alias> --cached

# After DNS/controller recovery, explicitly resume the same run:
npa workbench workflow submit "$SPEC" --project <alias> \
  --resume-run <run-id> --var bucket=<your-bucket> --runtime \
  --assume-decision promote_checkpoint --infra k8s/<your-kube-context>
```

The launch path does not treat the earlier cluster/GPU snapshot as controller
readiness. Immediately before each Kubernetes controller launch it requires a
stable series of API `/readyz` observations using the exact selected context and
SkyPilot environment. A transient refusal is reconciled first: NPA adopts an
exact job if the request landed, retries only after authoritative absence, and
blocks as indeterminate when structured queue evidence is unavailable.
Scheduler-managed Cosmos publication is stricter for both one-node and gang
stages: an inner replacement cannot supersede an existing same-token claim (but
may safely be the first claimant if the prior worker died before claiming). A
configured NPA runtime retry receives a higher ordered attempt only after the prior
managed job is terminal. `--resume-run` remains the driver crash/restart recovery
contract.

Status, logs, artifacts, and cancel share the same precedence: explicit URI,
owner-only per-project/run submission receipt, exact canonical PAIDF prefix,
exact pinned SkyPilot managed-job evidence, then the ordinary workflow layout.
Text and JSON show each checked source. A provider/auth error is
`VERIFICATION_UNAVAILABLE`, never “not found”; `NOT_FOUND` requires authoritative
absence from every applicable source. Runs found before their final manifest
remain actionable with `manifest_state: pending` plus scheduler, active-stage,
retry, heartbeat, failure, and log-command fields.

For a persisted run, `manifest_state: available` says only that the manifest was
read. The same runtime ledger supplies exact stage/attempt/job attribution to
status, logs, cancel, receipts, and the agent; cached/object-storage logs and live
scheduler logs are reported separately. `last_observed_at` advances on a poll,
while `last_heartbeat_at` advances only on real task progress. A DNS, RBAC, auth,
timeout, context, controller, or parse failure therefore leads with
`VERIFICATION_UNAVAILABLE`, exits nonzero, preserves a labeled last-known state,
and shows the unchanged heartbeat as stale plus an NPA retry command. It is not
itself a terminal workflow failure.

Runtime JSON records `logical_launch_id`, `launch_sequence`, `error_category`,
readiness samples, reconciliation outcomes, recovery decision, immutable adopted
job ID, and cancellation state. Cancellation state is one of `requested`,
`verified`, `failed`, or `not_applicable`; it is never inferred from merely
issuing a cancel command, and no cancellation is attempted without an exact job
ID.

PAIDF images are submitted using the digest verified during bootstrap preflight.
Use repeatable `--image-override TOOL_REF=IMAGE` arguments when validating
separately tagged Transfer, Evaluator, Curator, and FiftyOne images; the exact
tool override wins over a global `--image` fallback and is digest-pinned before
rendering.
The FiftyOne stage stays non-root; UID-0 pod overrides are rejected. After
completion, exact current-ledger, durable S3 manifest/artifact, immutable job,
and planning evidence share one precedence. A complete current-schema ten-wave
run remains terminal after active jobs disappear. `NOT_SUBMITTED` requires proof
that no launch occurred and cannot override later evidence. Exact durable
conflicts are typed as inconsistent; explicit resume skips every succeeded wave
and launches nothing.

Provisioning, inference, and curation durations are capacity/workload dependent,
not guarantees. For practical warm/cold ranges and recovery guidance, see
[the deploy runbook](physical-ai-data-factory-deploy.md#5-submit-the-physical-ai-data-factory-workflow).

> **Input is prepared before GPU work.** With no selector, submit verifies and
> stages the pinned real RoboPro starter. `--input-video` and `--input-uri`
> replace it; `--seed-fixture` is the only geometric synthetic path. Every path
> produces the exact conditioning clip and caption frames under the canonical
> run prefix, with fail-closed integrity/media validation and provenance. See the
> [deploy runbook](physical-ai-data-factory-deploy.md) (“Select the starter input”).

## S3 artifact layout (agent-viewable)

```
s3://<bucket>/physical-ai-data-factory/<run-id>/
  input/               # source.mp4 + conditioning clip/frames + provenance
  configs/             # Stage 1 manifest + adaptive refinement provenance -> json
  labeled_original/    # Stage 2a VLM captions                  -> json
  cosmos_augmented/    # append-only candidates by iteration    -> video / image / json
  selection/           # additive hard-pass holdout batches     -> video / json
  grade/               # ranking + holdout reports, decisions, disposition -> json
  review/
    fiftyone-dataset/  # portable real FiftyOneDataset + media, every terminal run
    fiftyone-review.json # review-only/accepted fields and preservation proof
    decision.json      # post-review route; canonical quality decision is unchanged
  labeled_augmented/   # Stage 3 VLM captions on augmented      -> json
  curation/
    cosmos_curator/    # Cosmos Curator output tree             -> video / json
      clips/           #   transcoded clips
      metas/v0/        #   per-clip metadata + motion scores
      processed_videos/
    cosmos_curator.json# Cosmos Curator run summary             -> json
    report.json        # FiftyOne review report (merges the above) -> json
  reports/sim2real.rrd # Rerun recording (input+augmented+captions) -> rerun
  reports/final.json   # finalize summary                       -> json
```

The `visualize` stage builds `reports/sim2real.rrd` from the run's input +
augmented frames and captions (via `npa.workflows.data_factory_viz.build_run_rrd`)
so the run renders in the NPA agent's **embedded Rerun viewer** — the agent
prefers `reports/sim2real.rrd`, so selecting the run and loading it (or clicking
the `.rrd` in the artifact browser) shows it in the Rerun panel.

After refinement, `quality-disposition` branches on the persisted final result.
Accepted runs continue through re-captioning and curation before `visualize`.
Every terminal run first exports every committed candidate to a portable real
FiftyOne dataset. Rejected samples are labeled `review-only`, expose score,
per-attribute results, hallucination status, iteration, and candidate identity,
and always have `promotion_eligible=false`; accepted-only relabeling, Cosmos
Curator, Brain selection, and finalization remain skipped. Rejected runs then
execute `visualize-rejected`, which embeds each committed candidate's actual PNG
and MP4 components plus a `REJECTED` evidence panel in the RRD before
`reject-quality` fails the workflow. A failure before any usable input or
augmented media exists still cannot produce an RRD.
The accepted branch begins with `require-accepted-quality`, which re-checks the
durable disposition. This additional guard is what keeps a one-shot serial plan
using a preview assumption from running accepted-only stages when the real report
was rejected; runtime execution follows the same guarded branch.

PAIDF uses Segment Anything only when `segmentation_mode` explicitly selects
`sam2-auto`. These frame-aligned protected-content masks are
distinct from Cosmos Evaluator's comparison masks and from Token Factory VLM
captions. The default/off path remains entirely non-SAM.

## View input / intermediate / output in the NPA agent

The agent discovers runs from its artifact bucket. Discovery paginates the full
configured source and prefers a provable complete strict-superset canonical run
over a same-ID one-file viewer mirror. Divergent duplicates stay ambiguous and
require the returned source-qualified `run_ref`; publication must never replace
or remove the canonical prefix. If the agent's base prefix is
`checkpoints`, place the run under `checkpoints/physical-ai-data-factory/<run-id>/`
(or pass the matching discovery prefix). Then:

```bash
# discover runs
GET /api/artifacts/runs?prefix=physical-ai-data-factory
# list a run's artifacts (render hints: video / image / json / text)
GET /api/artifacts/run/<run-id>?prefix=physical-ai-data-factory
# load one artifact into the viewer (exact URI or run-relative key)
POST /api/sim-viz/load-artifact  {"run_id":"<run-id>","key":"reports/sim2real.rrd","prefix":"physical-ai-data-factory"}
```

Input clips render as `video`, extracted frames as `image`, and every stage's
labels/reports as `json` — so the full input → intermediate → output flow is
browsable in the agent.

Load the portable review archive into an existing durable Voxel51 deployment
with `npa workbench fiftyone load-dataset --format fiftyone`; the loader imports
`fo.types.FiftyOneDataset` with its media, marks the dataset persistent, and makes
the disposition fields queryable after workflow compute has exited.

> **An RRD needs decodable input or augmented frames.** With an empty `input/` the
> run stops at `annotate-original` (only `configs/manifest.json` is written), so
> there is no frame evidence from which to build `reports/sim2real.rrd`. Once
> augmentation has produced frames, both accepted and quality-rejected paths
> materialize the recording; only accepted runs include the downstream curation
> and final report panels.
