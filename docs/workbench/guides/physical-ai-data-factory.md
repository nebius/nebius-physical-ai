# NVIDIA Physical AI Data Factory on NPA (no OSMO)

This guide runs the **NVIDIA Physical AI Data Factory** blueprint natively on
Nebius + SkyPilot. It is delivered as a single npa.workflow spec
(`npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml`) that
**composes existing workbench tools** — there is no OSMO orchestrator and no new
"data factory" tool. SkyPilot is the sole orchestrator; every stage hands off
through one S3 run prefix so input, intermediate, and output artifacts are all
viewable in the NPA agent artifact browser.

> **Want the from-zero runbook?** See
> [physical-ai-data-factory-deploy.md](physical-ai-data-factory-deploy.md) for a
> copy-paste deploy guide: install `npa`, set credentials/config, deploy the NPA
> agent, submit this blueprint (including multi-GPU fan-out and real FiftyOne
> curation), and view results in the agent.

## Blueprint mapping

NVIDIA blueprint (OSMO) → NPA stage (toolRef / run):

| NVIDIA stage | NPA state | Tool | Runtime |
| --- | --- | --- | --- |
| Stage 1 Config Generation | `generate-configs` | `run.shell` (sample appearance-only variables) | CPU |
| Stage 2a Understand & Annotate | `annotate-original` | `workbench.token_factory.caption` | Token Factory (zero-GPU) |
| Stage 2b Augment & Multiply | `augment` | `workbench.cosmos2.transfer_execute` | GPU (Cosmos Transfer 2.5) |
| Evaluate & Validate | `grade` loop + `quality-disposition` | Cosmos Evaluator + NPA source-relative temporal/appearance fidelity + fail-closed disposition | Token Factory + CPU |
| Stage 3 Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Stage 4a Curation | `cosmos-curate` | `workbench.cosmos_curate.curate` | CPU |
| Stage 4b Curation review | `curate` | `workbench.fiftyone.curate_augmented` (required FiftyOne Brain) | CPU |
| Visualize | `visualize` | `data_factory_viz.build_run_rrd` | CPU |
| Finalize | `finalize` | `data_factory_stages.finalize` | CPU |

Three NVIDIA components in that table are the real open-source projects:

- **Cosmos Transfer 2.5** augments (GPU diffusion, `npa-cosmos2-transfer` image).
  It is **not** a Token Factory model.
- **[Cosmos Evaluator](https://github.com/nvidia-cosmos/cosmos-evaluator)**
  (Apache-2.0) grades. The `evaluate` stage runs two of upstream's checks per
  variant: *attribute verification* (an LLM writes one multiple-choice question
  per sampled attribute, a VLM answers it from a frame — both on Token Factory,
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

**Model roles** (verified available on Nebius Token Factory):

- VLM captioning + the evaluator's attribute answering: `Qwen/Qwen2.5-VL-72B-Instruct`
- Cosmos-family reasoning critic: `nvidia/Cosmos3-Super-Reasoner`
- Prompt / MCQ LLM: `meta-llama/Llama-3.3-70B-Instruct`

> **Config → augment MULTIPLY.** The `augment` stage receives the Config-Gen
> manifest via `--configs-uri` and runs **one Cosmos Transfer 2.5 inference per
> sampled combo**: each combo's prompt drives a distinct appearance, published as
> its own per-clip dir (`cosmos_augmented/<clip>/`) with its own `metadata.json`
> `variables` (which drives that clip's Rerun label). So a config with N
> augmentations produces **N scenario variants**, recorded via `variant_count` /
> `multiply_mode` in the augment manifest, curation, and finalize reports.
> The managed transfer conditions every variant on a supported video under the
> run's `config.trigger_uri` (`input/`), preserving geometry/motion while changing
> appearance (edge control is computed on-the-fly).

### Input conditioning and its evaluation source

Before image checks or automatic provisioning, `workflow submit` validates and
stages a source, normalizes it to an H.264/yuv420p 1280×720 clip of exactly 93
frames at 16 fps, and extracts eight caption frames. The catalog always invokes
`workbench.cosmos2.transfer_execute` with `--condition-on-input` (equivalent to
`NPA_COSMOS_CONDITION_ON_INPUT=1`), and the resolver explicitly prefers
`input/conditioning.mp4` over `source.mp4`. Cosmos Evaluator uses that same clip
for hallucination scoring. Missing or invalid input therefore cannot turn into a
decorative staged object or a fixed control example.

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

Selection precedence is strict: one explicit local or S3 object wins; conflicting
selectors fail; no selector chooses the starter; `--seed-fixture` is the only
synthetic geometry path. User input is labeled “User-supplied input”—NPA does not
invent an authenticity or license claim for it.

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

# Developers/tests only — explicitly synthetic
npa workbench workflow submit "$SPEC" --run-id "$RUN_ID" --runtime \
  --var bucket="$BUCKET" --seed-fixture --assume-decision promote_checkpoint
```

Those are alternatives: run one with the prepared fresh ID. A previous run is
resumed only with an explicit `--resume-run "$RUN_ID"`; an unattended command
never reads or silently reuses the legacy global `~/.npa/paidf-first-run-id`.

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

The quality gate is fail closed. Promotion requires every variant to pass attribute
verification and, for input-conditioned variants, hallucination checking, plus the
aggregate threshold. Temporal consistency joins those hard checks only in calibrated
`required` mode; protected-appearance fidelity does the same when its mode is
`required`. If refinement is exhausted, `quality-disposition`
writes `grade/quality_disposition.json` with `quality_status: rejected` and stops
the workflow before labeling or curation. Workflow execution status and dataset
quality status therefore remain separate and auditable.

The all-variant batch policy is intentionally conservative: the reference workflow
does not yet quarantine failed variant directories before downstream labeling and
curation, so it will not promote a mixed-quality prefix. A future partial-promotion
mode must first route only accepted variants into a separate downstream prefix.
Attribute verification remains an all-attributes hard check; its score still
contributes to the aggregate. An unavailable VLM marks evaluation `degraded` and
fails closed instead of falling back to a motion-only promotion.

Each curation/evaluation tool has its own CPU-only workbench image:

| Image | Stage | Needs |
| --- | --- | --- |
| `npa-cosmos-evaluator` | `evaluate` | `NEBIUS_TOKEN_FACTORY_KEY` (attribute verification only) |
| `npa-cosmos-curate` | `cosmos-curate` | conda-forge ffmpeg with `libopenh264`, baked in |
| `npa-fiftyone` | `curate` | bundled `mongod`; real Brain uniqueness/similarity/PCA |

Without the curator image the stage records `engine: unavailable` plus the reason
and the required FiftyOne review still runs. If FiftyOne Brain is unavailable,
the `curate` stage fails closed rather than presenting report-only counts as a
FiftyOne review. Check what an environment resolves to with:

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
SPEC=npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
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
SPEC=npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
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
blocks as indeterminate when structured queue evidence is unavailable. A
recovered launch continues inside the same command; `--resume-run` remains the
crash/restart recovery contract.

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
  configs/             # Stage 1 sampled augmentation manifest  -> json
  labeled_original/    # Stage 2a VLM captions                  -> json
  cosmos_augmented/    # Stage 2b augmented clips + metadata    -> video / json
  grade/               # evaluator, decision, quality disposition -> json
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

## View input / intermediate / output in the NPA agent

The agent discovers runs from its artifact bucket. If the agent's base prefix is
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

> **Dataset provenance and the full Rerun recording only appear once the run gets past
> annotate → augment → curate → visualize.** With an empty `input/` the run stops
> at `annotate-original` (only `configs/manifest.json` is written), so there is no
> `cosmos_augmented/`, no `curation/report.json` for the dataset tab, and no
> `reports/sim2real.rrd` for the Rerun panel. Stage input frames first (see the
> callout above) so the pipeline reaches curate/visualize and those panels
> populate.
