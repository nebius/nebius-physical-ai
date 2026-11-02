# PAIDF with Cosmos 3 video conditioning

For whole-dataset submission, read-only robot input checks and private source/image integration,
see [PAIDF dataset batches](paidf-dataset-batches.md).

[Guides](README.md)

For the current generic single-run lifecycle and exact installed workflow
flags, start with [manual workflow operations](manual-workflow-operations.md).
It covers saved private configuration, `--var` overlays, preflight, durable
inspection, resume, scoped storage transfers, and the limits of batch and
multiview claims.

For a complete manual setup and run, start with the
[PAIDF Cosmos 3 setup and run guide](../../../workflows/guides/paidf-cosmos3.md). It covers
project setup, new or existing Kubernetes clusters, credential-file formats,
Token Factory key creation, Python and CLI installation checks, submission, and
artifact inspection.
For twelve candidates using the current workflow defaults, follow
[the count-only fanout](#twelve-variants-with-the-current-defaults) below.
For twelve separate appearance videos with task-specific profiles and automatic padding preservation, use
the [copy-paste recipe guide](paidf-appearance-12.md#apply-the-recipe).
Use this page for the workflow's input, generation, and acceptance contracts.
For the configuration keys and timing contract, see the setup guide's
[generation and evaluation settings](../../../workflows/guides/paidf-cosmos3.md#r5-find-and-change-generation-and-evaluation-settings).
For rejected runs, see its
[quality-rejection diagnostics](../../../workflows/guides/paidf-cosmos3.md#r6-diagnose-quality-rejection-and-prepare-the-next-run).
For an older installation, see
[upgrading an existing installation](../../../workflows/guides/paidf-cosmos3.md#upgrading-an-existing-installation):
new submissions stage the current NPA source automatically unless explicitly
overridden, unchanged datasets need no bucket resync, and executing submits
must omit `--assume-decision promote_checkpoint`. Check the Git SHA and import
path because the package version remains `0.1.0`.

`workflows/main/paidf-cosmos3.yaml` is an independent
Physical AI Data Factory composition. It does not replace or change
`physical-ai-data-factory.yaml`, whose augmentation engine remains Cosmos
Transfer 2.5.

The default Cosmos3, Evaluator, and Curator task images now select repaired,
digest-pinned public development candidates. See the
[image-selection contract](physical-ai-data-factory.md#runtime-placement)
for provenance, validation scope, and explicit image overrides. Historical
release tags remain quarantined while release qualification is pending.

The pipeline is:

1. select one generic MP4 or one camera from one LeRobot v2/v3 episode;
2. retain the original, prepare a constant-rate letterboxed reference, and caption it with Token Factory;
3. run native NVIDIA `cosmos-framework` edge transfer across the complete source for each variant;
4. verify corresponding timestamps and hashes, then grade every variant with real Cosmos Evaluator checks;
5. promote on a complete passing report, or retry with changed seed, guidance,
   and steps up to the configured bound;
6. caption accepted variants, then run real Cosmos Curator and FiftyOne Brain;
7. write a real Rerun recording and a fail-closed aggregate report.

Rejected runs skip labeling, Cosmos Curator, FiftyOne, and finalization. They
first write the available input, generated-video, evaluator, decision, and
quality-disposition evidence to `reports/quality-evidence.rrd`, then terminate with a
failure. Missing or incomplete evaluator reports also reject.

This workflow declares `metadata.executionMode: runtime`. The generic submit
command therefore selects the runtime orchestrator even when `--runtime` is
omitted, so every failed evaluation reaches the next bounded refinement pass and
the terminal disposition controls the final branch. An explicit `--no-runtime`
is rejected before staging or submission. `--assume-decision` is allowed only
for planning previews; execution requires actual evaluator decisions.

## Twelve variants with the current defaults

Complete the [operator setup](../../../workflows/guides/paidf-cosmos3.md#before-you-start)
and retain its verified project, bucket, GPU selection and Kubernetes context.
If setting `NPA_WORKFLOW_GPU_ACCELERATOR`, use the exact product name printed by
`npa workbench workflow gpus --cluster "$KUBE_CONTEXT"`; explicit overrides are
passed literally to SkyPilot.
The canonical workflow needs only `--var variant_count=12` to generate twelve
candidates from the pinned starter video. It retains the current prompts,
sampler, 24 base steps, native edge controls, guardrails and exploratory quality
thresholds. A quality-gate retry follows the shipped refinement policy, which
changes the attempt's seed, guidance and step count.
The separate [twelve-profile recipe](paidf-appearance-12.md) changes
those settings and is a different experiment.

Inline CPU stages use the existing pinned Rerun workbench image, including its supported
Python and media dependencies. This keeps source installation compatible with
current NPA, which requires Python 3.11 or newer. The submitted source also
registers Token Factory captioning in that image's light workbench CLI.

From the checkout installed in `npa/.venv`, run:

```bash
SPEC=workflows/main/paidf-cosmos3.yaml
RUN_ID="$(npa/.venv/bin/npa workbench workflow prepare-run "$SPEC" --project "$PROJECT_ALIAS")"
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$HOME/.npa/workflow-runs/$RUN_ID/skypilot"
export NPA_E2E_PAIDF_FANOUT_FRESH_AFTER="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  --project "$PROJECT_ALIAS" --run-id "$RUN_ID" --runtime --max-wait-seconds 0 \
  --infra "k8s/$KUBE_CONTEXT" --var bucket="$BUCKET" --var variant_count=12 \
  --secret-env HF_TOKEN --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Omitting an input selector uses the verified starter. Add `--input-video`,
`--input-uri`, or the documented LeRobot selectors to use your own source;
the starter-specific audit below then does not apply. Twelve candidates run
sequentially with the shipped `variant_parallelism: 1` and one-GPU task.
Increasing the candidate count does not request twelve concurrent GPUs.
`--max-wait-seconds 0` leaves the complete generation wave unbounded, as in
the setup guide; the shared runtime's one-hour default can expire a large batch.
Each candidate is evaluated, and every accepted candidate receives its own
captions before real Cosmos Curator, FiftyOne and finalization run. Actual
evaluation controls refinement and rejection; count alone never guarantees
acceptance.

After successful finalization, audit the starter run without launching work:

```bash
NPA_INTEGRATION_E2E=1 NPA_E2E_PROJECT="$PROJECT_ALIAS" \
  NPA_E2E_PAIDF_FANOUT_RUN_URI="s3://$BUCKET/paidf-cosmos3/$RUN_ID/" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_paidf_cosmos3_fanout_live.py -q
```

Require one passed test with no skip. The audit verifies fresh starter bytes,
all completed stages, exactly twelve configurations, published variants,
evaluated clips and annotated variants, unchanged default sampling and quality
settings, full-video alignment, effective model guardrails, real curation and
both Rerun recordings. Keep the pre-submission timestamp and private run state
for this read-only audit and the setup guide's scoped cleanup.

## Native variant recovery

The standard immutable-image `Cosmos3-Nano` workflow with
`structural_control: edge` retains each completely published variant across
worker recovery. It resolves the actual Hugging Face revision once and seals
that full revision with the run, prepared input bytes, captions, configuration,
effective prompts, seeds, sampling controls, source fingerprints, immutable
image request and measured GPU model/driver/capability. A changed contract or a
corrupt completed receipt fails closed. Existing progress documents and legacy
partials do not qualify for reuse.

The native pipeline keeps the explicit Cosmos3-Nano YAML and eager sampling.
The sound tokenizer is selected from that same pinned Nano snapshot using the
native [`from_checkpoint` path](https://github.com/NVIDIA/cosmos-framework/blob/5e67049cd94acb667786f1e6dd0dab821cb90c97/cosmos_framework/inference/inference.py#L1163).
This invokes the same [AVAE converter](https://github.com/NVIDIA/cosmos-framework/blob/5e67049cd94acb667786f1e6dd0dab821cb90c97/cosmos_framework/inference/common/checkpoints.py#L61)
used by the default sound download; converted tensor names, dtypes, shapes and values must
match the pinned source tensors. The separately registered Wan VAE remains on
its native immutable revision. All downloaded snapshot files, auxiliary bytes,
native source, derived sound tensors and the actual post-construction model
configuration are hash bound. The model-selection receipt states that its basis
is the native configuration and resolver contract; it does not claim loader-open
instrumentation.

Each clip is published under its batch and final published-video SHA, with
conditional creation and byte readback for the complete video, canonical
numbered PNGs, edge and optional RGB controls, metadata, native execution and
transfer reports. When source padding is restored, the same immutable object
set retains `raw_model_video.mp4` and its metadata: the native guarded hash
binds the raw bytes while the publication directory and descriptor bind the
final padding-preserved bytes. Completion is sealed last. Recovery verifies
both lineages, every object, and the full final video against the same source
timeline before skipping that variant. The normal collector writes the
canonical manifest after all requested variants complete;
`recovered_variant_count` reports how many verified variants that invocation
reused. Guardrails and evaluator thresholds retain their existing behavior.

This path requires the normal workflow's `NPA_TASK_IMAGE` immutable reference.
Direct calls lacking that identity, mutable image references, non-structural
generation, and custom checkpoints retain fresh generation. Set
`variant_recovery: disabled` (or `--variant-recovery disabled`) to deliberately
regenerate rather than reuse after an operational runtime change. An unchanged
immutable batch remains fail-closed: a changed GPU runtime rejects reuse and
points to that explicit regeneration mode. `NPA_COSMOS3_NANO_REVISION` is an
internal handoff set by the coordinator, not an operator model-selection option.
Stage the reviewed checkout through normal workflow submission. Keep native
acceptance pending until a fresh candidate-source run passes the
[read-only live receipt audit](../../../npa/tests/e2e/README.md#paidf-native-variant-recovery);
fixture interruption tests establish the recovery protocol and do not prove
model inference or training-data suitability.

## Inputs and configuration

For `workflow submit`, use `--input-video` / `--input-uri` for MP4, or
`--lerobot-uri` with `--lerobot-camera`, `--lerobot-episode`, and
`--require-explicit-lerobot-selection`. The setup guide includes a complete
[LeRobot submission and pinned public example](../../../workflows/guides/paidf-cosmos3.md#r3a-augment-one-lerobot-episode-and-camera).
The LeRobot CLI selector takes an S3 dataset prefix; upload local files while
preserving the dataset directory layout first.

For direct stage configuration, choose `input_kind: video` and set
`input_video_uri` to one MP4, or choose
`input_kind: lerobot` and set `lerobot_dataset_uri`, `input_episode`, and
`input_camera`. Dataset URIs may point to generic LeRobot v2.x or v3.x directory
trees. Use a full feature name such as `observation.images.front` for submission.
The local direct-stage selector also accepts an unambiguous camera suffix.
The S3 worker reads metadata and fetches only the selected video. Metadata
shards are searched independently of episode numbering. Shared v3 video files are trimmed
using the episode metadata timestamps; per-episode v2 video layouts are also
supported.

The result is augmented video and review artifacts for one episode/camera, not
a reconstructed LeRobot dataset containing action/state records.

For extending this data pipeline into training and policy evaluation, see the
[Cosmos 3 model factory assessment](../../architecture/cosmos3-model-factory.md).
It maps current NPA capabilities and defines the
dataset, checkpoint, and simulation evidence needed for each proposed addition.

The committed `example-bucket` and run-scoped fixture path are placeholders.
The generic workflow submit command stages a verified, pinned starter video
when no input is supplied; use `--input-video` or `--input-uri` for your own
source as shown in the setup guide. Direct stage execution still requires a
staged input. Both starter variants passed the exploratory default quality gate
in the recorded live run. Training-data suitability requires separate assessment,
and another run can still be rejected. No customer dataset, episode, camera,
bucket, or infrastructure identifier is embedded.

Generation behavior is configuration-driven through `cosmos3_checkpoint`,
`cosmos3_mode`, `seed`, `guidance`, `steps`, `variant_count`,
`variant_parallelism`, `parallelism_preset`, and `variant_recovery`.
`augmentation_seed` defaults to `30`, keeping appearance profiles consistent
across fresh run IDs; change it for new appearance experiments. Quality and retries use
`grade_threshold`, `attribute_threshold`, `refinement_iterations`, `retry_seed_stride`,
`retry_guidance_delta`, and `retry_steps_delta`. `source_motion_weight` is a
compatibility setting that must be `0.0` (the default). Nonzero values fail
before generation. Publication preserves every generated scene pixel without
source/model blending, resizing or changing the frame rate. When verified
padding exists, it restores those borders from the prepared source and writes
a lossless `augmented_video.mp4`, retaining `raw_model_video.mp4` separately.
The preservation receipt records the raw and published hashes and exact pixel
checks. Without verified padding, the model bytes pass through unchanged.
See [padding behavior and output checks](paidf-appearance-12.md#7-download-the-videos-and-verify-padding-preservation).
The check lives in the shared generation implementation, so custom workflows
using `workbench.cosmos3.generate_variants` and direct CLI callers receive the
same protection for any dataset. New submissions of the canonical workflow use
the current checkout through `source_overlay: true`; custom or direct deployments
must also use the updated NPA code. The older publisher's `source_motion_weight=0`
disables blending but does not add automatic padding detection or preservation.
The composition requires
`video2video`: selecting a text-to-video or image-to-video mode fails before GPU
inference rather than producing a misleading source-conditioned claim.

`caption_model` defaults to `MiniMaxAI/MiniMax-M3` for the original
and accepted-variant captions and for Cosmos Evaluator's visual questions.
Token Factory availability is key-scoped and can change, so run
`npa workbench token-factory models` before execution and override
`caption_model` with another reachable vision model when needed.

The canonical workflow enables `structural_control: edge`. The optional
`transfer_edge_threshold` selects a native Canny preset from `very_low`, `low`,
`medium` (default), `high`, and `very_high`; the native control receipt records
the actual preset and verified pixel hash. Lower thresholds retain weaker
edges, which can help expose small or dark features, but also admit noise.
Inspect the controls and qualify the generated output independently.
`transfer_rgb_weight` optionally adds native RGB conditioning to retain source
color and surface cues that edges omit. Zero (the default) disables it; a
positive weight is relative to edge weight 1. This uses the framework's `blur`
hint with its `none` preset and verified lossless RGB controls. It changes model
conditioning, never output-pixel blending, and can suppress the requested edit.
`transfer_first_chunk_conditional_frames` separately controls whether the original
RGB first frame anchors generation. Its compatibility default is 1; set 0 with
edge transfer to test a changed appearance from the first frame. Complete source
edges remain active, and later windows retain five generated overlap frames.
Releasing the first-frame anchor may reduce object-identity preservation; review
paired clips before adopting it. Native receipts record both conditioning counts.
`conditioning_fps`
defaults to 24; preparation letterboxes to 832×480 and preserves duration within
one prepared frame. `transfer_chunk_frames` defaults to 93 and `control_guidance`
to 1.5. Native chunks cover every prepared source frame, including a final
partial interval. The adapter verifies lossless edge-control readback, checks
every effective prompt, and saves the video guardrail's postprocessed output.
`alignment_mode: required` independently verifies complete decoding, frame
counts, timestamps and the generated/source hashes before quality scoring.
The workflow's exploratory `grade_threshold` and `attribute_threshold`
default to `0.2` and `0.25`; reports retain individual failed attribute checks. For
stricter acceptance, explicitly set `0.75` and `1.0` respectively. These quality
settings do not weaken complete decoding, alignment or model guardrails.
These controls apply to this workflow's prepare/generate-variants commands;
the generic `cosmos3 generate` command retains its existing mode behavior.

Guardrails are enabled and enforced for this composition. The image contains the
pinned OpenMDW-1.1 framework source but no weights. The operator supplies
`HF_TOKEN` at runtime after accepting the checkpoint, Wan VAE, and
`nvidia/Cosmos-Guardrail1` terms. Tokens and model weights are never serialized
into artifacts or Git.

### Double exposure in older runs

Older workflow defaults blended 80% source pixels with 20% generated pixels.
Because the model can move the camera, cloth, or grippers, this superimposed
different scenes and produced translucent duplicate objects. Alpha blending
does not align motion. Existing run artifacts and their rejection reports stay
unchanged; retrieve each variant's `raw_cosmos_video.mp4` into a fresh output
location to inspect the unblended result without repeating GPU generation.
Those bytes require their own evaluation before acceptance.

Earlier prefix-conditioned generation could also produce a fixed-length segment
with timing unrelated to the full source. The canonical workflow now uses
native structural controls across the complete prepared video. Inspect visual
identity, motion and contacts independently of successful timeline validation;
lowering a quality threshold changes acceptance criteria without improving pixels.

### Task-specific appearance profiles

`appearance_profiles_json` defaults to an empty string (the starter profiles).
Supply a JSON array with `lighting`, `background`, `color_grade`, and
`surface_finish` in every profile to select plausible edits for the chosen
camera and task. The manifest retains those profiles and evaluator choices.
Source captions sampled across the episode are separated from the requested
appearance edit in the effective prompt. `caption_instruction` supplies task
context from `augment_subject` and asks for uncertainty when features are unclear;
override it for the selected camera and inspect the resulting captions. See the
[realistic manipulation guide](paidf-realistic-augmentation.md) for battery
insertion, controlled sampling experiments and review criteria.

## Artifact contract

Every successful generation pass preserves the downstream layout:

```text
cosmos_augmented/
  manifest.json
  variant-0000/
    augmented_video.mp4
    frame-00001.png
    metadata.json
    source_edges.mkv
    transfer.json
```

Each successful variant is published as soon as generation and alignment checks
finish. `cosmos_augmented/generation-progress.json` records the requested,
published and failed counts, with clip IDs and failure phases. A blocked or
failed variant still fails the stage; the batch does not write a new successful
`manifest.json`. Other completed variants remain available as review evidence.
Consumers must use the committed manifest, never infer a complete batch by
listing the prefix. Progress evidence is not a training-data promotion signal.
The immutable Nano structural path verifies new completion receipts before
reusing variants within a recovered stage. Other paths retain workflow-stage
recovery and regenerate their variants; progress documents alone never skip work.

Concurrent generations lease distinct available GPUs. A faster variant can
release its GPU to the next waiting variant without assigning that work to a GPU
still occupied by another generation.

Each metadata file records the real engine (`nvidia-cosmos/cosmos-framework`),
`video2video` mode, source-video conditioning, checkpoint, seed, guidance,
steps, attempt number, guardrail posture, non-baked weights, and input lineage.
It records `motion_preservation: null` and the published model video's SHA-256
digest. Older blended runs additionally contain `raw_cosmos_video.mp4` and
compositing metadata; those are retained historical evidence.
The run manifest records non-empty video bytes, variant count, actual GPU
parallelism, and the same conditioning contract.
`input/timeline.json` records original and prepared media measurements.
`labeled_augmented/captions.json` records separate caption coverage and the
evaluated video hash for each variant. Annotation verifies the current video
bytes and extracts fresh frames; finalization rejects stale or missing coverage.

`generate-variants` publishes this distributed stage contract to S3 only. Its
`output_uri` must use `s3://`; local paths are rejected before generation so the
workflow never implies that a local path shared across SkyPilot stages is
supported.

## Validate, plan, and render

From the repository root, activate the virtual environment created during
[installation](../../install.md). These examples check the spec and render plans
with placeholder inputs; they do not launch the workflow. For manual execution,
follow the [setup and run guide](../../../workflows/guides/paidf-cosmos3.md).
For twelve profiles from your MP4 or pinned public ALOHA cup-opening data, use
the [manual twelve-profile recipe](paidf-appearance-12.md#apply-the-recipe)
after setup. Both cover real input selection, planning, image checks,
submission, output inspection and cleanup.

```bash
SPEC=workflows/main/paidf-cosmos3.yaml
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" --run-id demo \
  --assume-decision promote_checkpoint --var bucket=example-bucket --json
npa workbench workflow plan-spec "$SPEC" --run-id demo \
  --assume-decision loop_back --var bucket=example-bucket --json
```

For execution, pass only secret names to the generic workflow submit surface:
`HF_TOKEN`, `NEBIUS_TOKEN_FACTORY_KEY`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY`. Use an available supported `H100:1` or RTX PRO 6000
accelerator through the normal workflow resource override; do not put cluster
names into the spec.
Submit with `--runtime` and omit `--assume-decision`. The YAML requires actual
runtime decisions and automatically enables the submitted NPA source overlay.

## Live validation scope

See the setup guide's [validation record](../../../workflows/guides/paidf-cosmos3.md#validation)
for the current implementation's measured checks and limitations, and its
[full-pipeline checks](../../../workflows/guides/paidf-cosmos3.md#check-every-stage-and-full-pipeline-completion)
to validate a new run. The recorded starter run completed all 15 stages,
including accepted captioning, real Cosmos Curator and FiftyOne curation, and
final recording/report verification. The guide retains the measured quality
diagnostics, curation fallback, and limits; execution success does not certify
training-data suitability.

A fresh October 24, 2026 starter run with the startup repairs completed all 15
stages on one RTX PRO 6000 Blackwell GPU. Both default variants passed on the
first generation attempt with an aggregate evaluator score of `0.570243`, using
the unchanged `0.2` quality and `0.25` attribute thresholds. Both received
captions and remained in the curated output. The independent starter audit
passed, including fresh pinned input bytes, full decoded alignment, native
controls and guardrails, artifact hashes, real component reports, and Rerun
identity and media checks. The run and its owned controller/API were closed
with verified cleanup receipts. This establishes an exploratory starter smoke;
it does not qualify manipulation-task fidelity.

The count-only twelve-variant override generated all twelve videos in both the
initial and shipped refinement passes, with zero generation failures. The
first evaluator accepted ten clips; the final evaluator accepted eleven with
an aggregate score of `0.393762`. The remaining clip missed all four requested
appearance attributes, so it failed the unchanged `0.25` attribute threshold
despite clearing the `0.2` aggregate threshold. Increasing the variant count
therefore proved generation coverage, without establishing an accepted
twelve-variant dataset.
The workflow completed the evidence stages and stopped at `reject-quality`.
An independent read-only rejection audit verified all twelve decoded videos,
native controls and guardrails, fresh source provenance, and the Rerun video's
hashes, timelines and rejected dispositions. Accepted-only captioning, curation
and finalization artifacts were absent. The accepted twelve-variant audit above
was not passed by this rejected run.

The publication regression can separately reuse retained real GPU output.
Run `npa/tests/e2e/test_paidf_cosmos3_publication_live.py` with
`NPA_INTEGRATION_E2E=1`, `NPA_PAIDF_RAW_EVIDENCE_DIR` pointing to downloaded
variant directories, `NPA_PAIDF_REPAIR_URI` set to a fresh S3 augment prefix,
and `NPA_PAIDF_REPAIR_DIR` set to a private local readback directory. It requires
saved raw-output hashes, uploads through the real publisher, and verifies exact
bytes after S3 readback. This proves publication fidelity only.

The B200 validation also ran against the pinned RoboPro physical capture. Its
initial and refined Cosmos passes retained 27,090,575-byte and 25,931,172-byte
raw model videos with upstream guardrails enabled and weights fetched at runtime.
The then-current publisher evaluated source/model composites rather than those
raw bytes: at a validation-only `0.40` threshold, the composites scored
`0.154068` and `0.256552`; the second pass cleared hallucination and appearance
checks but not temporal consistency or the required 4/4 attributes. The workflow
therefore failed closed after the bounded refinement, retaining 98 artifacts and
a 316,545-byte Rerun quality-evidence recording. This is historical pre-fix
rejection-path evidence, not validation of the current full-video structural
transfer path or an accepted-quality claim. It does not change the workflow's
current exploratory thresholds or the stricter values operators may select.

## Optional Cosmos 3 versus Transfer 2.5 comparison

This workflow makes no superiority claim. A reproducible comparison uses one
repository-owned synthetic MP4, the same sampled config manifest, identical
variant count and seeds, and the same Cosmos Evaluator threshold/check modes:

1. stage the fixture once under a private run prefix;
2. run `paidf-cosmos3.yaml` with one configured variant;
3. run `nvidia-paidf-vda-cosmos-transfer25.yaml` with `n_augmentations=1`, the same fixture,
   sampled appearance combination, and evaluator configuration;
4. retain each engine's unmodified `cosmos_augmented/manifest.json` and
   `grade/cosmos_evaluator.json`;
5. compare evaluator `score`, per-check dispositions, output bytes, and artifact
   completeness, reporting both results without ranking the engines.

Use fresh run IDs so neither engine overwrites the other. Record the fixture
SHA-256, workflow commit SHA, model/checkpoint, seed, guardrail posture, and
evaluator config. Keep exact private object and infrastructure identifiers in
access-controlled evidence, not documentation or PR text.

Hosted generation is an opt-in backend for the main workflow; see the
[endpoint configuration and recovery contract](../../../workflows/guides/paidf-cosmos3.md#optional-hosted-cosmos3-generation). Native cluster inference remains the default.
