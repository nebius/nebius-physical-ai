# PAIDF with Cosmos 3 video conditioning

For a complete manual setup and run, start with the
[PAIDF Cosmos 3 setup and run guide](../../../workflows/guides/paidf-cosmos3.md). It covers
project setup, new or existing Kubernetes clusters, credential-file formats,
Token Factory key creation, Python and CLI installation checks, submission, and
artifact inspection.
Use this page for the workflow's input, generation, and acceptance contracts.
For the configuration keys and timing contract, see the setup guide's
[generation and evaluation settings](../../../workflows/guides/paidf-cosmos3.md#r5-find-and-change-generation-and-evaluation-settings).
For rejected runs, see its
[quality-rejection diagnostics](../../../workflows/guides/paidf-cosmos3.md#r6-diagnose-quality-rejection-and-prepare-the-next-run).

`workflows/main/paidf-cosmos3.yaml` is an independent
Physical AI Data Factory composition. It does not replace or change
`physical-ai-data-factory.yaml`, whose augmentation engine remains Cosmos
Transfer 2.5.

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

## Inputs and configuration

Choose `input_kind: video` and set `input_video_uri` to one MP4, or choose
`input_kind: lerobot` and set `lerobot_dataset_uri`, `input_episode`, and
`input_camera`. Dataset URIs may point to generic LeRobot v2.x or v3.x directory
trees. A full feature name such as `observation.images.front`, or an unambiguous
camera suffix such as `front`, is accepted. Shared v3 video files are trimmed
using the episode metadata timestamps; per-episode v2 video layouts are also
supported.

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
`variant_parallelism`, and `parallelism_preset`. `augmentation_seed` defaults to
`30`, keeping appearance profiles consistent across fresh run IDs; change it for
new appearance experiments. Quality and retries use
`grade_threshold`, `attribute_threshold`, `refinement_iterations`, `retry_seed_stride`,
`retry_guidance_delta`, and `retry_steps_delta`. `source_motion_weight` is a
compatibility setting that must be `0.0` (the default). Nonzero values fail
before generation. Publication copies the model output bytes without blending,
resizing, or changing the frame rate, and records their SHA-256 digest.
The check lives in the shared generation implementation, so custom workflows
using `workbench.cosmos3.generate_variants` and direct CLI callers receive the
same protection for any dataset. Existing deployed images need the updated code;
setting the workflow value to zero also disables blending in the older publisher.
The composition requires
`video2video`: selecting a text-to-video or image-to-video mode fails before GPU
inference rather than producing a misleading source-conditioned claim.

The canonical workflow enables `structural_control: edge`. `conditioning_fps`
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
with placeholder inputs; they do not launch the workflow. For execution, use the
[coding-agent workflow prompt](../agent-first-run.md#run-paidf-with-cosmos-3),
which covers real inputs, resource planning, image checks, submission, and output
inspection.

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

The publication regression can separately reuse retained real GPU output.
Run `npa/tests/e2e/test_paidf_cosmos3_publication_live.py` with
`NPA_INTEGRATION_E2E=1`, `NPA_PAIDF_RAW_EVIDENCE_DIR` pointing to downloaded
variant directories, `NPA_PAIDF_REPAIR_URI` set to a fresh S3 augment prefix,
and `NPA_PAIDF_REPAIR_DIR` set to a private local readback directory. It requires
saved raw-output hashes, uploads through the real publisher, and verifies exact
bytes after S3 readback. This proves publication fidelity only.

## Optional Cosmos 3 versus Transfer 2.5 comparison

This workflow makes no superiority claim. A reproducible comparison uses one
repository-owned synthetic MP4, the same sampled config manifest, identical
variant count and seeds, and the same Cosmos Evaluator threshold/check modes:

1. stage the fixture once under a private run prefix;
2. run `paidf-cosmos3.yaml` with one configured variant;
3. run `physical-ai-data-factory.yaml` with `n_augmentations=1`, the same fixture,
   sampled appearance combination, and evaluator configuration;
4. retain each engine's unmodified `cosmos_augmented/manifest.json` and
   `grade/cosmos_evaluator.json`;
5. compare evaluator `score`, per-check dispositions, output bytes, and artifact
   completeness, reporting both results without ranking the engines.

Use fresh run IDs so neither engine overwrites the other. Record the fixture
SHA-256, workflow commit SHA, model/checkpoint, seed, guardrail posture, and
evaluator config. Keep exact private object and infrastructure identifiers in
access-controlled evidence, not documentation or PR text.
