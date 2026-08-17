# PAIDF with Cosmos 3 video conditioning

`npa/workflows/workbench/npa-workflows/paidf-cosmos3.yaml` is an independent
Physical AI Data Factory composition. It does not replace or change
`physical-ai-data-factory.yaml`, whose augmentation engine remains Cosmos
Transfer 2.5.

The pipeline is:

1. select one generic MP4 or one camera from one LeRobot v2/v3 episode;
2. decode caption frames and understand the original with Token Factory;
3. run one real NVIDIA `cosmos-framework` `video2video` inference per variant;
4. grade every variant with real Cosmos Evaluator checks;
5. promote on a complete passing report, or retry with changed seed, guidance,
   and steps up to the configured bound;
6. caption accepted variants, then run real Cosmos Curator and FiftyOne Brain;
7. write a real Rerun recording and a fail-closed aggregate report.

Rejected runs skip labeling, Cosmos Curator, FiftyOne, and finalization. They
first write the available input, generated-video, evaluator, decision, and
quality-disposition evidence to `reports/sim2real.rrd`, then terminate with a
failure. Missing or incomplete evaluator reports also reject.

## Inputs and configuration

Choose `input_kind: video` and set `input_video_uri` to one MP4, or choose
`input_kind: lerobot` and set `lerobot_dataset_uri`, `input_episode`, and
`input_camera`. Dataset URIs may point to generic LeRobot v2.x or v3.x directory
trees. A full feature name such as `observation.images.front`, or an unambiguous
camera suffix such as `front`, is accepted. Shared v3 video files are trimmed
using the episode metadata timestamps; per-episode v2 video layouts are also
supported.

The committed `example-bucket` and run-scoped fixture path are placeholders.
They fail closed unless an operator stages input or the live harness seeds its
repository-owned synthetic MP4. No customer dataset, episode, camera, bucket, or
infrastructure identifier is embedded.

Generation behavior is configuration-driven through `cosmos3_checkpoint`,
`cosmos3_mode`, `seed`, `guidance`, `steps`, `variant_count`,
`variant_parallelism`, and `parallelism_preset`. Quality and retries use
`grade_threshold`, `refinement_iterations`, `retry_seed_stride`,
`retry_guidance_delta`, and `retry_steps_delta`. The composition requires
`video2video`: selecting a text-to-video or image-to-video mode fails before GPU
inference rather than producing a misleading source-conditioned claim.

Guardrails are enabled and enforced for this composition. The image contains the
pinned OpenMDW-1.1 framework source but no weights. The operator supplies
`HF_TOKEN` at runtime after accepting the checkpoint, Wan VAE, and
`nvidia/Cosmos-Guardrail1` terms. Tokens and model weights are never serialized
into artifacts or Git.

## Artifact contract

Every successful generation pass preserves the downstream layout:

```text
cosmos_augmented/
  manifest.json
  variant-0000/
    augmented_video.mp4
    frame-00001.png
    metadata.json
```

Each metadata file records the real engine (`nvidia-cosmos/cosmos-framework`),
`video2video` mode, source-video conditioning, checkpoint, seed, guidance,
steps, attempt number, guardrail posture, non-baked weights, and input lineage.
The run manifest records non-empty video bytes, variant count, actual GPU
parallelism, and the same conditioning contract.

`generate-variants` publishes this distributed stage contract to S3 only. Its
`output_uri` must use `s3://`; local paths are rejected before generation so the
workflow never implies that a local path shared across SkyPilot stages is
supported.

## Validate, plan, and render

```bash
SPEC=npa/workflows/workbench/npa-workflows/paidf-cosmos3.yaml
npa/.venv/bin/npa workbench workflow validate-spec "$SPEC" --json
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" --run-id demo \
  --assume-decision promote_checkpoint --var bucket=example-bucket --json
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" --run-id demo \
  --assume-decision loop_back --var bucket=example-bucket --json
npa/.venv/bin/npa workbench workflow submit "$SPEC" --run-id demo --runtime \
  --assume-decision promote_checkpoint --var bucket=example-bucket --plan-only
```

For execution, pass only secret names to the generic workflow submit surface:
`HF_TOKEN`, `NEBIUS_TOKEN_FACTORY_KEY`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY`. Use an available supported `H100:1` or RTX PRO 6000
accelerator through the normal workflow resource override; do not put cluster
names into the spec.

## Live validation scope

The complete synthetic workflow has succeeded on reserved RTX PRO 6000, and
real source-conditioned Cosmos 3 inference plus refinement semantics have
succeeded on reserved B200. The preserved reserved topology exposes one
requestable GPU per node on both paths. Because SkyPilot requires a two-GPU task
to fit on one node, `variant_count=2` with `variant_parallelism=2` remains
unit-tested rather than live-proven concurrently. No on-demand capacity was used;
a sequential two-variant run must not be described as concurrent evidence.

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
