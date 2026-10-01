# Video variant sweep

Turn source videos into a tracked synthetic dataset: describe the scene, merge an
appearance hint or retain a direct prompt, generate conditioned variants in
parallel, review fidelity, record Postgres/MLflow lineage, and publish accepted
clips. The operator kit now selects
[the Cosmos3 reference](../testing/video-variant-sweep-cosmos3.yaml) by default.
The [Transfer 2.5 reference](../testing/video-variant-sweep.yaml) remains available
for existing configurations and historical results.

This workflow ends at the synthetic dataset. Policy training, train/test splits,
simulator task-success evaluation, and deployment are separate workflows. The
review gate measures augmentation fidelity, not robot task success.

## Run through the operator kit

Use a Linux operator checkout with `npa[video-sweep]`, an existing Kubernetes
cluster, project-owned S3 storage, and reachable Postgres/MLflow services:

```bash
npa/.venv/bin/python -m npa.workflows.video_sweep.operator init \
  --config /private/video-sweep/sweep.json
# Fill in project, context, bucket, source URIs and exact accessible model IDs.
npa/.venv/bin/python -m npa.workflows.video_sweep.operator check \
  --config /private/video-sweep/sweep.json
npa/.venv/bin/python -m npa.workflows.video_sweep.operator run \
  --config /private/video-sweep/sweep.json \
  --output-dir /private/video-sweep/demo
```

`init` creates a private configuration with `generator: cosmos3-nano`, one B200
per worker, two appearance variants, eight frame samples, and threshold 0.8.
Set `accelerators` to a verified alternative such as `RTXPRO6000:1` when using
that GPU family. GPU placement and model access are checked before submission;
selecting a GPU does not establish its inference compatibility.

The current graph has two parallel workers. Each processes a disjoint partition
of **every source × variant** combination. This does not limit the number of
sources or variants. Changing worker concurrency still requires corresponding
worker states and output declarations. Parameter combinations are explicit rows;
parameter-axis expansion is not yet part of the operator kit.

`check` verifies operator prerequisites without submitting compute. `run` stages
immutable inventories and the current NPA source, then submits the standard
runtime with its own SkyPilot state directory. `resume` uses the same configured
run and durable runtime state. `export` needs only project storage and a completed
review/tracking result; it never submits compute. Choose a fresh export directory.
Use a new run ID when changing inputs, prompts, model selections or parameters.

The helper does not provision the cluster or tracking services. Both GPU workers
must fit concurrently. Service endpoints must be reachable from the worker pods;
operator-side credential checks do not establish that connectivity. macOS supports
authoring and export; isolated SkyPilot execution uses Linux.

## Native Cosmos3 generation

The native reference calls the real `run_cosmos3_generate` implementation with
`Cosmos3-Nano`, `video2video`, and explicit `TransferSettings`. It normalizes the
complete source to 832×480 at 24 fps with letterboxing, records both original and
prepared hashes, and supplies native edges from every source interval. It
verifies exact output frame count and timestamps against that prepared source.
This is full-source structural conditioning, not prefix-only continuation.

Native defaults use 93-frame windows, structural guidance 1.5, medium Canny
thresholds, no RGB hint, and zero first-chunk RGB conditional frames so appearance
can change. Later windows retain generated overlap for continuity. Guardrails
remain enabled and must report effective successful prompt/media evaluations.
Their configured state alone is insufficient. Temporal alignment verifies media
correspondence; it does not prove visual realism.

The v2 manifest declares the generator and its native controls explicitly:

```json
{
  "schema": "npa.video_sweep.variants.v2",
  "generator": "cosmos3-nano",
  "variants": [
    {
      "hint": "Dim the warehouse ambient lighting; preserve the vehicle and load",
      "seed": 17,
      "control_guidance": 1.5,
      "edge_threshold": "medium",
      "guidance": 5.0,
      "num_steps": 35
    },
    {
      "prompt": "A yellow warehouse forklift carries the same stable low pallet under warm overhead lighting. Preserve the exact vehicle, load, trajectory, floor contact, camera and timing.",
      "seed": 17,
      "control_guidance": 3.0,
      "edge_threshold": "medium",
      "guidance": 5.0,
      "num_steps": 35
    }
  ]
}
```

Each row has exactly one `hint` or `prompt`. A direct `prompt` is preserved
verbatim and its provenance records `direct-user-prompt`; an appearance `hint`
is merged with the source description. Source description and paired review
still use the selected reasoner. Supported edge thresholds are `very_low`, `low`,
`medium`, `high`, and `very_high`. `control_guidance` is native Cosmos3 structural
guidance, not Transfer 2.5's control weight. Seeds are nonnegative integers;
sampling steps and text guidance must be positive. No unchecked parameter bag
is forwarded into inference.

The optional `cfg_normalization` field accepts `enabled` or `disabled` (the
compatibility default). It forwards the native sampler's guided-prediction
normalization; it does not blend source pixels into the output. Strong structural
guidance can produce harsh outlines and distorted details. Compare gentler
sampling and normalization against the same source, retaining the original
quality threshold and rejected results. These controls do not guarantee realism.

The optional `first_chunk_conditional_frames` is 0 (default) or 1. One retains
the first source RGB frame as an appearance anchor, while complete source edges
still condition the whole clip. This can also constrain the first frame's
lighting and material edits. Use a source frame whose appearance is appropriate
for the requested variation and inspect transitions; anchoring is not proof of
photorealism or stable contacts.

Source inventories retain their existing schema:

```json
{"schema":"npa.video_sweep.sources.v1","clips":["s3://example-bucket/inputs/source.mp4"]}
```

For operator use, place only the variant rows in the configuration's `variants`
field; the kit writes the versioned manifest. Duplicate source URIs, duplicate
source bytes, duplicate variants, and invalid controls fail before generation.

## Reasoning and generation are separate models

`generator` selects video generation. `reasoner_model` selects source description
and paired visual review; `merge_model` selects prompt enhancement. The current
reasoning client uses Token Factory and requires exact available model IDs. A
Cosmos3 generation result does not prove Cosmos3 reasoning.

The reference reasoner is `nvidia/Cosmos3-Super-Reasoner`, which was unavailable
to the account used for the earlier live tests. Those tests explicitly selected
`MiniMaxAI/MiniMax-M3`. The reference merge model is
`nvidia/Nemotron-3_5-Lightning`. There is no silent substitution. Cluster-backed
Cosmos3 reasoning remains a separate integration requirement.

Source frames and hints are sent to the explicitly selected hosted service.
Use only inputs approved for that service. Public demonstrations must use
synthetic or otherwise approved imagery.

## Realism review

Before accepting a warehouse vehicle variant, inspect these independent aspects:

- Vehicle, mast, fork and pallet geometry remain rigid and consistent.
- Tires contact the floor; wheel rotation and translation are physically plausible.
- The load stays supported and moves with the vehicle without slipping or detaching.
- Source trajectory, occlusions, camera, timing and stationary background survive.
- The requested lighting or material change is visible, with plausible shadows.
- Chunk boundaries have no teleportation, abrupt shape changes or duplicate objects.

The reviewer receives explicit instructions about contact, rigid geometry, stable
loads and temporal continuity. Acceptance requires both a boolean pass and a
finite score meeting the configured threshold. Its evenly spaced sampled frames
include clip endpoints, but do not certify every frame. Inspect moving playback
and transition points as well as the score. Retain rejected candidates as evidence.

## Recovery, controls and export

Each completed candidate gets an immutable receipt before the worker proceeds.
A retry verifies the same plan/request identity, generator, seed, media hash,
full decoding and sampled metadata before reusing that candidate. Native receipts
also bind the generation evidence, prepared reference and actual edge-control
bytes. Corrupt or mismatched checkpoints fail closed instead of triggering a new
generation. A failed partition can reuse its completed candidates; an unfinished
candidate without a completed receipt is not claimed as recovered.

Native generation publishes the actual conditioning map and normalization /
temporal-alignment evidence. The offline exporter verifies those bindings and
shows the prepared source, generated video, real edge-control clip, model identity
and allowlisted numeric/categorical generation parameters. It excludes raw
prompts, review reasons, source URIs, run IDs, service addresses and provider IDs.
It removes audio and container metadata. Visible private imagery still requires
private handling.

The output directory contains a standalone `index.html`, silent `demo.mp4`,
`summary.json`, preview MP4s and posters. The HTML embeds all media and needs no
external requests. Playback and scrubbing include the native control video.
Threshold exploration does not alter recorded decisions or publication.

When all candidates are rejected, publication fails and no dataset or next-run
inventory is created. A completed, tracked rejection may still export a review
bundle. `run` and `resume` preserve the nonzero workflow result. Incomplete
tracking or failed publication of accepted clips cannot use that path.

## Credentials and artifacts

Keep credentials and operational configuration outside Git:

| Variable | Use |
| --- | --- |
| `HF_TOKEN` | Exact runtime-fetched model and guardrail access |
| `NEBIUS_TOKEN_FACTORY_KEY` | Selected hosted reasoning and merge models |
| `NPA_LINEAGE_POSTGRES_DSN` | Postgres role able to create/write lineage records |
| `MLFLOW_TRACKING_URI` | HTTPS tracking endpoint; loopback HTTP for local tests |
| `MLFLOW_EXPERIMENT_ID` | Existing tracking experiment |
| `MLFLOW_TRACKING_TOKEN` | Optional tracking bearer token |
| `MLFLOW_TRACKING_CA_PEM` | Optional private CA; TLS hostname verification remains enabled |

The kit resolves exact-project S3 credentials and forwards required values through
runtime secrets. An optional AWS session token is forwarded when present.
Artifacts live beneath the configured bucket, prefix and run ID. They include
`plan.json`, per-candidate `receipt.json`, native `generation.json` and controls,
worker receipts, `review.json`, `lineage.json`, and accepted-only
`dataset/manifest.json` plus `dataset/next-sources.json`.

Track every acceptance and rejection before publication. Postgres writes are
parameterized and replay-safe; MLflow records the actual per-candidate metrics.
The next-source inventory is for an explicit subsequent run, not a training loop.
Cancel the exact run and wait for terminal jobs before removing task-owned
controllers/services. Preserve dataset and lineage artifacts.

## Native runtime evidence

A synthetic warehouse forklift drive exercised the standard Cosmos3 graph on
two RTX PRO 6000 workers. Both outputs retained the complete 93-frame, 24 fps
timeline with zero timestamp error. Native receipts verified the actual edge
controls and effective successful text/video guardrails. MiniMax-M3 reviewed
twelve paired samples per clip; this did not exercise Cosmos3 reasoning.

The first warm/cool pair used structural guidance 2.5/3.0, text guidance 5.0,
35 steps and disabled CFG normalization. Strong outlines and deformed vehicle/load
details failed visual inspection. The unchanged 0.80 gate rejected both, scoring
0.25 and 0.15. Postgres retained two rows and MLflow retained two finished runs
with acceptance and review-score metrics. Publication correctly failed; no
dataset or next-source inventory was created, and the operator exported a tracked
rejection while preserving its nonzero result.

A second pair retained the same source bytes, direct prompts and seeds, using
structural guidance 1.0, text guidance 3.0 and enabled CFG normalization. Harsh
outlines improved visibly, and native receipts verified normalization, effective
guardrails and the same exact timeline. The reviewer still rejected both at
0.35: the warm clip retained a rendered material appearance, and the cool clip
failed its sampled motion-fidelity assessment. Both were tracked and withheld
from publication. Better-looking footage did not override the gate.

A third pair used a synthetic photographic edit of the first source frame as
the native one-frame appearance anchor, followed by the procedural motion
reference. Structural guidance remained 1.0 with CFG normalization enabled;
text guidance was 5.0. New direct prompts requested warm daylight and a gradual
cool LED transition. This changed both source and prompts, so it is not a
matched-input comparison. Native receipts verified the one-frame anchor,
effective guardrails, complete controls and the same exact output timeline.
The warm clip gained floor and material detail, while the cool clip developed
an excessive blue cast. Sampled review rejected both at 0.15 for geometry or
motion fidelity. An appearance anchor alone did not qualify physical realism.

A CPU replay verified both real candidate checkpoints with the inference
function set to raise if called. It reused both outputs without inference and
left both worker receipts unchanged. The exported native HTML passed synchronized
source/candidate/control playback, scrubbing, variant selection, unchanged gate
decisions, mobile overflow checks and zero external requests.

This validates the executed component path and strict rejection behavior.
It does not qualify realistic training data, B200 placement, multi-window
continuity or the default hosted Cosmos3 reasoner.

## Transfer 2.5 compatibility and evidence

Existing configurations without `generator`, or with `cosmos-transfer2.5`, use
the original reference and v1 variant rows: `hint`, `seed`, `control` (`edge` or
`vis`), `control_weight` in (0,1], and positive `guidance`. Native control names
are not silently translated to these older parameters. Historical runs remain
exportable with their actual generator label.

The earlier Transfer component run accepted one clip at 0.85 and rejected one at
0.30. A later full submission completed preparation, both GPU workers, review,
and Postgres/MLflow tracking; both candidates scored 0.30 against threshold 0.80
and publication correctly failed. These are procedural-source results with
MiniMax-M3 reasoning, not proof of a realistic Cosmos3 workflow. See the separate
readiness records beside each reference for the current scope of verification.
