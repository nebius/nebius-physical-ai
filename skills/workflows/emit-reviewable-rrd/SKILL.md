---
name: emit-reviewable-rrd
description: Use when authoring, reviewing, or operating an NPA workflow whose real run outputs should become a reviewable Rerun .rrd recording with factual timelines, provenance, declared artifacts, and independent content validation.
---

# Emit reviewable RRD artifacts

Build the recording from the run being reviewed. A stock example, generated
placeholder, screenshot, or renamed JSON file is never evidence for the run.

## Procedure

1. Identify the actual stage outputs that contain the facts to visualize. Keep
   raw customer inputs private; extract only the metrics, events, frames, or
   trajectories needed for review.
2. Choose timelines that preserve source semantics. Use `optimizer_step` for
   training progress, real capture time for timestamped sensors, and an
   explicitly labelled dataset/frame index when capture time does not exist.
3. Choose stable entity paths before writing. For training, prefer grouped
   entities such as `metrics/loss`, `metrics/learning_rate`,
   `throughput/global_samples_per_second`, `health/gradient_norm`,
   `checkpoint/materialized`, and `provenance/run`.
4. Set the Rerun recording id to the workflow run id. Record sanitized static
   provenance: producer, source revision, recipe/config identity, source
   artifact hashes, and factual limitations. Never embed credentials, signed
   URLs, customer payloads, hostnames, pod/node ids, or private infrastructure
   identifiers.
5. Write the recording with `rerun-sdk` and close its sink before inspection.
   Reuse an existing NPA Rerun converter or inspection helper when it matches
   the source; extend the producing workbench integration when it does not.
6. Put the file at a run-scoped private URI such as
   `s3://<bucket>/<workflow>/<run.id>/reports/<name>.rrd`. Declare that exact URI
   in the producing state's `outputs` with schema
   `application/vnd.rerun.rrd`. Keep inputs and all companion artifacts under
   the same run prefix so `npa workbench workflow artifacts` and artifact-first
   discovery can find them.
7. Fail the artifact stage if the required recording cannot be created,
   uploaded, or validated. Do not turn a mandatory RRD into a warning-only
   side effect.

## Make the content reviewable

- For optimization, log factual loss, the exact applied learning-rate schedule,
  interval timing/throughput, finite gradient or update-health diagnostics,
  checkpoint events, and aggregate distributed/device health on the
  `optimizer_step` timeline.
- Include before/after or held-out policy trajectories only when this run
  actually produced both sides with a valid alignment. Otherwise state the
  limitation in provenance and omit those entities.
- Use a blueprint when it materially improves the first view, but keep the
  underlying entities independently inspectable.
- Prefer a durable, deduplicated metric journal during long jobs and convert it
  deterministically after success. This makes resume factual without relying on
  unsupported append/recovery behavior for a partial RRD.

## Long-running staged recordings

Do not make a reviewer wait for a long workflow's terminal stage when factual
intermediate review points exist. Define a sparse milestone plan before launch:

- emit a preparation recording only after input verification and preprocessing
  succeed, using a preparation-specific timeline and actual coverage/progress;
- emit a separate qualification recording after a real bounded optimizer gate;
- during long training, emit an early journal-only snapshot and periodic
  checkpoint-aligned snapshots, including the final checkpoint;
- rebuild each snapshot deterministically from the durable journal prefix that
  existed at that milestone, then close, inspect, upload, and read it back as a
  standalone RRD; never append to or recover a partially written RRD;
- give every RRD and companion manifest a stage/milestone-specific immutable
  URI. Do not overwrite `latest.rrd` or use qualification facts as full-run
  progress;
- put the RRD byte hash, source-journal-prefix hash, decoded coverage, run id,
  stage, milestone, and checkpoint status in a write-once content-hashed
  manifest. Declare every known URI directly in the workflow outputs; and
- fail closed at a mandatory milestone while retaining its journal and any
  materialized checkpoint so an identical resume can rebuild missing artifacts.

Checkpoint callbacks may be asynchronous. A checkpoint-aligned snapshot is
eligible only after the checkpoint manager has finished and the expected
checkpoint directory is non-empty. Persist a run-scoped, atomic completion
marker only after that wait succeeds; resume must not infer completion from a
partially populated checkpoint directory. An explicitly labelled early
log-only snapshot is eligible after the corresponding metric record is flushed
and fsynced; its manifest must say that no checkpoint was claimed. If the
trainer numbers updates from zero, keep that factual timeline and state the
mapping between a human-facing completed-update milestone and its final source
step instead of inventing a future timeline row.

An operator-requested pause is different from an early log-only review point.
At the exact completed-update boundary, wait for the trainer's checkpoint
manager, require the optimizer-state checkpoint and its atomic completion
marker, and only then close the factual journal prefix and build the immutable
RRD. Record the zero-based source-step mapping, the original recipe target, the
completed and outstanding update counts, `operator_requested_pause`, and the
next resume step. The paused report must prove checkpoint bytes and readback as
well as RRD bytes and decoded coverage. A later resume must retain the same run
and recording identity, restore optimizer state, and verify existing milestone
RRDs/manifests byte-for-byte instead of rewriting them. Do not call a pause
full-recipe convergence, final model acceptance, or a failed run.

## Validate before handoff

Independently validate the uploaded bytes:

- run `rerun rrd verify <file>`;
- inspect with `rerun rrd print -vv <file>`;
- require the expected application id, run recording id, timelines, and entity
  paths in decoded output;
- decode a metric entity and compare its complete observed step sequence with
  the source journal prefix, not only its last value;
- verify non-empty S3 bytes by read-after-write; and
- confirm artifact discovery lists the exact run-scoped `.rrd`.

An extension, a viewer opening, or a producer's own success message is not
enough. Preserve the inspection result and content hash in the run report.

## Creation versus sharing

Creating and privately storing the recording is the workflow contract. Sharing
is optional and separate. Use `npa rerun host` or `npa rerun share` only when
the operator asks for a time-boxed presigned link; follow
`skills/tools/artifact-viz-share/SKILL.md` and treat the link as a credential.

## Verify repository changes

Run the relevant workflow/output tests, then:

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python /home/ubuntu/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/workflows/emit-reviewable-rrd
```
