# Recorded LeRobot subtask proof

[Watch the agent UI recording (MP4)](agent-ui-playback.mp4) ·
[Download the complete proof bundle](proof-bundle.zip) ·
[Rerun recording](lerobot-subtasks.rrd) ·
[Verification manifest](manifest.json) ·
[Every before/after row](before-after.json)

This is an executed **synthetic data-contract demonstration**, not captured
robot data or a screen recording of FiftyOne. Programmatic temporal tags pass
through the real production parser and label writer. The checked-in workflow
then runs locally and verifies the exported Parquet. No NVIDIA tool, cloud
job, model inference, or policy training was used.

This historical recording retains its [original recipe](../lerobot-subtask-recipes/README.md).
Current native-export validation is recorded [separately](../fiftyone-subtasks-native.md).

The **33-second MP4** is a real browser screen recording of this exact RRD
opened in the deployed agent UI's fullscreen embedded Rerun viewer. It steps
through all eight rows and pauses on `grasp` (visible around 9 seconds).
The synthetic-data disclaimer stays visible. Setup footage was trimmed and the
browser capture was transcoded to H.264; the frames were not recreated from
tables or screenshots. The [capture receipt](agent-ui-capture.json) binds the
MP4 to its source RRD hash and records full-video decoding checks. This MP4 is
a separate companion artifact, not part of the earlier proof ZIP.

## What the recording proves

```text
Original LeRobot-format input (no subtask_index)
  → subtask:* duration tags → labeled copy → YAML verification → Rerun recording
```

At **episode 0, frame 2, timestamp 0.2 seconds**, the derived row contains
**subtask_index = 2**, which resolves to **grasp** in `meta/subtasks.parquet`.
The original action `[0.2]`, frame identity, timestamp, and task index are
unchanged. All original input files retain their SHA-256 hashes.

| Dataset row | Episode | Frame | Time (s) | Original action | Label | Index |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.0 | `[0.0]` | approach | 1 |
| 1 | 0 | 1 | 0.1 | `[0.1]` | approach | 1 |
| 2 | 0 | 2 | 0.2 | `[0.2]` | grasp | 2 |
| 3 | 0 | 3 | 0.3 | `[0.3]` | grasp | 2 |
| 4 | 1 | 0 | 0.0 | `[1.0]` | align | 0 |
| 5 | 1 | 1 | 0.1 | `[1.1]` | place | 3 |
| 6 | 1 | 2 | 0.2 | `[1.2]` | place | 3 |
| 7 | 1 | 3 | 0.3 | `[1.3]` | place | 3 |

Coverage is **8/8 frames, 4 segments, 0 unlabeled frames**. Labels are test
inputs, not visually established ground truth. This demonstrates the label
handoff contract, not semantic accuracy or improved robot performance.

## Open the recording

From the repository root:

```bash
npa/.venv/bin/rerun docs/workbench/evidence/lerobot-subtasks/lerobot-subtasks.rrd
```

Scrub `dataset_row` to **2**. The original and derived rows appear side by side,
with the original action and categorical subtask-index series underneath.
The timeline is a dataset row ordinal, not elapsed capture or execution time;
the original per-episode timestamps are retained in each row.

The producer closes the recording, runs `rerun rrd verify` and `rrd print -vv`,
then independently decodes all eight samples from each of four entities and
compares them with the original and labeled Parquet rows. The manifest records
the result, recording hash, source hashes, recipe hashes, and limitations.

## Inspect or reproduce

The bundle includes `input/`, `reviewed/`, `temporal-tags.json`,
`export-report.json`, `subtask-proof.json`, `workflow-report.json`, the exact
workflow YAML, `before-after.json`, the recording, and `manifest.json`.
All manifest artifact paths are relative; there are no cloud credentials or
private storage locations. The workflow report records local execution even
though the reusable YAML also declares a Kubernetes resource profile.

```bash
PYTHONPATH=npa/src npa/.venv/bin/python npa/scripts/record_lerobot_subtask_proof.py \
  --output-dir results/lerobot-subtasks-new --run-id lerobot-subtasks-new

npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_lerobot_subtask_recording.py \
  npa/tests/workflows/test_lerobot_subtask_proof.py -q
```

Use a new output directory; the producer refuses to overwrite existing proof.
The recipe hashes identify the actual code used, including changes relative
to `checkout_base_revision`. Recording IDs and log timestamps differ on a new
run, so the RRD byte hash is not expected to stay constant. Regenerate the
committed evidence when changing its producer, exporter, verifier, or YAML;
the bundle test checks that its recipe hashes still match the checkout.

For real datasets and interactive labeling, follow the
[FiftyOne subtask-labeling guide](../../guides/lerobot-subtask-labeling.md).
