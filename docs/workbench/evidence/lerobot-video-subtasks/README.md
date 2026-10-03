# Real LeRobot video with eight synchronized subtasks

[Step-by-step labeling, reproduction, and Mac viewing guide](../../guides/lerobot-subtask-labeling.md)

[Watch the labeled robot video (MP4)](lerobot-labeled.mp4) ·
[Watch its agent UI playback (MP4)](agent-ui-playback.mp4) ·
[Open the interactive recording (RRD)](lerobot-video-subtasks.rrd) ·
[Frame-level proof](subtask-proof.json) · [Manifest](manifest.json)

This is **real SO-100 robot footage**, not the earlier synthetic numeric fixture.
The 15.13-second, 30 FPS video shows the top and wrist cameras together, a changing
active-label badge, eight persistent phase labels, a proportional phase timeline,
and the actual episode frame/time. All 454 frames have exactly one label.

The annotations are **assistant-authored demonstration labels**, visually checked
against both cameras, not upstream ground truth or expert-certified labels.
They pass through the production FiftyOne temporal-tag parser and LeRobot writer.
This does not claim an interactive FiftyOne editing session, NVIDIA inference,
hierarchical policy training, or improved policy performance.

## What this proves

```text
Public LeRobot episode + two camera streams
  → subtask:* temporal intervals → labeled LeRobot copy
  → executable YAML verification → synchronized MP4 + inspectable RRD
```

| Subtask | Frames (start inclusive, end exclusive) | Time (seconds, approximate) |
| --- | --- | --- |
| ready | [0, 80) | 0.00–2.67 |
| approach | [80, 160) | 2.67–5.33 |
| grasp | [160, 190) | 5.33–6.33 |
| transfer | [190, 245) | 6.33–8.17 |
| place | [245, 279) | 8.17–9.30 |
| release | [279, 300) | 9.30–10.00 |
| retract | [300, 360) | 10.00–12.00 |
| complete | [360, 454) | 12.00–15.13 |

At episode **0**, frame **160**, timestamp **5.333333492279053**, the exported
`subtask_index = 2` resolves to **grasp**. All original frame columns, including
actions and observations, retain their values and Arrow types. The original
episode files retain their SHA-256 hashes after labeling. The local YAML run
reports **454 labeled frames, eight segments, zero unlabeled frames**.

The MP4 labels are read from the exported Parquet/catalog, not a separate video-only
label list. The RRD embeds that exact MP4 and records all 454 source rows, resolved
labels, categorical indexes, and video references on the `episode_time` timeline.
Tests independently decode the video and RRD and compare complete sequences,
embedded media bytes, artifact hashes, and unchanged input columns.

These recordings retain their original [archived recipe](../lerobot-subtask-recipes/README.md).
The newer [native FiftyOne validation](../fiftyone-subtasks-native.md) is separate
evidence, not a claim that the old videos were re-recorded. The companion
[native-plane receipt](source-camera-planes.json) verifies exact decoded YUV
pixels against the pinned full source. CI uses these visible-plane hashes because
RGB conversion rounding differs across CPU architectures; the original RGB
capture receipt remains unchanged.

The additional **18.76-second agent UI MP4** is an actual browser screen recording
of this RRD playing in the deployed agent's embedded Rerun viewer, at 1× speed.
Only fullscreen viewer playback is retained; setup and private infrastructure UI
are excluded. The capture was transcoded to H.264 (1680×1120, 25 FPS, 469 frames).
The [capture receipt](agent-ui-capture.json) binds its bytes to the RRD and records
the observed eight-phase sequence. It is a companion to the original manifest.

## Source and changes

Source: [LeRobot SO-100 pick/place dataset](https://huggingface.co/datasets/lerobot/svla_so100_pickplace),
revision `728583b5eaf9e739a7f119e2def466fa1d552402`, episode 0, published under
Apache-2.0. See the preserved [dataset card](SOURCE-README.md) and [license](LICENSE).
The card's embedded format example is stale; the actual pinned `meta/info.json`
and Parquet use LeRobot v3.0.

Changes: episode 0 was selected from the shared Parquet and camera files; original
AV1 camera packets were remuxed without re-encoding; metadata was scoped to this
episode; demonstration labels and annotation metadata were added to a copied
dataset. A separate H.264 composition adds camera titles and labeling overlays.
The full original snapshot is not committed. Its immutable revision and six source
file hashes are recorded in the manifest; both self-contained episode copies are
included in `input/` and `reviewed/`.

## Reproduce

Download the public pinned snapshot (approximately 470 MB; no gated model or GPU):

```bash
hf download lerobot/svla_so100_pickplace --repo-type dataset \
  --revision 728583b5eaf9e739a7f119e2def466fa1d552402 \
  --local-dir results/so100-source

PYTHONPATH=npa/src npa/.venv/bin/python npa/scripts/record_lerobot_video_subtasks.py \
  --source-dir results/so100-source --output-dir results/so100-labeled-new \
  --run-id so100-labeled-new

npa/.venv/bin/python -m pytest npa/tests/workflows/test_lerobot_video_subtasks.py -q
```

The producer requires FFmpeg, PyAV, Pillow, PyArrow, Rerun 0.31.4, and DejaVu Sans
or Arial. The output directory must not already exist. Source hashes are checked
before any output is written. New RRD recording IDs/timestamps change its bytes.
The recipe and artifact hashes in the manifest bind this recorded run, not every
future reproduction.

Open `lerobot-video-subtasks.rrd` in the agent UI's Rerun tab or with
`npa/.venv/bin/rerun`. Select `episode_time`, return to the beginning, and play.
The default view is the labeled two-camera video; the underlying `labeled/frame`,
`subtasks/index`, and `provenance` entities remain independently inspectable.
