# Label LeRobot subtasks in FiftyOne

[Guides](README.md) · [LeRobot training](reachy2-lerobot-policy.md) · [FiftyOne skill](../../../skills/tools/fiftyone/SKILL.md)

Use this workflow when a policy needs an episode-level task for planning and a
frame-aligned subtask for low-level control. LeRobot remains the training data
contract; FiftyOne 1.22 supplies synchronized camera/state/action playback and
the human timeline editor. No NVIDIA model or service is required.

There are two distinct paths:

- [Reproduce the recorded SO-100 example](#reproduce-the-recorded-so-100-example):
  real robot footage with scripted demonstration annotations, local YAML
  verification, and a labeled MP4/RRD.
- [Label your own dataset](#before-labeling-your-own-dataset): human review in FiftyOne,
  followed by a derived LeRobot export and the same verification gate.

FiftyOne is the annotation/review interface, not the model deciding what a
subtask means. Workbench's exporter writes its temporal tags into LeRobot
Parquet using PyArrow. Rerun and PyAV/FFmpeg produce the recorded visualization.
An optional VLM can propose labels for review, but that automation is not part
of this example. Label export alone does not implement System 1/System 2 policy
training.

## Reproduce the recorded SO-100 example

The source is [lerobot/svla_so100_pickplace at the pinned revision](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/tree/728583b5eaf9e739a7f119e2def466fa1d552402),
published under Apache-2.0. Episode **0** shows a real SO-100 arm picking up a cube
and placing it in a box: **454 frames at 30 FPS, approximately 15.13 seconds**.
Inputs include top and wrist camera MP4s, joint states, actions, and timestamps.

The eight added labels are `ready`, `approach`, `grasp`, `transfer`, `place`,
`release`, `retract`, and `complete`. They are assistant-authored demonstration
annotations, visually checked against the cameras, **not upstream ground truth**.
The recording exercises the production parser/exporter programmatically; it
does not show someone labeling in the FiftyOne App.

### Watch the existing result

- [Labeled two-camera MP4](../evidence/lerobot-video-subtasks/lerobot-labeled.mp4)
- [Actual agent UI playback capture](../evidence/lerobot-video-subtasks/agent-ui-playback.mp4)
- [Interactive RRD](../evidence/lerobot-video-subtasks/lerobot-video-subtasks.rrd)
- [Frame boundaries, source license, manifests, and input/output datasets](../evidence/lerobot-video-subtasks/README.md)

From the repository root on a Mac:

```bash
open docs/workbench/evidence/lerobot-video-subtasks/lerobot-labeled.mp4
open docs/workbench/evidence/lerobot-video-subtasks/agent-ui-playback.mp4
npa/.venv/bin/rerun docs/workbench/evidence/lerobot-video-subtasks/lerobot-video-subtasks.rrd
```

For Rerun, select `episode_time`, move to the beginning, and play. The main view
shows both cameras, the current label, and all eight phases. Inspect
`labeled/frame` and `subtasks/index` to see the underlying exported frame values.

### Regenerate the labeled dataset, MP4, and RRD

Start from a checkout containing this guide and the producer script, with the
[repository virtualenv installed](../../../CONTRIBUTING.md#testing-requirements).
The local producer uses the NPA environment's PyAV, Pillow, PyArrow, and pinned
Rerun SDK. It also needs `ffmpeg` on `PATH`, DejaVu Sans or Arial, and the Hugging
Face CLI (`hf`) for the download below. On a Mac with Homebrew, install both
tools with `brew install ffmpeg hf`; Arial is supported by the renderer. Other
platforms can follow the [Hugging Face CLI installation guide](https://huggingface.co/docs/huggingface_hub/guides/cli#install-with-pip).
Run these commands from the repository root:

```bash
hf download lerobot/svla_so100_pickplace --repo-type dataset \
  --revision 728583b5eaf9e739a7f119e2def466fa1d552402 \
  --local-dir results/so100-source

PYTHONPATH=npa/src npa/.venv/bin/python npa/scripts/record_lerobot_video_subtasks.py \
  --source-dir results/so100-source \
  --output-dir results/so100-labeled-new \
  --run-id so100-labeled-new

npa/.venv/bin/rerun rrd verify results/so100-labeled-new/lerobot-video-subtasks.rrd
npa/.venv/bin/python -m json.tool results/so100-labeled-new/subtask-proof.json
```

The public snapshot is approximately 470 MB. Keep the pinned revision: the
producer verifies the original file hashes before writing anything. Use a new
output directory on every run; an existing destination is rejected.

Expected outputs under `results/so100-labeled-new/`:

| Artifact | What to check |
| --- | --- |
| `input/`, `reviewed/` | Original episode and separate labeled copy; original actions/states are unchanged. |
| `lerobot-labeled.mp4` | Two cameras with changing labels, 1280×720, 30 FPS, 454 frames. |
| `lerobot-video-subtasks.rrd` | The exact MP4 plus all 454 labeled rows and synchronized video references. |
| `subtask-proof.json` | `status: verified`, 454 labeled frames, eight segments, zero unlabeled frames; `grasp` at frame 160. |
| `manifest.json`, `workflow-report.json` | Source/artifact hashes and successful local YAML execution. |

The producer is deliberately fixed to this dataset, episode, and vocabulary.
It is **not a generic renderer for arbitrary labeled datasets**. For a different
dataset, follow the manual labeling/export path below; the current generic
conversion commands do not reproduce this example's custom subtask overlays.

### Record the RRD playing in the agent UI

The producer creates the labeled MP4 and RRD, but **does not create the separate
agent UI screen recording**. To make that recording yourself:

1. Open your authenticated agent UI, choose the **Rerun** tab, and drag the new
   `lerobot-video-subtasks.rrd` into its viewer. Confirm that the SO-100 cameras
   and eight phase labels appear, not a stock demo.
2. Expand the viewer, choose `episode_time`, pause, and seek to the beginning.
   Keep private endpoints, credentials, and infrastructure panels outside the
   captured area.
3. On macOS, press **Shift–Command–5**, select the viewer region, start recording,
   and click the viewer's **Play** control. Stop after the final `complete` phase.
4. Save the screen recording and convert it to a broadly playable MP4 if needed:

```bash
ffmpeg -n -i '<screen-recording.mov>' -an -c:v libx264 -crf 18 \
  -pix_fmt yuv420p -movflags +faststart results/so100-labeled-new/agent-ui-playback.mp4
```

Watch the saved file to confirm every label transition is visible. A screen
capture can repeat or drop displayed frames; use the direct MP4/RRD for exact
source-frame alignment. This records **playback**, not a FiftyOne editing session.

## Before labeling your own dataset

- Use a persistent, configured FiftyOne VM or container-on-VM workbench with
  **FiftyOne 1.22 or newer** and native `fo.types.LeRobotDataset` support. The
  load/export commands below use that same configured service and database;
  do not substitute a short-lived serverless job or an unrelated App instance.
- Verify the installed service version in its Python environment. Installing
  the new client code does not upgrade an older remote FiftyOne image. This PR's
  source candidate still needs the normal image acceptance/promotion gates.
- Keep the source LeRobot v3 metadata, Parquet, and referenced videos together in
  a readable S3 prefix. Configure source-read and destination-write credentials
  using the [Workbench setup guide](../getting-started.md), and choose a new
  derived-data prefix. Do not publish credentials or signed source URLs.
- Define a vocabulary and boundary rules first: for example, whether `grasp`
  starts when the fingers begin closing or only after contact. Use the same rule
  across episodes. Include genuine waiting/recovery phases so full coverage does
  not require labeling idle footage as motion.

## 1. Load the source dataset

Once those prerequisites are satisfied, load the LeRobot v3 dataset:

```bash
npa workbench fiftyone load-dataset \
  --name robot-subtasks \
  --input-path 's3://<bucket>/<source-dataset>/' \
  --format lerobot

npa workbench fiftyone open
```

The source is loaded as one multimodal sample per episode. Existing
`subtask_index` metadata is restored as timeline tags when present.
Keep `npa workbench fiftyone open` running while you use its authenticated local
tunnel. Use another terminal for export; Ctrl-C closes the tunnel, not the
remote dataset or workbench.

## 2. Review and label

Open `robot-subtasks` and choose an episode. Display both camera streams and
the state/action signals you need. Pause at a boundary, then hold Shift and drag
across the playback timeline to select an interval. Enter a temporal tag such as
`subtask:grasp`; a whole-episode sample tag is not a substitute. This uses
[FiftyOne's native temporal-tag interface](https://docs.voxel51.com/user_guide/multimodal.html#temporal-tags),
not the separate community Annotate-LeRobot plugin.

Repeat with this exact namespace, using your agreed vocabulary:

```text
subtask:approach
subtask:grasp
subtask:transfer
subtask:place
subtask:release
```

Intervals are half-open (`[start, end)`). They may touch, but must not overlap,
and every frame in every exported episode must be covered. Other temporal tags
are left untouched and are not training labels. The episode's existing LeRobot
task instruction remains the high-level task.
Replay every boundary from both cameras, then review the whole episode. The
export command processes the entire named dataset, not just the episode visible
in the App: finish coverage for every imported episode before exporting.

## 3. Export a derived LeRobot dataset

```bash
npa workbench fiftyone export-lerobot-subtasks \
  --dataset-name robot-subtasks \
  --output-path 's3://<bucket>/<derived-dataset>/' \
  --output-format json
```

The output prefix must be empty. The command exports a self-contained copy,
validates coverage, and writes:

- `subtask_index` on every frame row;
- `meta/subtasks.parquet` as the label catalog;
- `meta/lerobot_annotations.json` for resumable annotation state; and
- the `subtask_index` feature declaration in `meta/info.json`.

The original dataset is never rewritten. Train against the derived prefix only
after the report shows `unlabeled_frame_count: 0`.

## 4. Prove the reviewed labels with YAML

Use [`lerobot-subtask-proof.yaml`](../../../workflows/testing/lerobot-subtask-proof.yaml)
for the repeatable check after interactive review. It
reads the derived LeRobot Parquet files, requires every frame to resolve through
`meta/subtasks.parquet`, and publishes one row-level proof without rewriting the
dataset.

```bash
npa workbench workflow validate-spec \
  workflows/testing/lerobot-subtask-proof.yaml --json

npa workbench workflow plan-spec \
  workflows/testing/lerobot-subtask-proof.yaml \
  --run-id subtask-proof-preview \
  --var 'bucket=<configured-bucket>' \
  --var 'reviewed_dataset_uri=s3://<bucket>/<derived-dataset>/' \
  --var 'proof_uri=s3://<bucket>/<proof-prefix>/subtask-proof.json' \
  --json
```

Submit the same spec and overrides after the plan is correct:

```bash
npa workbench workflow submit workflows/testing/lerobot-subtask-proof.yaml \
  --run-id '<new-run-id>' \
  --var 'bucket=<configured-bucket>' \
  --var 'reviewed_dataset_uri=s3://<bucket>/<derived-dataset>/' \
  --var 'proof_uri=s3://<bucket>/<proof-prefix>/subtask-proof.json' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

| Setting | Meaning |
| --- | --- |
| `reviewed_dataset_uri` | Required derived LeRobot v3 dataset prefix, after export. |
| `expected_subtask` | Label to prove; defaults to `grasp`. Override for your vocabulary. |
| `proof_uri` | Exact output object, outside the dataset. Defaults to the run's `proof/subtask-proof.json`. |
| `source_overlay` | Defaults to `true` so the worker runs the submitted checkout. |

The proof includes
the episode, frame, timestamp, resolved label, integer `subtask_index`, source
Parquet and label-catalog SHA-256, row SHA-256, and dataset-wide coverage counts.
The worker downloads metadata and frame Parquet only; videos are not needed for
this check. Missing frames, unlabeled rows, unknown indexes, and an absent
requested label fail the workflow before it publishes a proof.

YAML runs **after** annotation; it neither opens the editor nor discovers subtask
boundaries. The SO-100 example above executes this same YAML locally. It is not
evidence of a completed cloud submission; check the
[readiness record](../../../workflows/testing/lerobot-subtask-proof.readiness.json)
before attempting the cloud route.

## Troubleshooting and completion checklist

| Symptom | Action |
| --- | --- |
| No multimodal episode viewer or native LeRobot type | Check the remote FiftyOne version and accepted deployment, not only the local CLI. |
| Export cannot find the dataset | Use the exact `--name` from import as `--dataset-name`, against the same persistent service/database. |
| Gaps or overlapping labels | Inspect adjacent `subtask:` boundaries and all episodes; the default export requires complete non-overlapping coverage. |
| Derived prefix is nonempty | Choose a new prefix. Do not overwrite the source or an earlier reviewed export. |
| YAML cannot find `grasp` | Pass `--var 'expected_subtask=<your-label>'` to planning and submission. |
| Reproduction rejects source hashes | Fetch the exact pinned full snapshot, not a different revision or only the trimmed episode clips. |
| Video has no changing labels | Open `lerobot-labeled.mp4` or the new RRD; a raw camera MP4 and the earlier numeric fixture are different artifacts. |

Before training, require zero unlabeled frames, inspect representative boundaries,
and retain the source/derived dataset identities and verification report. Have
the training code explicitly consume the labels; validation proves the data
contract, not semantic correctness or improved policy performance.

## Reproducible test evidence

The [native FiftyOne validation report](../evidence/fiftyone-subtasks-native.md)
tests the actual 1.22 importer/exporter and MongoDB, not a mocked dataset.
It imports real SO100 episodes in order `7, 2`, exports them deterministically as
`2 → 0, 7 → 1`, checks all 771 frame identities and four labels, and re-imports
the result. The export report includes `episode_index_mapping`; source episode
indexes are not the same as the contiguous indexes in the derived dataset.

When LeRobot stores task text as a Pandas index, `load-dataset` makes a separate
staging copy with an additional `task` column for FiftyOne 1.22 compatibility.
Allow disk space for that copy; the original files remain unchanged. Re-import
datasets loaded by older Workbench versions before exporting if FiftyOne reports
`LeRobot tasks metadata must contain task_index and task`.

To run the opt-in native test, use a dedicated development environment containing
`fiftyone==1.22.0`, FFmpeg, and the pinned full SO100 snapshot downloaded above:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_FIFTYONE_LEROBOT_SOURCE="$PWD/results/lerobot-video-proof/source" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_fiftyone_subtasks_native.py -q
```

`NPA_E2E_FIFTYONE_LEROBOT_SOURCE` has no default and must point to that local
snapshot. The test creates temporary MongoDB data and deletes its own FiftyOne
datasets. Its tags are programmatic test labels, not a recorded human review.
S3 upload uses the shared `StorageClient` with adaptive retries and a nonempty
prefix check. Use a unique destination per export: that check is not a lock
against simultaneous writers.

Start with the [real labeled LeRobot MP4](../evidence/lerobot-video-subtasks/lerobot-labeled.mp4):
two synchronized SO-100 cameras, eight changing subtask labels, and a phase
timeline. Its [evidence and reproduction guide](../evidence/lerobot-video-subtasks/README.md)
includes the actual 454-frame input/derived episode, executable YAML report,
and an RRD containing the video and every labeled row. These are visually checked
demonstration annotations, not labels supplied by the original dataset.

Open the [recorded proof bundle](../evidence/lerobot-subtasks/README.md) for the
actual input and derived Parquet files, a scrub-able Rerun recording, every
before/after row, the executed YAML report, and hash-bound verification. The
recording uses synthetic numeric data and programmatic tags; it does not claim
to show a robot capture or FiftyOne UI interaction. The linked instructions
regenerate the whole bundle into a new local directory.

The local test executes the YAML through the workflow interpreter against a
synthetic LeRobot Parquet fixture after applying the actual subtask exporter:

| Episode | Frame interval | Subtask | `subtask_index` |
| --- | --- | --- | --- |
| 0 | `[0, 2)` | approach | 1 |
| 0 | `[2, 4)` | grasp | 2 |
| 1 | `[0, 1)` | align | 0 |
| 1 | `[1, 4)` | place | 3 |

It proves `grasp` at episode `0`, frame `2`, timestamp `0.2`, with `8/8` frames
labeled across four segments. Tests independently recompute the evidence hashes
and confirm that verification preserves the source bytes. This validates the
data contract; the fixture is synthetic and does not establish real-world label
accuracy or policy improvement.

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_lerobot_subtask_proof.py -q
```

The CPU live-submit matrix seeds a separate four-frame fixture and reads back
the published S3 proof, data Parquet, and catalog to verify the actual row and
digests. Current cloud prerequisites and execution status are recorded in the
[readiness record](../../../workflows/testing/lerobot-subtask-proof.readiness.json).

## Scope

This path is manual and CPU-only. A VLM can propose labels upstream, but it is
optional and proposals still need review before export. No NVIDIA labeling tool
is required. The source candidate is not a newly published FiftyOne image until
its exact image bytes pass the normal build, security, functional, and promotion
gates.

Training must explicitly consume `subtask_index` or its decoded instruction in
the policy or sampler. Exporting labels alone does not make a standard LeRobot
policy use hierarchical conditioning.
