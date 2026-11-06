<!-- register: partner runbook | reader: Encord operators | consumed: task-time reference -->
# Encord media and labeling workflows

These workflows use the real Encord SDK and S3. They run on a local CPU or in
the standard NPA workflow runtime; they do not deploy an Encord service.

| Workflow | Result |
| --- | --- |
| [Labeling demo](encord-labeling-demo.yaml) | Upload videos → create ontology/project → save box/polygon tracks and scene labels → export labels/media → verify → render annotated MP4s. |
| [Roundtrip demo](encord-roundtrip-smoke.yaml) | Push media, filter it into a Collection, pull the selection, and verify item identities and bytes. |
| [Push](encord-push.yaml) | Register or explicitly upload media and write a durable receipt. |
| [Pull](encord-pull.yaml) | Materialize an existing dataset, collection, or project into S3. |

Register mode requires source read permission and can read the full input
prefix for SHA-256 hashing. See the [transfer and manifest migration notes](../../../docs/workbench/encord.md).
New pull artifacts use `pull_manifest.v2`; readers retain v1 support.
The roundtrip demo uses the intrinsic Encord width filter by default. Set
`encord_curate_filters` to a comma-separated filter list for a different rule.
For brightness, sharpness, or file-size filters, first compute quality metrics
for the folder in Encord.
The opt-in live test has completed push, curate, pull, and verification against
Encord and S3. Its run artifacts are retained privately. Kubernetes pod
execution remains untested.

Imported prelabels are real object annotations that still need human review.

## Prepare inputs

Install `npa[encord]` and configure S3 credentials plus `ENCORD_SSH_KEY_FILE`
locally, or `ENCORD_SSH_KEY_B64` for forwarding to workflow pods. Check access
with `npa workbench health preflight --checks encord,s3`.
Select the input prefix, fresh output prefix, and Encord targets before execution.
The labeling workflow defaults to **upload**, creating an Encord-managed media
copy, a dataset, an ontology, and a new project. It explicitly initializes label
rows. Existing projects are not modified.

Stage MP4 videos in S3 and a separate JSON label plan. Each plan video must
match an exact source URI and SHA-256 in the push receipt. The plan covers all
pushed videos; filename matches are insufficient. Prelabels can come from a
model or your own annotation process. The workflow imports supplied prelabels;
it does not invoke an Encord model or invent annotations.

The plan has this shape (replace placeholders with actual values):

```json
{
  "schema_version": "npa.encord.label_plan.v1",
  "provenance": "Describe how these unreviewed prelabels were produced",
  "videos": [{
    "source_uri": "s3://<bucket>/input/robot.mp4",
    "source_sha256": "<64 lowercase hexadecimal characters>",
    "width": 640,
    "height": 480,
    "frame_count": 169,
    "tracks": [{
      "track_id": "bottle-1",
      "class_name": "green bottle",
      "boxes": [{"frame": 0, "x": 0.75, "y": 0.28, "width": 0.12, "height": 0.28}]
    }]
  }]
}
```

Frames are **zero-based**. Coordinates are fractions of image dimensions and
boxes must remain inside the image. Track IDs must be unique within a video,
and frames unique within a track. Supply every desired frame explicitly;
there is no implicit interpolation. Geometry and frame count must match
Encord's decoded media. Each distinct `class_name` becomes an ontology object. There is no one-object
limit: use distinct `track_id` values for multiple instances of the same class,
and add videos to the `videos` list.

### Polygons and temporal scene labels

Use `npa.encord.label_plan.v2` for mixed shapes and classifications. Existing v1
box plans remain supported. A box track may omit `shape` (default
`bounding_box`). A polygon track requires `shape: polygon` and `polygons` in
place of `boxes`:

```json
{
  "track_id": "basket-1",
  "class_name": "basket",
  "shape": "polygon",
  "polygons": [{
    "frame": 0,
    "points": [{"x": 0.4, "y": 0.1}, {"x": 0.7, "y": 0.1}, {"x": 0.6, "y": 0.35}, {"x": 0.45, "y": 0.3}]
  }]
}
```

Vertices are normalized, ordered, distinct, and implicitly closed. Polygons
must have nonzero area and no crossing edges; holes and multiple rings are not
supported. A class uses the same shape across all videos. A polygon follows
its own persistent object identity just like a box.

Add optional `classifications` beside each video's `tracks`. Each name becomes
a radio classification in the shared ontology; values become its options:

```json
"classifications": [{
  "name": "bottle visibility",
  "segments": [
    {"start_frame": 0, "end_frame": 96, "value": "detected"},
    {"start_frame": 97, "end_frame": 168, "value": "not detected"}
  ]
}]
```

Intervals include both endpoints and must fit the video. Segments of the same
classification cannot overlap; gaps remain unclassified. Names must be unique
within each video and distinct from object class names. Multiple classifications
can coexist, such as object visibility and lighting. A value changing over time
creates separate Encord classification instances with exact frame ranges.

## Run and inspect

```bash
npa workbench workflow validate-spec workflows/partners/encord/encord-labeling-demo.yaml
npa workbench workflow submit workflows/partners/encord/encord-labeling-demo.yaml \
  --run-id <fresh-run-id> \
  --var bucket=<bucket> \
  --var encord_media_uri=s3://<bucket>/input/media/ \
  --var encord_label_plan_uri=s3://<bucket>/input/label-plan.json \
  --secret-env ENCORD_SSH_KEY_B64
```

Add `--plan-only` to inspect the resolved commands before execution. The `prefix`
default is `encord-labeling/{{run.id}}`; `encord_folder`, `encord_dataset`, and
`encord_project` are also run-scoped. Override targets with `--var` as needed.
`encord_integration` is empty for upload; register requires a cloud integration.

For local execution against real services, create a private JSON config outside
the repository with `bucket`, `prefix`, `encord_media_uri`, `encord_integration`,
and `encord_label_plan_uri`. Keep `{{run.id}}` in `prefix` for fresh outputs:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_ENCORD_LABEL_CONFIG=/private/encord-label-config.json \
NPA_E2E_ENCORD_EVIDENCE_DIR=/private/encord-evidence \
npa/.venv/bin/python -m pytest npa/tests/e2e/test_encord_labeling_live.py -q
```

Without `NPA_E2E_ENCORD_LABEL_CONFIG`, the test skips without contacting Encord.
`NPA_E2E_ENCORD_EVIDENCE_DIR` defaults to pytest's temporary directory. It retains
private execution evidence and downloads annotated MP4s, checks hashes, and
decodes every frame. Created Encord resources and S3 artifacts remain available
for inspection; deletion is a separate operator action.

Open the project named by `encord_project` in Encord Annotate, open its video
task, and scrub the timeline to inspect the persistent object tracks. They are
programmatic prelabels (`manual_annotation=False`), with human review pending.
A label-editor screen recording requires a signed-in browser; the workflow uses
SDK access independently of the browser session.

## What the output proves

The graph is `push → import_labels → pull → verify → render_labels`.

| Relative artifact | Evidence |
| --- | --- |
| `push/push_receipt.json` | Source URI, source SHA-256, dataset and Encord item identities. |
| `labels/receipt.json` | Plan digest, provenance, project/ontology identities, label rows and object hashes. |
| `pull/manifest.json`, `pull/labels/*.json` | Media and actual labels freshly exported from the new project. |
| `verify/roundtrip_report.json` | Exact media identity and byte integrity. |
| `demo/demo.json`, `demo/annotated-*.mp4` | Verified exported-label counts, decoded frame counts, and output hashes. |

The renderer compares exported row/object identities, classes, frames, and
coordinates with the import plan, including polygon vertices and classification
identities, options, and inclusive frame ranges. Missing, extra, or changed
annotations fail. Overlays display stable track IDs and colors, polygon outlines,
and the current scene values. `demo.json` reports box/polygon annotation counts,
classification segment counts, and the number of frames with scene labels.
Returned video bytes must match the plan's SHA-256, and the entire output MP4
must decode. Overlays use **exported Encord labels**, not substitute local labels.
Annotated MP4s are silent visual previews; original returned media is retained.

`import-labels` claims a new receipt before changing Encord and checkpoints each
mutation. Existing project titles and output receipts fail closed. After an
interruption, inspect the receipt before choosing new targets: blind retries
with new targets can create duplicates. Persistence failure immediately after
an API mutation can leave the last remote change absent from the receipt.
Failed runs do not automatically delete remote evidence.

Live validation covers both the original two-track box demo and a richer
three-clip run. The latter saved **nine video-local tracks, 346 boxes, 331 polygon
annotations, and ten classification segments**, then exported and compared all
of them. The clips show a bottle, basket, and moving gripper under neutral and
warm lighting; scene labels record bottle color-cue detection and lighting.
All 331 source frames and the resulting MP4s decoded successfully. The committed
live test also completed the entire graph against fresh Encord/S3 targets.

These demonstration prelabels use color/region heuristics, white balance, and
convex hulls. They are approximate, especially around occlusion, and still need
human review. The workflow accepts externally prepared plans, including model
predictions; it does not implement a general-purpose object detector. Exact
account/storage evidence, source clips, and the walkthrough MP4 stay private.

See the [transport guide](../../../docs/workbench/encord.md) for the roundtrip
demo, credentials, transfer modes, and standalone CLI/SDK usage.

## Additional live checks

`submit --plan-only` renders the CPU stages without launching a pod. The renderer
installs the `encord` extra from staged NPA source and forwards
`ENCORD_SSH_KEY_B64`. This validates the generated configuration; a real CPU
submission on a configured Linux operator and Kubernetes context is still needed
to verify pod bootstrap and credential access together.

The register discovery checksum fallback has a separate real-S3 test. Supply a
private JSON file with `bucket` and a nonempty `prefix`; the test uploads a small
repository MP4 under a fresh child prefix with a CRC32 checksum, proves there is
no full-object SHA-256, and observes HEAD → conditional GET → second HEAD:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_ENCORD_HASH_CONFIG=/private/encord-hash-config.json \
NPA_E2E_ENCORD_EVIDENCE_DIR=/private/encord-evidence \
npa/.venv/bin/python -m pytest npa/tests/e2e/test_encord_register_live.py -q
```

The uploaded fixture remains available for inspection. This test passed against
real S3; it exercises the register discovery path without making Encord mutations.
A full register push still needs an accessible Encord cloud integration. To test
that complete path, set `encord_transfer: register` and `encord_integration` in
the private config for `test_encord_roundtrip_live.py`, with a source prefix
containing objects without full-object SHA-256 metadata. Full register and CPU
pod execution are distinct from the completed local upload/labeling run.
