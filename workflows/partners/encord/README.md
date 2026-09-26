<!-- register: partner runbook | reader: Encord operators | consumed: task-time reference -->
# Encord media and labeling workflows

These workflows use the real Encord SDK and S3. They run on a local CPU or in
the standard NPA workflow runtime; they do not deploy an Encord service.

| Workflow | Result |
| --- | --- |
| [Labeling demo](encord-labeling-demo.yaml) | Upload videos → create ontology/project → save bounding-box tracks → export labels/media → verify → render annotated MP4s. |
| [Roundtrip demo](encord-roundtrip-smoke.yaml) | Push → pull → verify exact media identity and bytes; no labeling. |
| [Push](encord-push.yaml) | Register or explicitly upload media and write a durable receipt. |
| [Pull](encord-pull.yaml) | Materialize an existing dataset, collection, or project into S3. |

The three transport workflows moved here from `workflows/testing/`. Their file
names and transport behavior are unchanged. Imported prelabels are real object
annotations, with human review still pending.

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
Encord's decoded media. Each distinct `class_name` becomes an ontology object.

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
coordinates with the import plan. Missing, extra, or changed annotations fail.
Returned video bytes must match the plan's SHA-256, and the entire output MP4
must decode. Overlays use **exported Encord labels**, not substitute local labels.
Annotated MP4s are silent visual previews; original returned media is retained.

`import-labels` claims a new receipt before changing Encord and checkpoints each
mutation. Existing project titles and output receipts fail closed. After an
interruption, inspect the receipt before choosing new targets: blind retries
with new targets can create duplicates. Persistence failure immediately after
an API mutation can leave the last remote change absent from the receipt.
Failed runs do not automatically delete remote evidence.

The initial live validation imported two tracks with 266 frame-level boxes,
exported and compared every box, verified the source SHA-256, and rendered and
decoded all 169 frames. These were simple color-based prelabels for a visible
bottle and basket, not a trained model or human-approved ground truth. Exact
account and storage evidence stays private.

See the [transport guide](../../../docs/workbench/encord.md) for the roundtrip
demo, credentials, transfer modes, and standalone CLI/SDK usage.
