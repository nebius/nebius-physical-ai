<!-- register: operator guide | reader: Encord integration operators | consumed: task-time reference -->
# Encord transport keeps S3 as the dataset of record

Encord is remote SaaS. NPA runs a stateless CPU API and transfer client locally
or in the default workflow pod. This integration does not deploy an Encord
service or container.

The integration originated in Jonathan Lwowski's
[PR #339](https://github.com/nebius/nebius-physical-ai/pull/339).
Stew Tong authored the fail-closed transport, CLI, SDK, and workflows in
[PR #363](https://github.com/nebius/nebius-physical-ai/pull/363).

## Choose the transfer contract explicitly

`register` is the default. Encord receives object URLs and exact NPA identity
metadata while the bytes remain in S3. This retains S3 as the governed dataset
of record. Registration can still transfer the entire source through the NPA
client for integrity hashing; it is not a metadata-only operation.

Select `upload` explicitly with `--transfer upload` to create an Encord-managed
copy with an independent retention and deletion lifecycle.
Repeated uploads may create duplicates. A failed registration never activates
upload as a fallback.
Temporary upload files retain their media extension so the Encord SDK can
identify image MIME types correctly.

Strict client-only cloud integration access may limit Encord features that
need server-side media access. Confirm the features your Encord project needs
before choosing that access posture.

## Credentials

Create an Encord SSH public key in Encord and keep the corresponding private
key outside the repository. NPA accepts these credential references:

- `ENCORD_SSH_KEY` for the PEM value
- `ENCORD_SSH_KEY_B64` for a base64-encoded PEM, including workflow forwarding
- `ENCORD_SSH_KEY_FILE` for a local key path; `npa configure --save-env-credentials`
  persists this path without copying the key contents
- `ENCORD_DOMAIN` for a regional API domain override

Do not place credential values in workflow specs or examples. A local key file
path is not transferable to a workflow pod.

```bash
npa workbench health preflight --checks encord --offline
```

Omit `--offline` to perform the cheapest read-only authentication probe.

## Push media

```bash
npa workbench encord push \
  --input-path s3://<bucket>/raw-media/ \
  --integration <encord-cloud-integration> \
  --folder <encord-folder> \
  --dataset <encord-dataset> \
  --output-path s3://<bucket>/encord/push/push_receipt.json
```

Register mode requires an Encord cloud integration that can read the source
objects. Exact source URI, complete object key or URL, namespaced client
metadata, stable item UUID, or an explicit identity sidecar establishes
lineage. A basename never establishes identity. Conflicting exact assertions
remain unresolved and fail the completed receipt contract.

When S3 does not expose a full-object SHA-256, registration reads and hashes
the source before changing Encord state. This requires `s3:GetObject` access
in the NPA client role and transfers the source bytes once. Roles that previously had only listing/metadata
access must add read permission. Without it, discovery fails before any Encord
mutation. The hashing requirement has no opt-out; a full-object S3
`ChecksumSHA256` avoids this read. Budget bandwidth and latency for the entire
input prefix when that checksum is absent. A conditional GET and a second HEAD
reject objects that change during hashing. The receipt retains the opaque ETag
separately from the computed SHA-256 used by roundtrip verification.

Exact `npa.source_uri` metadata identifies the object even when its HTTP host
changes between pushes, provided the complete object path still agrees.
URL-only views of the same UUID may use another host under that same condition.
Contradictory object paths, source URIs or record IDs, or multiple matching UUIDs,
still fail closed. Without matching source metadata, conflicting URLs remain an
error, including when a sidecar supplies the UUID.

Use `--identity-sidecar s3://<bucket>/<key>.json` when existing Encord rows
cannot expose enough exact metadata for reconciliation.

The receipt is created before the first Encord mutation and checkpointed as
work progresses. Checkpoints reduce evidence loss, but no client can guarantee
a final durable checkpoint if the artifact store fails after Encord accepts a
mutation. NPA stops further mutation and exits nonzero in that case.

## Curate headlessly

`curate` asks Encord to select items into a Collection. It does not move media
bytes. Give each run a new Collection title. The command rejects a populated
Collection because those items could be mistaken for this run's selection.

```bash
npa workbench encord curate \
  --folder <encord-folder> \
  --filter width:128:16384 \
  --collection <new-collection> \
  --receipt-uri s3://<bucket>/encord/push/push_receipt.json \
  --output-path s3://<bucket>/encord/curate/curate_receipt.json
```

Add more `--filter` options as needed. Use `width`, `height`, `area`, or
`aspect-ratio` to filter on media metadata.
`brightness`, `sharpness`, and `file-size` require quality metrics already
computed for the folder in Encord. In Encord, open the folder in **Explore**,
choose **Configure folder** in the Embeddings panel, enable **Compute Encord
Embedding & Quality Metrics**, and wait for folder activity to show **Analysis
up to date**. The public SDK does not document a call to start that computation.
If nothing matches, the command writes a failed receipt. Check whether the
metrics are ready and whether the range is too narrow.

The receipt records the preset JSON and selected item UUIDs. Encord does not
expose a completion handle for preset insertion, so treat the UUID list as a
snapshot. The workflow verifier checks that the later pull matches it exactly.

## Pull media

```bash
npa workbench encord pull \
  --source collection \
  --source-id <encord-source-id> \
  --output-path s3://<bucket>/encord/pull
```

Project pulls do not initialize or export labels by default. The explicit
`--label-export initialize` option may create a label row or change remote
label status in Encord. Its manifest records that mutation posture.

New pulls emit `npa.encord.pull_manifest.v2`. In v2, `source_size` is zero until
transfer and then records the exact streamed byte count (or the exact S3 source
size for a server-side copy). `provider_reported_size` preserves Encord's catalog
size in both the manifest item and its sidecar; it can be rounded. Destination
size and SHA-256 verification use actual media bytes.

Readers in `verify-roundtrip` and `render-labels` accept both v1 and v2. Existing
v1 artifacts keep their original `source_size` value and meaning: provider/source
metadata, with a streamed-size fallback when unavailable. They are not silently
rewritten or relabeled as v2. Downstream schema allowlists must accept v2 before
consuming new pulls; consumers needing the original catalog value should read
`provider_reported_size` for v2. The partner specs now declare v2 handoffs.

## Verify a roundtrip

```bash
npa workbench encord verify-roundtrip \
  --receipt-uri s3://<bucket>/encord/push/push_receipt.json \
  --manifest-uri s3://<bucket>/encord/pull/manifest.json \
  --output-path s3://<bucket>/encord/verify/roundtrip_report.json
```

For a curated subset, also pass
`--curate-receipt-uri s3://<bucket>/encord/curate/curate_receipt.json`.
The verifier then expects exactly the selected UUIDs instead of every pushed
item, and checks that the pull came from the recorded Collection.

A roundtrip is verified only when this command consumes both final artifacts
and passes exact item identity, destination existence, size, and compatible
checksum checks.

When the source receipt contains SHA-256 but the destination bucket exposes only
an opaque ETag, verification streams the destination bytes and computes SHA-256.
The GET is conditional on the observed ETag, its byte count must match, and a
second HEAD checks that the object did not change during the read. ETags remain
opaque version identifiers; they are never treated as content hashes. This path
requires permission to read the destination object and transfers its full size.
Read failures and changed or mismatched bytes produce a failed durable report.

Three transport reference specs are available under
`workflows/partners/encord/`: `encord-push.yaml`,
`encord-pull.yaml`, and `encord-roundtrip-smoke.yaml`. Each spec writes artifacts
to S3. Push and roundtrip may also create or update Encord media records. Select
the operation explicitly and confirm the target integration, folder, dataset or
source, and S3 prefixes before submission. Pull with `--label-export none`
leaves Encord label state unchanged; `--label-export initialize` explicitly
initializes remote label state.

## Label videos and render exported annotations

The [Encord partner runbook](../../workflows/partners/encord/README.md) describes
`encord-labeling-demo.yaml`: upload → import labels → pull project labels and
media → verify → render. `npa workbench encord import-labels` creates a new
ontology and project from a content-bound plan with any number of box/polygon
tracks and temporal scene classifications. The matching
`render-labels` command checks actual exported labels against that plan before
rendering annotated MP4s. Both are exposed through `npa.sdk.workbench.encord`.
Imported labels remain programmatic prelabels awaiting human review.

## Run the workflow locally and retain an MP4 demo

Install the `encord` and `adapter` extras, configure Encord and S3 credentials,
and stage at least one valid MP4 under the selected source prefix. The Encord
client runs on the local CPU; Encord and object storage remain real services.

Create a private JSON file outside the repository containing config overrides
for `encord-roundtrip-smoke.yaml`. Explicitly set `bucket`, `prefix`,
`encord_media_uri`, and `encord_integration`. The prefix should include
`{{run.id}}` so each run gets new artifacts. The default folder and dataset are
also scoped to that run. Additional overrides use the spec's existing config
keys. Stage media on both sides of the default `width:16:64` filter,
including at least one retained MP4; the live test requires a strict subset.
Set `encord_transfer` to `upload` explicitly and `encord_integration` to
an empty string only when you want an Encord-managed media copy.

After confirming those targets, run the opt-in live test:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_ENCORD_CONFIG=/private/encord-config.json \
NPA_E2E_ENCORD_EVIDENCE_DIR=/private/encord-evidence \
npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_encord_roundtrip_live.py -q
```

`NPA_E2E_ENCORD_CONFIG` is required; without it the test skips without contacting
Encord. `NPA_E2E_ENCORD_EVIDENCE_DIR` is optional and defaults to pytest's temporary
directory. Its run subdirectory holds private workflow evidence, unchanged
`demo-*.mp4` returns, and `demo.json` with decoded frame counts and SHA-256 hashes.
The test runs push → curate → pull → verify, downloads the verified MP4,
checks its bytes against the report, and decodes every frame. It leaves the
Encord folders, datasets, and S3 artifacts available for review. Keep the
private workflow evidence outside the repository.

## Python SDK

```python
from npa.sdk.workbench import encord

receipt = encord.push(
    input_path="s3://<bucket>/raw-media/",
    integration="<encord-cloud-integration>",
    folder="<encord-folder>",
    output_path="s3://<bucket>/encord/push/push_receipt.json",
)

curation = encord.curate(
    folder="<encord-folder>",
    filters=["width:128:16384"],
    collection="<new-collection>",
    source_receipt_uri=receipt.receipt_uri,
    output_path="s3://<bucket>/encord/curate/curate_receipt.json",
)

manifest = encord.pull(
    source="collection",
    source_id=curation.collection_uuid,
    output_path="s3://<bucket>/encord/pull",
)
```

The CLI and SDK call the same fail-closed implementation.
