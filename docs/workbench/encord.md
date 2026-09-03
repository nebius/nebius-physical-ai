<!-- register: operator guide | reader: Encord integration operators | consumed: task-time reference -->
# Encord transport keeps S3 as the dataset of record

Encord is remote SaaS. NPA runs a stateless CPU API and transfer client locally
or in the default workflow pod. This integration does not deploy an Encord
service or container.

## Choose the transfer contract explicitly

`register` is the default. Encord receives object URLs and exact NPA identity
metadata while the bytes remain in S3. This retains S3 as the governed dataset
of record.

Select `upload` explicitly with `--transfer upload` to create an Encord-managed
copy with an independent retention and deletion lifecycle.
Repeated uploads may create duplicates. A failed registration never activates
upload as a fallback.

Strict client-only cloud integration access may limit Encord features that
need server-side media access. Confirm the features your Encord project needs
before choosing that access posture.

## Credentials

Create an Encord SSH public key in Encord and keep the corresponding private
key outside the repository. NPA accepts these credential references:

- `ENCORD_SSH_KEY` for the PEM value
- `ENCORD_SSH_KEY_B64` for a base64-encoded PEM, including workflow forwarding
- `ENCORD_SSH_KEY_FILE` for a local key path
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

Use `--identity-sidecar s3://<bucket>/<key>.json` when existing Encord rows
cannot expose enough exact metadata for reconciliation.

The receipt is created before the first Encord mutation and checkpointed as
work progresses. Checkpoints reduce evidence loss, but no client can guarantee
a final durable checkpoint if the artifact store fails after Encord accepts a
mutation. NPA stops further mutation and exits nonzero in that case.

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

## Verify a roundtrip

```bash
npa workbench encord verify-roundtrip \
  --receipt-uri s3://<bucket>/encord/push/push_receipt.json \
  --manifest-uri s3://<bucket>/encord/pull/manifest.json \
  --output-path s3://<bucket>/encord/verify/roundtrip_report.json
```

A roundtrip is verified only when this command consumes both final artifacts
and passes exact item identity, destination existence, size, and compatible
checksum checks.

Three reference specs are available under
`npa/workflows/workbench/npa-workflows/`: `encord-push.yaml`,
`encord-pull.yaml`, and `encord-roundtrip-smoke.yaml`. Each spec writes artifacts
to S3. Push and roundtrip may also create or update Encord media records. Select
the operation explicitly and confirm the target integration, folder, dataset or
source, and S3 prefixes before submission. Pull with `--label-export none`
leaves Encord label state unchanged; `--label-export initialize` explicitly
initializes remote label state.

## Python SDK

```python
from npa.sdk.workbench import encord

receipt = encord.push(
    input_path="s3://<bucket>/raw-media/",
    integration="<encord-cloud-integration>",
    folder="<encord-folder>",
    output_path="s3://<bucket>/encord/push/push_receipt.json",
)

manifest = encord.pull(
    source="collection",
    source_id="<encord-source-id>",
    output_path="s3://<bucket>/encord/pull",
)
```

The CLI and SDK call the same fail-closed implementation.
