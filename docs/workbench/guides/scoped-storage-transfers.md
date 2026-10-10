# Scoped GCS and S3 transfers

[Manual operations](manual-workflow-operations.md) · [Dataset batches](paidf-dataset-batches.md) · [Workbench docs](../README.md)

Copy one selected input into S3, or one reviewed output back to GCS. The examples
verify readback bytes and remove private staging files on success or failure.
They use Bash on Linux, authenticated Google Cloud and AWS CLIs, and
`sha256sum`. Set `RUN_ID` from the [manual run sequence](manual-workflow-operations.md#4-run-and-inspect-one-workflow).

| Task | Example |
| --- | --- |
| Stage a GCS input for PAIDF | [GCS → S3](#copy-input-from-gcs-to-s3) |
| Return a reviewed output to GCS | [S3 → GCS](#copy-output-from-s3-to-gcs) |

Use a run-scoped prefix, separate credentials for each provider, and the
smallest permissions each identity needs. A practical split is: ingress can
read one source object, the workflow can read its input and write its own run
prefix, and egress can read only selected final objects. Do not grant a workflow
identity whole-bucket read/write/delete permissions, and do not share one
credential between the external source and the S3-compatible destination.

PAIDF accepts `s3://` inputs; stage a `gs://` object into S3 first. Set the
selected project's verified storage endpoint and private AWS profile explicitly.
`configure --show --env` exports host defaults; it does not verify that the
endpoint belongs to your selected project. The examples follow the official
[Google Cloud Storage](https://cloud.google.com/sdk/gcloud/reference/storage/cp)
and [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html) references.

## Copy input from GCS to S3

```bash
(
set -euo pipefail
umask 077
PRIVATE_INGRESS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-private-ingress.XXXXXX")"
trap 'rm -rf -- "$PRIVATE_INGRESS_DIR"' EXIT
chmod 700 "$PRIVATE_INGRESS_DIR"
S3_ENDPOINT='<verified-selected-project-storage-endpoint>'
S3_PROFILE='<private-scoped-ingress-aws-profile>'
GCS_OBJECT='gs://<source-bucket>/<input-key>.mp4'
LOCAL_INPUT="$PRIVATE_INGRESS_DIR/source.mp4"
S3_READBACK="$PRIVATE_INGRESS_DIR/source.s3-readback.mp4"
S3_BUCKET='<your-bucket>'
S3_KEY="ingress/$RUN_ID/source.mp4"
INPUT_URI="s3://$S3_BUCKET/$S3_KEY"

GCS_SOURCE_BYTES="$(gcloud storage objects describe "$GCS_OBJECT" --format='value(size)')"
gcloud storage cp "$GCS_OBJECT" "$LOCAL_INPUT"
LOCAL_SOURCE_BYTES="$(wc -c < "$LOCAL_INPUT" | awk '{print $1}')"
test "$GCS_SOURCE_BYTES" = "$LOCAL_SOURCE_BYTES"
SOURCE_SHA256="$(sha256sum "$LOCAL_INPUT" | awk '{print $1}')"

aws --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT" s3 cp "$LOCAL_INPUT" "$INPUT_URI"
aws --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT" s3 cp "$INPUT_URI" "$S3_READBACK"
test "$(wc -c < "$S3_READBACK" | awk '{print $1}')" = "$LOCAL_SOURCE_BYTES"
test "$(sha256sum "$S3_READBACK" | awk '{print $1}')" = "$SOURCE_SHA256"
)
```

The size check compares the downloaded input with GCS metadata. The hash check
verifies that S3 readback matches the downloaded bytes. The GCS identity reads
only `GCS_OBJECT`; the separate S3 identity writes and reads only `INPUT_URI`.
Keep the destination URI privately for submission through `--input-uri`;
the example's variables are local to its subshell. A failed copy or verification
stops later transfers and removes staging files.

## Copy output from S3 to GCS

Choose one exact output object from the reviewed run. Copy it to a separate
Cloud Storage destination, read it back, and compare the complete bytes:

```bash
(
set -euo pipefail
umask 077
PRIVATE_EGRESS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-private-egress.XXXXXX")"
trap 'rm -rf -- "$PRIVATE_EGRESS_DIR"' EXIT
S3_ENDPOINT='<verified-selected-project-storage-endpoint>'
S3_PROFILE='<private-scoped-egress-aws-profile>'
OUTPUT_URI='<exact-s3-uri-from-npa-workflow-artifacts>'
LOCAL_OUTPUT="$PRIVATE_EGRESS_DIR/augmented_video.mp4"
GCS_OUTPUT='gs://<egress-bucket>/<run-scoped-prefix>/augmented_video.mp4'
GCS_READBACK="$PRIVATE_EGRESS_DIR/augmented_video.gcs-readback.mp4"

aws --profile "$S3_PROFILE" --endpoint-url "$S3_ENDPOINT" s3 cp "$OUTPUT_URI" "$LOCAL_OUTPUT"
OUTPUT_BYTES="$(wc -c < "$LOCAL_OUTPUT")"
OUTPUT_SHA256="$(sha256sum "$LOCAL_OUTPUT" | awk '{print $1}')"
gcloud storage cp "$LOCAL_OUTPUT" "$GCS_OUTPUT"
gcloud storage cp "$GCS_OUTPUT" "$GCS_READBACK"
test "$(wc -c < "$GCS_READBACK")" = "$OUTPUT_BYTES"
test "$(sha256sum "$GCS_READBACK" | awk '{print $1}')" = "$OUTPUT_SHA256"
cmp -s "$LOCAL_OUTPUT" "$GCS_READBACK"
)
```

This egress step verifies the complete selected object rather than relying on a
listed size or optional provider checksum metadata. Use an egress GCS identity
that can write and read only `GCS_OUTPUT`, distinct from both ingress identities.
Use a recursive copy only for an explicitly reviewed run-scoped prefix, never a
whole bucket.
