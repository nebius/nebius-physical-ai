# Archive completed Ray CLIP results

After a native Ray CLIP Job succeeds, download its results with the
[existing rsync guide](fast-source-iteration.md) and stop every writer to that
local directory. The optional CPU companion below archives a complete result
tree to your private S3 storage and restores it after the original driver is
gone. The local workflow needs no S3 account.

This companion handles completed basic `embed.py` and advanced `application.py`
results. It does not resume incomplete checkpoints, recover a lost driver, or
promise that GPU inference will not repeat. Ray Jobs still owns submission and
status; there is no uploader Job or required finish handshake.

## Archive and restore

Use Linux and an installed NPA checkout (`npa[dev,adapter]` provides the test
runtime). Configure `AWS_ENDPOINT_URL` or `NEBIUS_S3_ENDPOINT`,
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` privately for your selected
workload bucket. Verify bucket ownership and exact prefix write/read permission;
run `npa workbench health preflight --checks s3 --json` with
`NPA_CHECKPOINT_BUCKET` set privately to that same bucket's unsigned S3 URI.
Keep results, receipts and credentials outside the submitted source directory.

From the repository root, choose an unused S3 prefix per result and retain the
returned JSON in a private file:

```bash
umask 077
npa/.venv/bin/python npa/workflows/workbench/ray-clip-development/archive.py archive \
  --input-path "$COMPLETED_RESULTS" \
  --output-path "$CLIP_ARCHIVE_URI" > "$ARCHIVE_RECEIPT"
```

`COMPLETED_RESULTS` is an absolute, quiescent directory. `CLIP_ARCHIVE_URI` is an
explicit unsigned `s3://<your-bucket>/<unique-result-prefix>` URI. The receipt
contains `manifest_sha256`, the format, file count and byte count. Keep that hash
with the archive location: restore requires this independently retained identity.
The JSON does not disclose resource addresses or report contents.

Set `RESTORE_PARENT` to an absolute path, create that owner-only parent, and
restore into a **new** child directory:

```bash
mkdir -m 700 "$RESTORE_PARENT"
npa/.venv/bin/python npa/workflows/workbench/ray-clip-development/archive.py restore \
  --input-path "$CLIP_ARCHIVE_URI" \
  --output-path "$RESTORE_PARENT/results" \
  --manifest-sha256 "$MANIFEST_SHA256"
```

Set `MANIFEST_SHA256` to the hash from the retained receipt. Inspect the restored
`preview.png`, `report.json`, `embeddings.parquet` and `lance/embeddings.lance`.
Original bytes and provenance remain unchanged. An existing `.rrd` is preserved
when the source checksum inventory includes it; no converter is required or run.

## Verification and failure behavior

The companion checks the complete `SHA256SUMS` inventory and, for basic results,
the matching `sha256.json`. It rejects missing or unlisted files, duplicate
names, URL query/fragment delimiters (`?` and `#`), traversal, symlinks, hard
links and special files. It reopens Parquet and
Lance, checks every record and normalized 512-dimensional vector, compares
preview input bytes and retrieval inventories, and verifies report hashes.
Advanced results additionally require execution identity, complete committed
shards and successful actor-cleanup evidence. A cancelled tree is insufficient.

Before opening Lance, it validates the local manifest and transaction against
the initial-version format emitted by the current recipe (LanceDB 0.30.2,
Lance writer 4.0.0, data format 2.0). Every fragment must reference inventoried
local data. Shallow clones, extra versions, indexes, deletions, external row
metadata, branches, extensions and unknown metadata are unsupported and rejected
before any Lance reader runs. This restriction prevents a restored table from
silently depending on files outside its result tree. Keep the recipe's original
Lance directory; use a separate copy for subsequent table edits.

Archive freezes a private local copy, checks its formats, creates each
`files/<relative-name>` object conditionally, and reads back exact bytes. It
rescans the original tree, including file identity and modification metadata,
before creating `complete.json`. That deterministic manifest binds every name,
size and SHA-256; its own SHA-256 is the retained restore identity. The source
must remain quiescent throughout the command. These checks detect mutation;
they do not lock an independently running application.

Identical concurrent publications and retries accept only byte-identical
existing objects. Different bytes fail without overwriting. Interrupted uploads
can leave partial payload objects; retry the same tree and prefix. Completion
is never inferred from a prefix listing or an earlier manifest alone. If a
completed archive is subsequently changed by another storage writer, restore
and retry detect the corruption. Conditional publication is not S3 Object Lock;
bucket access and retention policy remain yours.

Restore reads only the hash-verified manifest and listed objects. It validates
bytes and formats in a private sibling staging directory, then atomically
publishes with Linux `renameat2(RENAME_NOREPLACE)`. Even a racing empty
destination is preserved. A failed verification removes staging and leaves no
successful destination. Unsupported atomic publication fails closed. Preserve
the restore parent path and its ancestors throughout the command; a detected
replacement fails before publication. Preserve the source until archive succeeds;
retain your own backup/retention policy.

The current StorageClient byte API buffers one object at a time. Local staging
needs space for one complete tree; archive also retains the original. This is a
CLIP recipe companion, not a general storage service or workload orchestrator.

## Run the live CPU test

Prepare an owner-only JSON file outside the checkout with `source_paths` (a list
of complete public/synthetic CLIP directories), `archive_prefix` (a unique prefix
in a preflighted workload bucket), and `evidence_dir` (a new private local path).
Verify original Job completion, source provenance and driver absence separately
before reusing retained GPU outputs. Never select a trajectory dataset as
workload storage. Then run:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_RAY_CLIP_ARCHIVE_LIVE_CONFIG="$PRIVATE_LIVE_CONFIG" \
npa/.venv/bin/python -m pytest npa/tests/e2e/test_ray_clip_archive_live.py -q
```

The test archives and reads back actual bytes, retries identically, restores
twice, rejects a conflicting result, deliberately corrupts a proven owned test
object and verifies rejection without a destination. It journals exact object
creation intents privately and deletes only successfully created test objects,
checking each is absent afterward. Keep that ledger if a transport interruption
requires reconciliation. It creates no compute, bucket or IAM resources.
