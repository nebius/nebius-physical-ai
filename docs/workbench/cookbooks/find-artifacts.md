# Find artifacts across object-storage buckets

Use `npa studio search` to find videos, simulation recordings, images, datasets,
and reports from Workbench runs. Search works before you create a Studio project
and does not require a renderer, agent VM, GPU, or workflow registry. It lists S3
objects using your external NPA configuration; it does not provision resources,
download media, or modify storage.

## Choose an interface

| Need | Interface |
| --- | --- |
| Search object keys in one or several buckets | `npa studio search --bucket "<bucket>"` |
| Search buckets visible to configured credentials | `npa studio search` or `--all-projects` |
| Discover buckets through the selected tenant's inventory | `npa studio search --discover-tenant` |
| List bucket resources in a project | `npa storage bucket list --project "<project-alias>"` |
| Browse runs and open their media interactively | The [Workbench agent's artifact browser](../../agent.md) |

Bucket inventory and object access are different permissions. A bucket returned
by Nebius inventory may still deny S3 listing or downloads. Search preserves
successful results and reports denied sources instead of claiming complete
coverage.

## Prerequisites and first search

Install a version of NPA containing Studio and configure your own project and
storage credentials using [Workbench setup](../getting-started.md) and
[project configuration](../../configuration.md). Check the installed surface:

```bash
npa studio search --help
```

Select a configured project alias and an existing bucket. Quoted angle-bracket
values below are placeholders to replace with your own values:

```bash
npa studio search --project "<project-alias>" \
  --bucket "<artifact-bucket>" --prefix "runs/" \
  --kind video --read-metadata > artifact-search.json
```

The result is a JSON inventory, not downloaded media. `--prefix` is an exact S3
key prefix, not an S3 URI. Omit it to search the whole bucket. Search follows all
object-listing pages; it has no implicit first-page or known-workflow restriction.
No cleanup is needed in the cloud. Keep or remove the local inventory as needed.

## Select the scope

Repeat `--bucket` to search only named buckets. This skips S3 bucket enumeration,
which helps when a credential can list objects but cannot list all buckets:

```bash
npa studio search --project "<project-alias>" \
  --bucket "<simulation-bucket>" --bucket "<evaluation-bucket>" \
  --kind video --output-format text
```

Without `--bucket`, search enumerates credential-visible buckets and also checks
the configured checkpoint bucket. The default uses the current credential
context. Select several configured aliases or all configured projects:

```bash
npa studio search --project "<simulation-alias>" --project "<evaluation-alias>" \
  --query rollout --kind video > rollout-search.json
npa studio search --all-projects --kind rerun > recordings.json
```

`--all-projects` means every project configured locally, not every project in a
tenant. Each alias resolves its own credentials and endpoint. A repeated bucket
is searched in each selected context; retain the returned project and endpoint
with the bucket and key.

To discover projects and buckets from the selected tenant's Nebius inventory:

```bash
npa studio search --project "<project-alias>" --discover-tenant \
  --query isaac --kind video --read-metadata > tenant-artifacts.json
```

This requires a configured tenant and authenticated Nebius inventory access.
Regional endpoints come from discovered project regions. Known local project
aliases use their own S3 credentials; other discovered buckets are attempted
with the selected credential context. Permissions still apply. This mode accepts
one project context and cannot combine with `--all-projects`. An optional
`--bucket` filters the discovered inventory; a requested bucket that was not
discovered is reported explicitly.

## Filter the results

```bash
npa studio search --project "<project-alias>" --bucket "<artifact-bucket>" \
  --prefix "runs/" --query evaluation --kind json \
  --since "2026-01-01T00:00:00Z" --read-metadata > evaluations.json
```

| Option | Meaning |
| --- | --- |
| `--prefix` | Server-side, case-sensitive key prefix; the best way to narrow a known run |
| `--query` | Case-insensitive substring of the object key; not semantic, file-content, or metadata search |
| `--kind` | `video`, `image`, `rerun`, `mcap`, `json`, `text`, or `download` |
| `--since` | Inclusive object modification time, with an explicit timezone |
| `--read-metadata` | HEAD requests for matching objects to read declared provenance |
| `--output-format` | `json` by default, or a compact `text` inventory |

Except for the prefix, filters are applied after listing. There is no wildcard
or regular-expression query syntax. Unknown file formats remain visible with
the `download` hint when no kind filter is set; search does not hide them because
they came from a new model or an unfamiliar workflow.

## Interpret coverage and provenance

The JSON schema is `npa.studio.artifact-search/v1`. Inspect these fields:

| Field | Interpretation |
| --- | --- |
| `complete` | All requested discovery and listing operations completed within the selected scope |
| `discovery` | Project/bucket discovery coverage and errors |
| `sources` | Each project, endpoint, bucket, prefix, objects scanned, matches, and listing error |
| `artifacts` | Exact object identity, size, modification time, ETag, render hint, and optional provenance |
| `count` | Number of matched object rows, not a count of successful runs |

Exit code **0** means complete listing coverage; **1** means incomplete discovery
or listing, with successful rows retained. Invalid command arguments also return
a nonzero exit code. An empty, complete result means no matches in that scope;
an empty, incomplete result does not prove there are no artifacts. When S3 bucket
enumeration is denied, the configured bucket may still be searched, but broad
discovery remains incomplete. Retry with explicit buckets when their names are
known.

Metadata status is separate from listing coverage. With `--read-metadata`,
`provenance.status` is `declared`, `unknown`, or `unavailable`; without it, status
is `not_read`. A failed HEAD request does not turn a successful object listing
into a failed listing. Read its per-object error before relying on provenance.

Supported metadata includes tool/generator, model, run ID, GPU, SHA-256, and tool
evidence. Writers can use `npa-tool`, `npa-model`, `npa-run-id`, `npa-gpu`,
`npa-sha256`, and `npa-tool-evidence`; the corresponding unprefixed names are also
recognized. Search reports declarations, not independently verified execution.
It does not automatically read neighboring run manifests or infer a generating
tool from a filename. For older outputs with no metadata, inspect the associated
`result.json`, run manifest, and source hashes before attributing the artifact.
An ETag is not generally a SHA-256 or content-integrity proof.

## Retrieve an explicitly selected result from Python

Keep the complete source tuple from the search result. The project-aware NPA
client preserves the selected credential context and regional endpoint:

```python
import hashlib
import json
from pathlib import Path

from npa.clients.project_credentials import s3_client_for_project

report = json.loads(Path("artifact-search.json").read_text())
source = next(
    row for row in report["artifacts"]
    if row["key"] == "<selected-object-key>"
    and row["bucket"] == "<selected-bucket>"
    and row["project"] == "<selected-project-alias>"
)
client = s3_client_for_project(source["project"], endpoint_url=source["endpoint"])
destination = Path("selected-video.mp4")
client.download_file(source["bucket"], source["key"], str(destination))
digest = hashlib.sha256()
with destination.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
receipt = {
    "source": source,
    "local_path": str(destination),
    "downloaded_sha256": digest.hexdigest(),
}
Path("selected-video.provenance.json").write_text(json.dumps(receipt, indent=2))
```

Run this example with NPA installed. Choose the exact
row after reviewing the inventory; for the default context, `project` is `null`
in JSON and `None` in Python. Use a fresh destination filename. A download can
still fail even after listing succeeds. Compare the downloaded SHA-256 with the
original run evidence; the receipt alone does not prove the object stayed
unchanged between search and download.

For Studio editing, create a local project from the downloaded media and retain
the private provenance sidecar. Follow the [Studio guide](../../demos/workbench-studio/README.md)
to author the storyboard and render it. Search itself is not a media-quality
review, a model run, or an archive/publishing command.

## Browse through the Workbench agent

The authenticated agent discovers runs from S3, groups their artifacts, and opens
videos, images, Rerun recordings, reports, or downloads for unknown formats.
Use its artifact browser or ask the coding agent to find outputs for a run.
The underlying API starts with `GET /api/artifacts/runs`, then
`GET /api/artifacts/run/{run_id}` and `POST /api/sim-viz/load-artifact`.
Preserve server-issued run references and the selected source context through
preview and load actions; a bare S3 URI is not a substitute for that authorization
context. See the [artifact-discovery skill](../../../skills/atomic/find-artifacts/SKILL.md)
for the API sequence.

## Keep operational results private

Search JSON and download receipts contain project identities, regional
endpoints, bucket names, and object keys. Store them outside Git and public
documentation. Credentials remain in external NPA configuration; no customer
bucket, tenant, or GPU reservation is built into Studio. This command runs once
per invocation and does not install a background polling process.
