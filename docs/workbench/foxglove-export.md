# Export MCAP and open it in Foxglove Web

The NPA agent preserves two distinct operations over one canonical artifact:

- **Download MCAP** resolves the active run's canonical S3 recording and downloads
  byte-identical data from the agent's bounded public HTTPS transport cache.
- **Open in Foxglove Web** converts or reuses that same MCAP, uploads it to
  Foxglove Cloud under a content-derived key, waits until the recording is
  indexed, and opens the official recording deep link.

There is no Foxglove Desktop action. The single user-facing Lichtblick control
is the in-page **Lichtblick** viewer tab.

## Canonical S3 artifact contract

Every explicit export starts from a real S3-discovered run. Its stable artifact
is `<run-prefix>/<run-id>/reports/sim2real.mcap`; provenance is stored beside it
as `reports/sim2real.mcap.provenance.json` using schema
`npa.canonical-mcap.v1`. The sidecar records the S3 URI, SHA-256, byte size,
source artifacts, source mode (`native-reused` or
`generated-from-s3-artifacts`), channels, message count, timestamp ranges, and
timestamp semantics.

A valid existing native `reports/sim2real.mcap` is authoritative and is never
repacked. The agent validates MCAP magic and structure before reuse. If the run
has no canonical recording, its S3 artifacts are staged into a temporary
directory, converted once, and uploaded to that reserved reports key. Export
fails visibly if discovery, download, validation, upload, or provenance storage
fails. Successful writes invalidate the run-list cache, so Runs & Artifacts
shows the MCAP immediately.

Lichtblick, Download MCAP, and Foxglove Cloud all receive those same bytes. The
fixed Lichtblick recording and random Foxglove publication are ephemeral caches;
the S3 URI and SHA-256 identify the persistent artifact. Run switching clears
the previous run's viewer, canonical, transport, and Cloud state.

## Foxglove Web contract

The official link uses Foxglove's documented Cloud data-source form:

```text
https://app.foxglove.dev/~/view?ds=foxglove-stream&ds.recordingId=<recording-id>&layoutId=<layout-id>&ds.start=<rfc3339>&ds.end=<rfc3339>&time=<rfc3339>
```

The agent derives a v1 presentation from the canonical MCAP inspection: up to
two Image panels are bound only to real `foxglove.CompressedImage` topics, 3D is
included only when a PointCloud, FrameTransform, or SceneUpdate schema exists,
Plot paths come only from numeric JSON-schema fields, and Log is included only
for a real `foxglove.Log` topic. The first seek is 250 ms into the bounded
recording range, where synchronized topics have begun painting.

The shared organization layout is keyed by the stable name
`NPA Physical AI rich visualization v1`. The API lists and compares its opaque
data before writing: unchanged layouts are reused, while a changed compatible
topic contract updates the existing ID instead of spending another layout.
Foxglove requires API-key-created layouts to use `ORG_WRITE`. If the token or
plan permits recording upload but not layout mutation, opening the indexed
recording remains available without `layoutId` and the API response reports the
layout limitation honestly; the user must select or create a suitable shared
layout in Foxglove.

The API token is read server-side from
`tokens.FOXGLOVE_API_TOKEN` in `~/.npa/credentials.yaml` (mode `0600`). It is
used only in the Foxglove API Authorization header. It is never included in the
recording link, agent config/status responses, subprocess arguments, browser
payloads, or normal logs.

Uploads are explicit: only **Open in Foxglove Web** invokes them. The content
SHA-256 determines a stable recording key, so unchanged MCAPs and in-progress
imports are reused rather than uploaded again. The action waits for Foxglove's
`complete` import state and surfaces missing-token, project-selection,
permission, indexing, and plan/storage/rate-quota errors.

Foxglove sign-in and organization access are still required in the browser. The recording is already
stored and indexed in Foxglove Cloud before the tab opens, so Foxglove does not
need to fetch the agent's self-signed IP URL.

## Download and transport contract

The random agent URL remains intentionally unauthenticated for MCAP download,
CORS, and byte-range validation. It is not the Foxglove Web data source. Anyone
holding the URL can read it until the agent prunes the publication. The export
route refuses missing, non-HTTPS, credential-bearing, loopback, private,
link-local, reserved, and metadata origins.

Generated image sequences use `timestamps=synthetic-fps`. Independent camera
topics share one synthetic epoch and advance by each topic's own frame index,
so concurrent views overlap without claiming capture synchronization. Explicit
source frame timestamps are preserved and reported as `source` (or
`source-and-synthetic-fps` when mixed with untimestamped artifacts). The
conventional `/camera` topic is assigned to a real stream named `camera`, or to
the first deterministic real image stream when no stream has that name. The
remaining streams retain descriptive topics. No image, transform, pose, point
cloud, joint state, or telemetry is fabricated.
The embedded default selects only that guaranteed real image topic; optional
channels remain available in Topics instead of being pinned to empty panels.

CLI callers can convert/export locally with `npa workbench foxglove export-run`
and build a web-only link for an already indexed recording with:

```bash
npa workbench foxglove open --recording-id <recording-id>
```

References: [Foxglove shareable links](https://docs.foxglove.dev/docs/visualization/shareable-links),
[Foxglove layouts](https://docs.foxglove.dev/docs/visualization/layouts),
[Foxglove panels](https://docs.foxglove.dev/docs/visualization/panels),
[Foxglove importing data](https://docs.foxglove.dev/docs/data/importing-data),
[Foxglove API](https://docs.foxglove.dev/api), and
[Foxglove pricing](https://foxglove.dev/pricing).
