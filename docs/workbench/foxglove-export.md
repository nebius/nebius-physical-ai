# Export MCAP and open it in Foxglove Web

The NPA agent preserves two distinct operations over one canonical artifact:

- **Download MCAP** resolves the active run's canonical S3 recording and downloads
  byte-identical data from the agent's bounded public HTTPS transport cache.
- **Open in Foxglove Web** converts or reuses that same MCAP and opens the
  official `remote-file` deep link to the agent's public HTTPS transport URL.
  It does not put basic auth or an API token in the URL.

There is no Foxglove Desktop action. Lichtblick remains the in-page MCAP viewer
tab, with its established open and reload controls.

## Canonical S3 artifact contract

Every explicit export starts from a real S3-discovered run. Its stable artifact
is `<run-prefix>/<run-id>/reports/sim2real.mcap`; provenance is stored beside it
as `reports/sim2real.mcap.provenance.json` using schema
`npa.canonical-mcap.v2`. The sidecar records the S3 URI, SHA-256, byte size,
source artifacts, source mode (`native-reused` or
`generated-from-s3-artifacts`), channels, message count, timestamp ranges, and
timestamp semantics.

A valid existing native `reports/sim2real.mcap` is authoritative when it carries
the current visualization contract. The agent validates MCAP magic, structure,
and contract version before reuse; an older rich contract is regenerated at the
same exact S3 key so incompatible cached bytes cannot survive. If the run
has no canonical recording, its S3 artifacts are staged into a temporary
directory, converted once, and uploaded to that reserved reports key. Export
fails visibly if discovery, download, validation, upload, or provenance storage
fails. Successful writes invalidate the run-list cache, so Runs & Artifacts
shows the MCAP immediately.

Lichtblick, Download MCAP, and Foxglove Web all receive those same bytes. The
fixed Lichtblick recording and random Foxglove publication are ephemeral caches;
the S3 URI and SHA-256 identify the persistent artifact. Run switching clears
the previous run's viewer, canonical, and transport state.

## Foxglove Web contract

The official link uses Foxglove's documented remote-file data-source form:

```text
https://app.foxglove.dev/~/view?ds=remote-file&ds.url=<absolute-public-https-mcap>&time=<rfc3339>
```

`ds.url` is encoded once as a query component; after Foxglove parses the deep
link it is byte-for-byte the `recording_url` returned by the agent. The link may
include an initial `time` 250 ms into the inspected recording. Live sources use
the same documented URL surface with `ds=foxglove-websocket` or
`ds=rosbridge-websocket` and one `ds.url`.

The MCAP URL must be absolute public HTTPS, have no userinfo, and remain
unauthenticated so the cross-origin Foxglove application can fetch it. nginx
serves it with wildcard CORS, an `OPTIONS` response that allows `Range`, byte
ranges, and compression disabled. The filename contains an unguessable token
and old publications are pruned.

Foxglove sign-in and organization access may still be required in the browser.
The agent reports the hosted iframe as connecting or ready and surfaces SDK or
host errors; it does not claim that an external sign-in wall loaded recording
pixels. The cross-origin iframe also means Describe this is state/text-only.

## Optional Foxglove Cloud import

The backend retains an explicit `cloud_import: true` export mode for operators
who want a content-addressed Foxglove Cloud recording and managed layout. The
ordinary UI action does not invoke it and does not require a Foxglove API token.

The API token is read server-side from
`tokens.FOXGLOVE_API_TOKEN` in `~/.npa/credentials.yaml` (mode `0600`). It is
used only in the Foxglove API Authorization header. It is never included in the
recording link, agent config/status responses, subprocess arguments, browser
payloads, or normal logs.

When the token is already exported in the operator shell, persist it without
putting its value on a command line:

```bash
npa configure --no-interactive --save-env-credentials
```

The next agent deploy or bootstrap copies it into the VM's private
`credentials.yaml`; it is not added to the shared workbench environment.

Cloud imports are content-addressed by the canonical SHA-256, reuse unchanged or
in-progress imports, and surface missing-token, project-selection, permission,
indexing, plan, storage, and rate-quota errors. The indexing wait has a monotonic
300-second server deadline by default. Set
`NPA_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS` before deploy/bootstrap to override
it with a positive finite number of seconds up to 3600; invalid values fail before any
infrastructure or remote bootstrap mutation. Cloud-import browser requests use a
finite 360-second deadline, while local MCAP conversion retains its uncapped
browser request because large local conversions can legitimately take longer.

## Download and transport contract

The random agent URL remains intentionally unauthenticated for MCAP download and
is the Foxglove Web `remote-file` data source. Anyone
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
The shared layout discovers every `foxglove.CompressedImage` topic. It presents
the guaranteed `/camera` stream first and exposes every additional source camera
as a clearly labelled tab; single-camera recordings retain one working tab. The
current rich contract uses the official `@foxglove/schemas@2.1.0`
`foxglove.SceneUpdate` JSON shape, including an explicit `items` schema on every
primitive array. RGB frames are source-faithful; no depth, calibration,
extrinsics, or world reprojection is implied unless the input provides it.

CLI callers can convert/export locally with `npa workbench foxglove export-run`
and build a web-only link for an already indexed recording with:

```bash
npa workbench foxglove open --recording-id <recording-id>
```

## Cypress regression tiers

The mocked tier serves the production Agent UI, the real pinned
`@foxglove/embed` browser build, and a protocol-accurate local viewer stand-in:

```bash
bash npa/scripts/run_agent_cypress.sh --mock
```

The live tier is opt-in and read-only apart from preparing the selected
canonical MCAP. It never submits a workflow or provisions resources. Pass the
HTTPS URL and basic auth only through the process environment; do not put them
in command arguments or committed Cypress files:

```bash
NPA_AGENT_CYPRESS_LIVE=1 \
NPA_AGENT_BASE_URL=https://agent.example \
NPA_AGENT_USER='<runtime-user>' \
NPA_AGENT_PASSWORD='<runtime-password>' \
NPA_AGENT_CYPRESS_FOXGLOVE_RUN_ID='<real-run-id>' \
bash npa/scripts/run_agent_cypress.sh --live
```

Live mode fails closed unless the opt-in and all three access variables are
present, requires HTTPS, accepts the agent's self-signed certificate, disables
screenshots and video, and does not print credential values. It proves the SDK
backend/iframe contract, selected remote-file destination, popup-safe open,
CORS preflight, byte Range response, and MCAP magic without reading pixels from
the cross-origin hosted viewer.

References: [Foxglove shareable links](https://docs.foxglove.dev/docs/visualization/shareable-links),
[Foxglove layouts](https://docs.foxglove.dev/docs/visualization/layouts),
[Foxglove panels](https://docs.foxglove.dev/docs/visualization/panels),
[Foxglove importing data](https://docs.foxglove.dev/docs/data/importing-data),
[Foxglove API](https://docs.foxglove.dev/api), and
[Foxglove pricing](https://foxglove.dev/pricing).
