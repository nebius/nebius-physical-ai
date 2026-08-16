---
name: foxglove
description: Use when embedding, operating, or debugging the Foxglove viewer in the NPA agent (the @foxglove/embed TypeScript SDK, MCAP recordings, the npa-foxglove-embed container, or /api/foxglove/* endpoints).
---

# Foxglove Embedded Viewer

NPA embeds Foxglove with the official
[`@foxglove/embed` TypeScript SDK](https://docs.foxglove.dev/docs/embed/typescript-sdk):
the SDK creates an iframe pointing at a Foxglove deployment and drives it over
`postMessage` (`setDataSource`, `selectLayout`, `seekPlayback`, `ready`/`error`).

**What NPA ships vs what Foxglove hosts** — NPA serves the MIT-licensed SDK
(fetched unmodified from npm, sha512-verified) plus your recordings. The viewer
*application* is Foxglove's: either `https://embed.foxglove.dev/` (users sign in to
a Foxglove organization on a plan that allows embedding) or your self-hosted
deployment. Without a configured embed source the viewer pane says so — it never
renders an empty frame and calls it a viewer.

## Which viewer when

NPA ships **two** MCAP viewers; they are complements, not alternatives:

| | `skills/tools/lichtblick/SKILL.md` (OSS) | this skill (official Foxglove) |
| --- | --- | --- |
| What runs | Lichtblick web build (MPL-2.0) served by the agent, in-page | Foxglove's own app in a cross-origin iframe, driven by `@foxglove/embed` |
| Account | none | a Foxglove org on a plan that allows embedding |
| Renders MCAP | yes, out of the box | yes, after sign-in |
| Recording path | same-origin `/lichtblick/recordings/` (no CORS) | `/foxglove/data/` (CORS + byte ranges) |
| Use it for | default operator playback, CI, air-gapped | customers standardized on Foxglove (layouts, extensions, org sharing) |

The agent's **Foxglove** pane picks between them at runtime
(`/api/foxglove/config` → `viewer_backend`): the official app when
`NPA_FOXGLOVE_EMBED_SRC` is configured and the SDK assets are installed,
otherwise the self-hosted OSS viewer, otherwise an explained unavailable state.
`NPA_FOXGLOVE_VIEWER_BACKEND=foxglove-sdk|self-hosted` forces one.

## When To Use

- Adding/changing the agent's Foxglove viewer pane or `/api/foxglove/*` endpoints
- Publishing `.mcap` / `.bag` recordings for playback in the agent
- Building/running the `npa-foxglove-embed` container
- Converting run artifacts to MCAP (`npa workbench foxglove convert-run`)
- Debugging "viewer unavailable", CORS, or byte-range playback problems

## Architecture

| Piece | Where |
| --- | --- |
| Pinned SDK version + integrity + asset probe | `npa/src/npa/workbench/foxglove/__init__.py` |
| One install recipe (fetch + verify + extract) | `npa/docker/workbench/foxglove-embed/install-sdk.sh` |
| Shared browser glue (`mountFoxgloveViewer`) | `npa/src/npa/cli/assets/foxglove/npa-foxglove-host.js` |
| Standalone host page | `npa/src/npa/cli/assets/foxglove/index.html` |
| Container (caddy, `:8099`, non-root) | `npa/docker/workbench/foxglove-embed/` |
| Agent backend helpers (shipped module) | `npa/src/npa/agent_backend/foxglove.py` (shim: `cli/agent_foxglove.py`) |
| Agent routes (shipped module) | `npa/src/npa/agent_backend/foxglove_routes.py` |
| nginx serving policy | `npa/src/npa/cli/agent_site.py` |
| Agent UI pane + lazy SDK import | `npa/src/npa/cli/agent_ui.html` (`ensureFoxgloveViewer`) |
| MCAP writer / reader | `npa/src/npa/workbench/foxglove/{mcap_writer,inspect}.py` |
| CLI / SDK | `npa/src/npa/cli/workbench/foxglove.py`, `npa/src/npa/sdk/workbench/foxglove.py` |

On the agent VM, bootstrap installs assets to `/opt/npa-agent/foxglove/`:
`sdk/` (SDK), `app/` (glue), `data/` (published recordings). nginx serves
`/foxglove/` behind basic auth and `/foxglove/data/` **without** auth but with CORS,
`Accept-Ranges`, gzip off, and an `OPTIONS` preflight — the cross-origin viewer
iframe cannot send credentials and streams recordings with Range requests.
Published names are random (`<token>-<stem>.mcap`) and pruned to the newest few.

## Agent endpoints

| Route | Purpose |
| --- | --- |
| `GET /api/foxglove/config` | Everything the UI needs to mount a viewer: `viewer_backend`, `self_hosted_url`, SDK/embed settings, data source; `available:false` + `reason` when neither backend can render |
| `GET /api/foxglove/status` | Readiness + active recording (also grounds the `foxglove_viewer` chat intent) |
| `POST /api/foxglove/load-artifact` | Load a discovered `.mcap`/`.bag`/`.db3`/`.ulg`/`.ulog` artifact (`run_id` + `s3_uri`, or `run_id` + `key`) |
| `POST /api/foxglove/convert-run` | Convert the active run's local artifacts to MCAP and load it |
| `POST /api/foxglove/export` | Authorize and prepare/reuse the exact selected MCAP; return the viewer config plus phase timings, download it, open its official remote-file link, or explicitly upload/index it |
| `POST /api/foxglove/live` | Point the viewer at a public `ws://`/`wss://` Foxglove or ROS-bridge URL |

Configuration (no secrets): `NPA_FOXGLOVE_EMBED_SRC`, `NPA_FOXGLOVE_ORG_SLUG`,
`NPA_FOXGLOVE_LIVE_URL`, `NPA_FOXGLOVE_COLOR_SCHEME`,
`NPA_FOXGLOVE_LAYOUT_STORAGE_KEY`, `NPA_FOXGLOVE_ENABLED`, and the positive,
finite `NPA_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS` (default `300`, maximum
`3600`); or
`npa agent bootstrap --foxglove-embed-src <url> --foxglove-org-slug <slug>`.

## CLI

```bash
npa workbench foxglove config --output json
npa workbench foxglove install-sdk --dest /opt/npa-agent/foxglove/sdk
npa workbench foxglove convert-run --input-path <run-dir> --output-path run.mcap --fps 10
npa workbench foxglove export-run --input-path <run-dir> --output-path run.mcap
npa workbench foxglove open --recording-id <indexed-recording-id>
npa workbench foxglove inspect --input-path run.mcap
```

`open` uses Foxglove's official `foxglove-stream` recording deep-link contract
for an explicitly indexed Cloud recording.
Agent export persists exactly one canonical run artifact at
`<run-prefix>/<run-id>/reports/sim2real.mcap`, with
`sim2real.mcap.provenance.json` beside it. A valid native MCAP is reused;
otherwise real S3 run artifacts are converted and the run-list cache is
invalidated. Lichtblick, the download transport, and Cloud import use identical
canonical bytes and report the same SHA-256.
The artifact card's ordinary **View in Foxglove** action opens the embedded SDK
pane and binds the exact selected MCAP as a `remote-file` source. Before any
no-download reuse, the backend authorizes the immutable run reference and exact
key. Exact artifact cards use a narrow fail-closed authorization check: verify
the selected project is tenant-visible, verify only its selected bucket, and
probe that bucket's current read scope. Do not rebuild the tenant-wide access
report or probe unrelated buckets on the exact artifact-inventory or playback
path. Once discovery has issued this exact source tuple, browser verification
must reuse it instead of repeating tenant-wide run search. A successfully
rendered exact artifact card refreshes a 30-second, full-source-keyed access proof
and exact inventory for its immediate playback click; explicit access refresh
clears access proofs. Rendering the cards and binding their actions precedes
slower run-detail enrichment, so playback never waits behind that optional UI
work.
After the 30-second inventory window expires, exact playback must re-probe only
the authorized `prefix/run-id/` scope and list that exact run. It must never fall
back to rebuilding the bucket-wide run index for a source-qualified card.
The backend then reads a fresh strong object-store identity (ETag or version id)
and verifies the published
bytes against the persisted SHA-256 and provenance. An unchanged selection skips
download, conversion, and publication; a changed object identity or mismatched
local byte invalidates the cache. A canonical cache miss prepares and applies the
local result once rather than downloading it again. The export response includes
the matching viewer config, `cache_reused`, and phase timings so the UI does not
need a redundant config request.

The SDK iframe mounts before backend preparation finishes and remains mounted
across exact-card selections. The UI sends `setDataSource` and `selectLayout`
only when their identities change. Status, artifact actions, and visualization
summary participate in normal layout above the viewer canvas; they must never be
absolutely or fixed-positioned over playback controls. Success text stays compact
while its full accessible value remains available through the status element.

The separate **Open in Foxglove** action uses Foxglove's documented
`ds=remote-file&ds.url=<public HTTPS MCAP>` link. The recording URL is encoded
exactly once, contains no credentials, and is the same CORS + byte-range
transport used by the official embed SDK. The button synchronously reserves a
popup during the user gesture and reports blocked or failed navigation honestly.

For recordings that advertise `npa.foxglove.robot-motion.v3`, the server
idempotently creates the versioned `NPA Physical AI robot motion v3` organization
layout and adds its non-secret `layoutId` to the hosted link. The layout seeds a
large 3D robot/trajectory panel, discovered source-camera tabs, metrics,
phase/state, and logs. Existing
versioned layouts and SDK `storageKey` arrangements are reused without forcing or
overwriting user changes. If the layout API is unavailable, the link still opens
the rich topics and the UI says that a saved layout must be selected after sign-in.

An explicit backend `cloud_import` mode can additionally upload the MCAP once
under a stable content key, reuse unchanged or in-progress imports, and wait for
indexed `complete` state.
A server-side API token is required for shared-layout creation and the optional
Cloud import at
`tokens.FOXGLOVE_API_TOKEN` in `~/.npa/credentials.yaml` (mode `0600`). It is
never part of browser config, deep links, subprocess arguments, shared
workbench env, or the agent's `foxglove.env`. If it is already exported in the
operator shell, persist it with
`npa configure --no-interactive --save-env-credentials`; never pass its value as
an argument.

`convert-run` packs real artifacts into Foxglove well-known schemas:
`foxglove.CompressedImage` on `/camera/<name>` (PNG/JPEG passed through, PPM/BMP/TIFF
transcoded), `foxglove.Log` on `/log`, and `npa.RunMetrics.<name>` on `/metrics/<name>`.
When a real `npa.sim2real.action_rollout.v1` artifact is present, it also emits
the `npa.foxglove.robot-motion.v3` contract: an explicitly labelled action-derived
diagnostic robot (`foxglove.SceneUpdate`), end-effector pose and cumulative
trajectory (`PoseInFrame` / `PosesInFrame`), `foxglove.JointStates`, actuator
commands, run phase/progress, camera transforms, metrics, and logs on one coherent
clock. This schematic must remain labelled as uncalibrated diagnostic kinematics;
camera frames and copied simulator-ground-truth fields retain their source fidelity.
An explicitly declared `npa.foxglove.pointcloud-series.v1` artifact becomes
`foxglove.PointCloud` on `/trajectory` plus its real frame relationship as
`foxglove.FrameTransform` on `/tf`; its provenance must describe the coordinate
semantics and must not imply world geometry when the points represent state space.
Run artifacts carry no capture time, so frame timestamps come from `--fps` and are
recorded as `timestamps=synthetic-fps` in the MCAP metadata — never present them as
sensor time. Needs the optional extra: `pip install "npa[foxglove]"`.

## Container

```bash
docker build -t npa-foxglove-embed:0.58.0 -f npa/docker/workbench/foxglove-embed/Dockerfile npa
docker run --rm -p 8099:8099 -v $PWD/recordings:/srv/data:ro npa-foxglove-embed:0.58.0
# http://localhost:8099/?src=https://embed.foxglove.dev/&org=<slug>&mcap=/data/run.mcap
docker exec <container> sh /usr/local/bin/npa-foxglove-smoke.sh   # golden eval
```

Serves `/sdk`, `/app`, `/data` (CORS + ranges + listing), `/` (host page), `/healthz`.
It has no authentication of its own: keep it cluster-internal or behind an auth proxy.

## Gotchas

- **Absolute URLs only.** `new FoxgloveViewer({src})` throws on a relative URL, and
  data-source URLs are fetched by the cross-origin iframe — always absolutize.
- **Secure context.** Foxglove requires HTTPS (or `localhost`).
- **Ranges break under compression.** Never gzip `/foxglove/data/` or `/data/`.
- **Range triggers a CORS preflight.** `OPTIONS` must answer with
  `Access-Control-Allow-Headers: Range`.
- **No pixel capture.** The embed is cross-origin, so "Describe this" sends viewer
  *state* (source type, recording URL, run/artifact ids) and says so. Do not add a
  screenshot path for this pane.
- **Lazy load.** The SDK is imported only when the Foxglove tab is opened; keep it
  that way so the Rerun-first boot path stays fast.
- **`ds.url` must be absolute.** The self-hosted viewer's `remote-file` source
  silently ignores a relative URL (no range request, "No data source"), so always
  pin it onto the browsed origin.
- **Hosted remote-file URLs require public HTTPS.** Refuse relative, HTTP,
  credential-bearing, loopback, private, link-local, reserved, or metadata
  targets. Foxglove Web fetches this exact URL, so its certificate, CORS preflight,
  and byte-range behavior must work from a clean browser.
- **An API token is not browser sign-in.** The token remains server-side for the
  layout/recording APIs. The cross-origin hosted or embedded app can still require
  an interactive Foxglove sign-in and an eligible plan; report that surface rather
  than calling an iframe or handshake useful rendering proof.
- **No implicit hosted app.** An unset `NPA_FOXGLOVE_EMBED_SRC` means "no official
  app", not `embed.foxglove.dev` — otherwise a stock deploy shows a sign-in wall
  instead of rendering.
- Bump the SDK by editing `FOXGLOVE_EMBED_SDK_VERSION` + `FOXGLOVE_EMBED_SDK_INTEGRITY`
  (npm `dist.integrity`) and the Dockerfile ARGs together —
  `npa/tests/docker/test_foxglove_image.py` fails if they drift.

## Verify

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/cli/test_agent_foxglove.py npa/tests/cli/test_foxglove_cli.py \
  npa/tests/workbench/test_foxglove_mcap.py npa/tests/docker/test_foxglove_image.py \
  npa/tests/cli/test_agent_backend_render.py npa/tests/smoke/test_agent_smoke.py -q
bash npa/scripts/run_agent_cypress.sh --mock     # includes agent_foxglove.cy.js
# Explicit live opt-in; URL/user/password are runtime environment variables.
NPA_AGENT_CYPRESS_LIVE=1 bash npa/scripts/run_agent_cypress.sh --live
```

The browser spec drives the **real** `@foxglove/embed` build against a
protocol-accurate stand-in for the Foxglove application (the licensed viewer cannot
run in CI): it asserts the handshake, the data source actually sent, error surfacing,
the unconfigured path, lazy loading, and the text-only Describe-this contract.
