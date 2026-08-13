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
| `POST /api/foxglove/live` | Point the viewer at a public `ws://`/`wss://` Foxglove or ROS-bridge URL |

Configuration (no secrets): `NPA_FOXGLOVE_EMBED_SRC`, `NPA_FOXGLOVE_ORG_SLUG`,
`NPA_FOXGLOVE_LIVE_URL`, `NPA_FOXGLOVE_COLOR_SCHEME`,
`NPA_FOXGLOVE_LAYOUT_STORAGE_KEY`, `NPA_FOXGLOVE_ENABLED`; or
`npa agent bootstrap --foxglove-embed-src <url> --foxglove-org-slug <slug>`.

## CLI

```bash
npa workbench foxglove config --output json
npa workbench foxglove install-sdk --dest /opt/npa-agent/foxglove/sdk
npa workbench foxglove convert-run --input-path <run-dir> --output-path run.mcap --fps 10
npa workbench foxglove inspect --input-path run.mcap
```

`convert-run` packs real artifacts into Foxglove well-known schemas:
`foxglove.CompressedImage` on `/camera/<name>` (PNG/JPEG passed through, PPM/BMP/TIFF
transcoded), `foxglove.Log` on `/log`, and `npa.RunMetrics.<name>` on `/metrics/<name>`.
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
```

The browser spec drives the **real** `@foxglove/embed` build against a
protocol-accurate stand-in for the Foxglove application (the licensed viewer cannot
run in CI): it asserts the handshake, the data source actually sent, error surfacing,
the unconfigured path, lazy loading, and the text-only Describe-this contract.
