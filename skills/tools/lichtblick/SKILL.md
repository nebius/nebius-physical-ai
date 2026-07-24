---
name: lichtblick
description: Use when deploying, launching, or reviewing the Lichtblick web viewer — an open-source, Foxglove-compatible MCAP / ROS-bag / robotics log viewer served from S3 artifacts.
---

# Lichtblick

Lichtblick is an **open-source (MPL-2.0), Foxglove-compatible** MCAP / ROS-bag /
robotics log viewer. It is the browser-based analog of `rerun-viewer`: a static
web viewer that opens artifacts, with no GPU. The image ships
**Lichtblick** (`lichtblick-suite/lichtblick`), the actively maintained community
fork of the archived, relicensed Foxglove Studio. It is a distinct product from
the now-proprietary Foxglove; no Foxglove account or proprietary component is
required.

- Tool name: `lichtblick` — Image: `npa-lichtblick` — Port: `8080` — Tier: `service`.
- Pinned OSS version: Lichtblick `1.26.0` (`[tool.npa.supported-tools]`).
- CPU-only; not part of the `cuda12` / `cuda13-b300` tag families.

## Interfaces

CLI:

```bash
# View an existing MCAP (plan only by default; --execute stages + launches):
npa workbench lichtblick serve --input-path s3://bucket/run/recording.mcap --execute

# Pack a robot camera-frame sequence (e.g. sim2real rollout/augment frames) into
# a real MCAP of foxglove.CompressedImage messages, then serve it:
npa workbench lichtblick serve \
  --input-path s3://bucket/sim2real-b/<run-id>/augment/frames/ \
  --from-frames --fps 10 --topic /sim2real/augment/camera --execute

npa workbench lichtblick launch  # alias for serve
npa workbench lichtblick status
npa workbench lichtblick list
```

SDK: `npa.sdk.workbench.lichtblick` (`serve`, `launch`, `status`, `list`).

Container: `caddy file-server` serves the static Lichtblick bundle on `:8080`
(default CMD), with a `HEALTHCHECK` on `/`. Final `USER nobody`; Caddy's XDG
data/config dirs are owned by that user so the server starts cleanly.

## What it does (tangible)

Lichtblick consumes real Physical AI workflow artifacts from S3 (mirroring how
`rerun-viewer` consumes `sim2real.rrd` — a separate viewer, not embedded in the
pipeline):

- **MCAP export** (`--from-frames`): `build_mcap_from_frames` turns the Sim2Real
  pipeline's `rollouts/.../camera` and `augment/frames` image artifacts into a
  real MCAP of `foxglove.CompressedImage` messages at a chosen `--fps`, so the
  robot camera stream plays on a timeline. PNG/JPEG frames are packed byte-for-byte;
  raw `.ppm` rollout dumps (and other PIL-readable formats) are transcoded to PNG
  first (via `encode_frame_to_compressed_bytes`), so genuine rollout cameras render
  instead of being silently skipped. Verified end-to-end: 32 Cosmos-Transfer2.5
  augment frames → a 9.5 MB MCAP → rendered in a headless browser on topic
  `/sim2real/augment/camera` (200 + 206 range fetches, 0 console errors).
- **Native Sim2Real MCAP** (finalize stage): the Sim2Real viz/finalize stage
  (`sim2real_viz.emit_sim2real_mcap`, gated by `NPA_SIM2REAL_MCAP`, default on when
  rerun is on) natively emits `reports/sim2real.mcap` alongside `reports/sim2real.rrd`
  from the same rollout data — camera frames as `foxglove.CompressedImage`, VLM
  critiques as `foxglove.Log`, and reward/advantage/score signals as numeric samples
  a Plot panel can chart. Open it with
  `npa workbench lichtblick serve --input-path s3://bucket/sim2real-b/<run-id>/reports/sim2real.mcap --execute`.
- **Staging + launch** (`--execute`): `serve_viewer` stages the artifact from S3
  (`stage_input_to_mcap`) and runs the `npa-lichtblick` container so the log is
  live at the returned URL. Without `--execute` it prints the plan (infra-free).

## Deploy / launch contract

- Cross-tool data flows through S3 only. `serve`/`launch` take `--input-path`
  (S3 or local MCAP/bag/db3, or a camera-frames prefix with `--from-frames`) and
  optional `--output-path`; the artifact is **staged into the viewer's own origin**
  (`/srv/data/<name>`, served by the same Caddy on `:8080`) and opened via a
  deep-linked `?ds=remote-file&ds.url=...` URL.
- Because the MCAP is co-served from the viewer origin, the browser fetch is
  same-origin: **no bucket CORS, no pre-signed URL, and no http/https
  mixed-content block** are involved. (Pointing `ds.url` directly at an S3 URL
  instead would require bucket CORS + a presigned URL + an https viewer — that is
  deliberately not this tool's path.) Caddy serves the MCAP with
  `Accept-Ranges: bytes`, so Lichtblick streams it via HTTP range requests.
- The deep link always targets the app root `/` (data source in the query
  string), never a client-routed sub-path, so `caddy file-server` needs no SPA
  fallback: `GET /` always returns `index.html`.
- `--host` / `--port` control the bind (default `0.0.0.0:8080`).
- The CLI resolves the `npa-lichtblick` image via
  `npa.deploy.images.container_image_for_tool` (registry from
  `resolve_container_registry`; never hardcode registry IDs) and emits the
  container run command + viewer URL. Container launch itself is performed by the
  deploy/workflow path, mirroring how `rerun-viewer` is workflow-launched.

## Source of truth

Launch logic lives in `npa/src/npa/workbench/lichtblick/`; the CLI and SDK call
into it. The Dockerfile is `npa/docker/workbench/lichtblick/Dockerfile`
(multi-stage `node:22-bookworm` build of the pinned OSS tag → `caddy:2.11.4-alpine`
runtime, both digest-pinned). Golden eval and safety posture: `lichtblick` in
`npa/src/npa/smoke/golden_evals.yaml` (kind `build-import`, `gpu: none`).
