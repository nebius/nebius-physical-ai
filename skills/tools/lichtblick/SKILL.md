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
npa workbench lichtblick serve --input-path s3://bucket/run/recording.mcap
npa workbench lichtblick launch --input-path s3://bucket/run/recording.mcap
npa workbench lichtblick status
npa workbench lichtblick list
```

SDK: `npa.sdk.workbench.lichtblick` (`serve`, `launch`, `status`, `list`).

Container: `caddy file-server` serves the static Lichtblick bundle on `:8080`
(default CMD), with a `HEALTHCHECK` on `/`. Final `USER nobody`; Caddy's XDG
data/config dirs are owned by that user so the server starts cleanly.

## Deploy / launch contract

- Cross-tool data flows through S3 only. `serve`/`launch` take `--input-path`
  (S3 or local MCAP/bag/db3) and optional `--output-path`; the viewer opens the
  staged artifact via a deep-linked `?ds=remote-file&ds.url=...` URL.
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
