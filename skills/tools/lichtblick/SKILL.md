---
name: foxglove
description: Use when deploying, launching, or reviewing the Foxglove (Lichtblick OSS) web viewer for MCAP, ROS-bag, and robotics log artifacts staged from S3.
---

# Foxglove (Lichtblick)

Foxglove is the browser-based MCAP / ROS-bag / robotics log viewer. It is the
direct analog of `rerun-viewer`: a static web viewer that opens artifacts, with
no GPU. The image ships **Lichtblick** (`lichtblick-suite/lichtblick`), the
actively maintained **MPL-2.0** OSS fork of the archived, relicensed Foxglove
Studio. No Foxglove account or proprietary component is required.

- Tool name: `foxglove` — Image: `npa-foxglove` — Port: `8080` — Tier: `service`.
- Pinned OSS version: Lichtblick `1.26.0` (`[tool.npa.supported-tools]`).
- CPU-only; not part of the `cuda12` / `cuda13-b300` tag families.

## Interfaces

CLI:

```bash
npa workbench foxglove serve --input-path s3://bucket/run/recording.mcap
npa workbench foxglove launch --input-path s3://bucket/run/recording.mcap
npa workbench foxglove status
npa workbench foxglove list
```

SDK: `npa.sdk.workbench.foxglove` (`serve`, `launch`, `status`, `list`).

Container: `caddy file-server` serves the static Lichtblick bundle on `:8080`
(default CMD), with a `HEALTHCHECK` on `/`. Final `USER nobody`.

## Deploy / launch contract

- Cross-tool data flows through S3 only. `serve`/`launch` take `--input-path`
  (S3 or local MCAP/bag/db3) and optional `--output-path`; the viewer opens the
  staged artifact via a deep-linked `?ds=remote-file&ds.url=...` URL.
- `--host` / `--port` control the bind (default `0.0.0.0:8080`).
- The CLI resolves the `npa-foxglove` image via
  `npa.deploy.images.container_image_for_tool` (registry from
  `resolve_container_registry`; never hardcode registry IDs) and emits the
  container run command + viewer URL. Container launch itself is performed by the
  deploy/workflow path, mirroring how `rerun-viewer` is workflow-launched.

## Source of truth

Launch logic lives in `npa/src/npa/workbench/foxglove/`; the CLI and SDK call
into it. The Dockerfile is `npa/docker/workbench/foxglove/Dockerfile` (multi-stage
`node:22-bookworm` build of the pinned OSS tag → `caddy:2.11.4-alpine` runtime,
both digest-pinned). Golden eval and safety posture: `foxglove` in
`npa/src/npa/smoke/golden_evals.yaml` (kind `build-import`, `gpu: none`).
