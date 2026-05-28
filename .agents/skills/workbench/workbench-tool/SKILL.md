---
name: workbench-tool
description: Use when adding, changing, deploying, or calling any NPA workbench tool; captures the API/CLI/SDK/container architecture and data-flow contract.
last_verified: 2026-05-26
owner: workbench
version: 1.0.0
---

# Workbench Tool Pattern

Every workbench tool is a containerized FastAPI service. The container is the unit of deployment; the service endpoint is the unit of invocation; the CLI and SDK are clients.

Each capability must have one source of truth. Put behavior in the service or shared implementation layer, then have all access paths call it. Never duplicate training, inference, import, or status logic separately across API, CLI, and SDK layers.

## Three Access Modes

- API: HTTP endpoints exposed by the FastAPI service.
- CLI: `npa workbench <tool> ...`.
- SDK: `npa.sdk.workbench.<tool>`.

## Standard Endpoints

Workbench services should expose these standard surfaces unless a tool-specific skill documents an exception:

- `GET /health`
- `POST /train` or `POST /run`
- `GET /status`
- `GET /system-info`
- `GET /list`

## Deployment

Always pass `--storage-endpoint storage.eu-north1.nebius.cloud` when deploying or configuring workbench tools. The CLI default `storage.uk-south1.nebius.cloud` is wrong for the primary cluster.

Kubernetes namespace split:

- `workbench`: deployed workbench services.
- `default`: SkyPilot task pods.

## Cross-Tool Data Flow

Tools communicate through S3, never by directly calling each other for data transfer. All tool commands must support `--input-path` and `--output-path` so pipelines can pass S3 URIs across stages.

## Changelog

- 2026-05-26: Added frontmatter metadata (last_verified, owner, version) and Changelog section per skill-authoring.
