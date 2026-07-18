---
name: workbench-tool
description: Use when adding, changing, deploying, or calling any NPA workbench tool; captures the API/CLI/SDK/container architecture and data-flow contract.
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

Exception / gotcha: a few tools historically use `--input-uri` / `--output-uri`
instead (e.g. `npa workbench cosmos2 transfer`, `cosmos3 reason`). When you wire a
tool into an npa.workflow `toolRef` (`npa/src/npa/orchestration/npa_workflow/catalog.py`),
the argv template MUST match that tool's **actual** CLI option names and include
required flags (e.g. `--run-id`). A mismatch passes `validate-spec`/`plan-spec`
but crashes on real submit with an unknown-option error. Verify against the CLI
signature, and keep `catalog.py` and `docs/workbench/npa-workflow-tool-catalog.md`
in sync. Prefer standardizing new tools on `--input-path`/`--output-path`.
