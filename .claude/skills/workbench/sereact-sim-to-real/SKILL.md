---
name: sereact-sim-to-real
description: Use when working on Sereact sim-to-real pipeline data import, Cosmos autoscaling, VLM eval stubs, or the sim-to-real controller-loop SkyPilot workflow.
---

# Sereact Sim-To-Real

Sereact sim-to-real is a Workbench pipeline pattern that stages raw S3 data,
runs parallel Cosmos candidate generation, and gates each iteration with the
stub VLM eval backend.

## Interfaces

CLI:

```bash
npa workbench data sync
npa workbench data status
npa workbench data list
npa workbench cosmos autoscale
npa workbench vlm-eval run
npa workbench vlm-eval status
npa workbench vlm-eval list
```

Workflow:

```bash
npa/scripts/run_sim_to_real_loop.py --render-only
npa workbench workflow submit npa/workflows/workbench/skypilot/sim-to-real-loop.yaml
```

## Data Flow

Use S3 URIs for every handoff. The controller loop writes iteration-scoped
outputs under:

```text
s3://${NPA_S3_BUCKET}/sereact-sim-to-real/<run-id>/iter-XX/
```

The data bridge supports scoped project credentials and only falls back to host
credentials when the operator passes `--allow-host-creds`.

## Cosmos Autoscale

`npa workbench cosmos autoscale` is for saved serverless Cosmos endpoint
aliases. Use it to set replica bounds before running a parallel candidate
generation loop.

## Validation Scope

The VLM eval backend is intentionally a deterministic stub. Do not add a real
VLM backend, Lightwheel integration, native SkyPilot loop syntax, or ONNX export
as part of this pipeline surface unless a later task explicitly asks for it.
