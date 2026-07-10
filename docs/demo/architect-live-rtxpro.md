# Live Workbench Demo (Architect Pack)

Operator runbook for reproducing the multi-tool Physical AI demo on live Nebius
infrastructure for project alias `rtxpro`.

Canonical product flow: [8gpu-h200.md](./8gpu-h200.md).

Presentation on a **dedicated** agent UI (not a reused agent):
[architect-live-agent-ui.md](./architect-live-agent-ui.md).

## What this demo provisions

| Role | GPU | Runtime | Tools |
|------|-----|---------|-------|
| Shared inference host | 8× H200 (`gpu-h200-sxm` / `8gpu-128vcpu-1600gb`) | managed VM + BYOVM siblings | Cosmos `:8081`, GR00T `:8082`, FiftyOne `:5151` |
| Simulation host | 1× RTX PRO 6000 (`gpu-rtx6000` / `1gpu-24vcpu-218gb`) | managed VM | Isaac Lab (RT cores) |

Artifacts land under `s3://$BUCKET/demo-8gpu-h200/<run-tag>/`.

This project does not expose L40S platforms; Isaac Lab is routed to RTX PRO 6000
Blackwell instead of the historical L40S host from the original May demo.

## Prerequisites

```bash
export NPA_NEBIUS_PROFILE=npa-mk8s
export PROJECT_ALIAS=rtxpro
export REGION=us-central1
# Set from your Nebius console / ~/.npa config — never commit these values:
export PROJECT_ID=...
export TENANT_ID=...
export BUCKET=...
export NPA_SSH_KEY=~/.ssh/id_ed25519
```

## Runner

```bash
cd ~/nebius-physical-ai
./scripts/run_workbench_demo_live.sh
```

The runner provisions sequentially (Cosmos, then Isaac) to reduce
`compute.disk.count` quota races. If disk quota is exhausted, delete orphan
disks / unused instances before retrying.

## Known gaps from live validation

| Area | Notes |
|------|-------|
| Managed Isaac on `gpu-rtx6000` | Some image families need driver ≥580; prefer a validated Ubuntu/CUDA family |
| Cosmos `infer` | Serve/status can be healthy while text-to-world hits `_execution_device` |
| FiftyOne | Registry deny for `npa-fiftyone` until IAM docker login succeeds |
| Historical `demo-prestage` | May be absent; regenerate/stage media instead of `npa demo stage` |

## Presentation

Use [architect-live-agent-ui.md](./architect-live-agent-ui.md) with run id
`demo-workbench-ui` and absolute HTTPS Rerun recording URLs.
