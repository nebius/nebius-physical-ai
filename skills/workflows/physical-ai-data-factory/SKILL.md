---
name: physical-ai-data-factory
description: Use when authoring, running, submitting, or viewing the NVIDIA Physical AI Data Factory blueprint on Nebius + SkyPilot (no OSMO) — annotate → Cosmos Transfer augment → evaluate/validate gate → re-label → FiftyOne curate → Rerun visualize — implemented as an npa.workflow that composes existing workbench tools.
---

# Physical AI Data Factory (NPA-native, no OSMO)

## Source And Attribution

NPA-native re-implementation of the NVIDIA Physical AI Data Factory / Video Data
Augmentation workflow. Design adapted from NVIDIA agent skills
(https://github.com/NVIDIA/skills), primarily `physical-ai-video-data-augmentation`.
Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. Upstream licenses:
Apache-2.0 and CC-BY-4.0. See `skills/NOTICE-NVIDIA-SKILLS`. NPA orchestrates on
SkyPilot (not OSMO) and composes existing workbench tools.

## When To Use

Load this skill when the user wants to author, validate, submit, run, or view the
`physical-ai-data-factory.yaml` blueprint, adapt it to a new dataset, run it on
GPUs, or troubleshoot why a run's Rerun panel / augmented output looks wrong.

Do NOT invent an `npa workbench data-factory` tool — there is none. The blueprint
is pure composition of existing toolRefs; only add real tools with tests.

## What It Is

`npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml` — one
`npa.workflow/v0.0.1` spec. Blueprint → NPA stage mapping:

| NVIDIA stage | NPA state | Tool | Runtime |
| --- | --- | --- | --- |
| Config Generation | `generate-configs` | `run.shell` (sample appearance-only vars) | CPU |
| Understand & Annotate | `annotate-original` | `workbench.token_factory.caption` | Token Factory (zero-GPU) |
| Augment & Multiply | `augment` | `workbench.cosmos2.transfer` | GPU (Cosmos Transfer 2.5) |
| Evaluate & Validate | `grade` loop (`attribute-verify` + `quality-gate`) | `workbench.vlm_eval.run` + `workbench.sim2real.write_decision` | Token Factory + CPU |
| Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Curation | `curate` | `workbench.fiftyone.launch_app` | CPU |
| Visualize | `visualize` | `npa.workflows.data_factory_viz.build_run_rrd` (run.shell) → `reports/sim2real.rrd` | CPU |
| Finalize | `finalize` | `workbench.sim2real.finalize` | CPU |

Verified Token Factory model roles: `Qwen/Qwen2.5-VL-72B-Instruct` (VLM),
`meta-llama/Llama-3.3-70B-Instruct` (LLM), `nvidia/Cosmos3-Super-Reasoner`
(Cosmos-family critic). Cosmos Transfer 2.5 is the GPU augment engine, not a
Token Factory model.

## Commands

```bash
SPEC=npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec   "$SPEC" --run-id demo --assume-decision promote_checkpoint --json
# Render/submit on GPUs (needs NPA_SRC_S3_URI or --image, and secret-envs):
npa workbench workflow submit "$SPEC" --run-id "$(date -u +paidf-%Y%m%dt%H%M%sz)" \
  --assume-decision promote_checkpoint --var bucket=<bucket> \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN
```

## Key Operational Notes

- **GPU accelerator name is cluster-specific.** The spec uses canonical
  `RTXPRO6000:1`; some clusters advertise `RTXPRO-6000-BLACKWELL-SERVER-EDITION`.
  If `sky` reports `FAILED_PRECHECKS` / no matching resources, check
  `sky gpus list` and resubmit with the cluster's accelerator name.
- **Reproducible deploy/submit:** unset a stale `NEBIUS_IAM_TOKEN` before
  `sky`/`terraform` (the Nebius provider prefers the ambient token over the
  fresh one). `provisioner._run` scrubs it for agent deploys.
- **Augment output contract:** `cosmos2.transfer` writes a contract manifest by
  default; the downstream `grade` / `annotate-augmented` stages need augmented
  **frames** under `augment_uri`. Real photoreal augmentation needs Cosmos
  Transfer 2.5 `--execute` output wired to S3 (roadmap); a CPU appearance
  transform can stand in for viewable frames without a GPU.
- **Viewing in the NPA agent:** every stage lands under one S3 run prefix
  (`input/ configs/ labeled_original/ cosmos_augmented/ grade/ labeled_augmented/
  curation/ reports/`). The `visualize` stage writes `reports/sim2real.rrd`,
  which the agent's embedded Rerun viewer renders. Browse via
  `/api/artifacts/runs?prefix=physical-ai-data-factory`. See
  `docs/workbench/guides/physical-ai-data-factory.md`.

## Testing (live-infra is a priority)

Follow `skills/atomic/testing-conventions/SKILL.md` ("Live-Infra Testing Is A
Priority"). The blueprint is registered in `SUBMIT_LIVE_MATRIX` and
`DYNAMIC_SPECS`; smoke via `test_all_workflow_yamls.py`; the Rerun builder via
`npa/tests/workflows/test_data_factory_viz.py`.
