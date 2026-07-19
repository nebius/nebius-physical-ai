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

| NVIDIA stage | NPA state | Tool (all REAL — no stubs) | Runtime |
| --- | --- | --- | --- |
| Config Generation | `generate-configs` | `data_factory_stages.generate_configs` (run.shell) | CPU |
| Understand & Annotate | `annotate-original` | `workbench.token_factory.caption` | Token Factory (zero-GPU) |
| Augment & Multiply | `augment` | `workbench.cosmos2.transfer_execute` (real Cosmos Transfer 2.5 `--execute`; uploads video+frames to S3) | GPU |
| Evaluate & Validate | `grade` loop (`attribute-verify` + `quality-gate`) | `workbench.vlm_eval.run` + `data_factory_stages.grade_gate` (reads the real VLM score) | Token Factory + CPU |
| Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Curation | `curate` | `data_factory_stages.curate` (real dataset report) | CPU |
| Visualize | `visualize` | `data_factory_viz.build_run_rrd` → `reports/sim2real.rrd` | CPU |
| Finalize | `finalize` | `data_factory_stages.finalize` (real aggregate report) | CPU |

Every stage invokes a real component (enforced by `test_real_components.py` and
the `real-components` skill). The `augment` stage runs the real Cosmos Transfer
2.5 model on GPU via `--execute` and publishes the generated video + extracted
frames to `augment_uri`, which the grade / re-label / visualize stages consume.

**Config → augment scope (honest caveat).** `generate-configs` samples
appearance combos (weather / time-of-day / road-condition); the `augment`
toolRef passes `--configs-uri` so the first sampled combo is recorded as the
clip's `metadata.json` `variables` (which drives the Rerun label and proves the
config manifest is consumed, not decorative). Cosmos Transfer 2.5 itself still
runs a **fixed control spec** (bundled `robot_depth_spec.json`), so the geometry
of the re-render is not yet conditioned on the sampled weather/time text, and a
single `--execute` emits **one** variant. Full config-driven appearance
conditioning and N-variant "multiply" (one inference per sampled combo) are
tracked follow-ups — do not describe the current blueprint as generating N
condition-specific variants. The single-variant limitation is also surfaced in
the machine-readable artifacts (`multiply` in the curation report, `multiply_mode`
in the finalize report), so the outputs are honest on their own, not only in docs.

**Naming caveat:** the `attribute-verify` stage runs the REAL `vlm_eval` tool
with `--backend api`; its output file is `vlm_eval_stub.json`, a LEGACY filename
of the vlm_eval tool (`RESULT_FILENAME`), not a stubbed stage. `grade_gate`
imports that constant instead of hardcoding the string, so it stays in sync.

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
- **Augment output contract (per-clip layout):** `cosmos2.transfer` writes a
  contract manifest by default; with `--execute` on S3 output,
  `publish_transfer_to_s3` uploads the real Cosmos Transfer 2.5 result in the
  **per-clip** layout the consumers require:
  `cosmos_augmented/<clip>/{augmented_video.mp4, frame-*.png, metadata.json}`
  plus a run-level `cosmos_augmented/manifest.json`. `curate` counts clip
  subdirs (not top-level files) and `build_run_rrd` reads each clip's
  `metadata.json` for its Rerun label. Producer and consumers share this shape;
  `test_publish_transfer_layout_interoperates_with_curate_and_viz` guards it. A
  single `--execute` emits one clip dir; N-variant "multiply" (one dir per
  sampled augmentation) needs one inference per combo (follow-up).
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
