---
name: inference
description: Use when running or modifying Cosmos3 inference through NPA, especially the public text-to-image SkyPilot workflow, guardrails behavior, prompt/input handling, or upstream Cosmos inference arguments.
---

# Cosmos3 Inference

## Source And Attribution

Adapted from NVIDIA cosmos-framework
`.agents/skills/cosmos3-inference/SKILL.md`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Used under OpenMDW-1.1.
See `.agents/skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1` and
`.agents/skills/NOTICE-NVIDIA-COSMOS3`.

## When To Use

Use this skill when the user wants to generate an image or video with Cosmos3,
change inference defaults, verify prompt handling, inspect guardrails behavior,
or connect NPA's Cosmos3 workflow to upstream Cosmos framework inference docs.
For environment errors, use `../env-troubleshoot/SKILL.md`.

## Real NPA Workflow

The retained NPA Cosmos3 inference workflow is:

```text
npa/workflows/workbench/skypilot/cosmos3-text-to-image-inference.yaml
```

It is a real H100 text-to-image smoke workflow. It clones the Cosmos framework,
downloads the configured Hugging Face model, creates a text-to-image JSON input,
runs `python -m cosmos_framework.scripts.inference`, validates the produced
image, and optionally uploads the image plus success JSON to S3.

Do not replace this with a Cosmos skill-display subcommand; Cosmos3 skills are
SKILL.md files for agents, not commands.

## Guardrails

Guardrails are on by default.

In the workflow, `NPA_COSMOS3_NO_GUARDRAILS` defaults to an empty string, and
the inference command expands `--no-guardrails` only when that variable is set.
Only set it when the user explicitly requests an opt-out:

```yaml
NPA_COSMOS3_NO_GUARDRAILS: "1"
```

The agent should preserve this default in CLI, SDK, Docker image, and workflow
changes.

## Running The Workflow

Before launch, confirm credentials and access:

```bash
npa/.venv/bin/npa workbench cosmos check --output json
```

Review or override these environment fields in the SkyPilot YAML:

- `NPA_COSMOS3_SOURCE_REPO`
- `NPA_COSMOS3_MODEL_ID`
- `NPA_COSMOS3_CACHE`
- `NPA_COSMOS3_HF_TOKEN_ENV`
- `NPA_COSMOS3_INFER_PROMPT`
- `NPA_COSMOS3_OUTPUT_DIR`
- `NPA_COSMOS3_OUTPUT_IMAGE`
- `NPA_COSMOS3_SUCCESS_JSON`
- `NPA_COSMOS3_OUTPUT_S3_URI`
- `NPA_COSMOS3_NO_GUARDRAILS`

The workflow uses node-local temporary paths by default. Do not write model
checkpoints or generated outputs into the repository.

## Upstream Inference Map

In a clone of `https://github.com/NVIDIA/cosmos-framework.git`, inspect:

| Need | Upstream path |
| --- | --- |
| Batch inference script | `cosmos_framework/scripts/inference.py` |
| Sampling args and validation | `cosmos_framework/inference/args.py` |
| Per-modality defaults | `cosmos_framework/inference/defaults/<mode>/sample_args.json` |
| Inference docs | `docs/inference.md` |
| FAQ for overrides, shift, and online serving | `docs/faq.md` |
| Example low-level APIs | `examples/inference.py`, `examples/inference_pipeline.py` |

Path handling follows upstream behavior: relative paths in input JSON files are
resolved relative to the JSON file's directory. Use explicit `--seed` for
reproducible smoke runs.

## Test Expectations

When changing this area, keep tests focused on behavior that does not require a
GPU:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_cosmos3_access.py \
  npa/tests/cli/test_cosmos3_cli.py
```

Expected checks include:

- The inference YAML name is `cosmos3-text-to-image-inference`.
- `image_id` is not hard-coded in the resources.
- The command invokes `python -m cosmos_framework.scripts.inference`.
- `NPA_COSMOS3_NO_GUARDRAILS` defaults to empty.
- `--no-guardrails` is not present by default.
- S3 output remains optional.
