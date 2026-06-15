---
name: cosmos3-setup
description: Use when setting up Cosmos3 access through NPA, checking source or Hugging Face reachability, staging the public Cosmos3 framework and checkpoint cache, or deciding which NPA workflow to use before inference.
---

# Cosmos3 Setup

## Source And Attribution

Adapted from NVIDIA cosmos-framework
`skills/atomic/cosmos3-setup/SKILL.md`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Used under OpenMDW-1.1.
See `skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1` and
`skills/NOTICE-NVIDIA-COSMOS3`.

## When To Use

Use this skill when the task is about Cosmos3 first-run setup, source checkout,
checkpoint access, credential verification, CUDA or uv install choices, or
selecting the right NPA Cosmos3 workflow. If the user is already failing with an
ImportError, CUDA error, Docker runtime error, or checkpoint download failure,
switch to `skills/atomic/cosmos3-env-troubleshoot/SKILL.md`.

## NPA Boundary

Cosmos3 skills are agent skills. Do not add or invoke Cosmos skill-display
subcommands; those are intentionally not part of the NPA human CLI surface.

Real NPA setup surfaces are:

- `npa workbench cosmos check` for redacted source, Hugging Face, optional NGC,
  and parser-setting validation.
- `npa workbench cosmos fetch` for source plus checkpoint staging in a temporary
  runtime cache.
- `npa/workflows/workbench/skypilot/cosmos3-ea-fetch.yaml` for a SkyPilot fetch
  workflow that runs those real commands.
- `npa/workflows/workbench/skypilot/cosmos3-text-to-image-inference.yaml` for
  the H100 text-to-image smoke workflow.

## Setup Flow

1. Inspect the NPA integration before changing behavior:

   ```bash
   rg -n "Cosmos3|cosmos3|NPA_COSMOS3" npa/src/npa npa/workflows npa/tests
   ```

2. Confirm access locally without downloading weights:

   ```bash
   npa/.venv/bin/npa workbench cosmos check --output json
   ```

   This command must redact model repo IDs, private source URLs, and token
   values in its output.

3. Stage source and checkpoint only when credentials are available:

   ```bash
   npa/.venv/bin/npa workbench cosmos fetch --output json
   ```

   Use `--skip-checkpoint` for source-only checks. Use `--cache-dir` for an
   ephemeral path, not a repository path.

4. For GPU inference, use the text-to-image SkyPilot workflow. The setup skill
   should point to that workflow; it should not create a skill launcher command.

5. Keep guardrails on by default. The only opt-out is an explicit user request
   mapped to `NPA_COSMOS3_NO_GUARDRAILS=1` in the inference workflow.

## Credential Inputs

Defaults are controlled by these environment variable names:

- `NPA_COSMOS3_SOURCE_REPO`, default
  `https://github.com/NVIDIA/cosmos-framework.git`
- `NPA_COSMOS3_MODEL_ID`, default `nvidia/Cosmos3-Nano`
- `NPA_COSMOS3_CACHE`, default `/tmp/npa-cosmos3-cache`
- `NPA_COSMOS3_GITHUB_TOKEN_ENV`, default `GITHUB_TOKEN`
- `NPA_COSMOS3_HF_TOKEN_ENV`, default `HF_TOKEN`
- `NPA_COSMOS3_NGC_API_KEY_ENV`, default `NGC_API_KEY`
- `NPA_COSMOS3_REQUIRE_NGC`, default `0`

Never print token values. When diagnosing, report only whether each credential
is configured, missing, skipped, or failed.

## Upstream Cosmos3 Notes

When an agent needs upstream package guidance, clone or inspect
`https://github.com/NVIDIA/cosmos-framework.git` and read:

- `docs/setup.md` for system packages, uv groups, CUDA variants, and container
  setup.
- `skills/atomic/cosmos3-setup/SKILL.md` for NVIDIA's original setup guidance.
- `skills/atomic/cosmos3-env-troubleshoot/SKILL.md` for known error signatures.

Training extras use `cu130-train` or `cu128-train`. Inference-only groups omit
training dependencies. Inside NGC PyTorch containers, clear `LD_LIBRARY_PATH`
before Python imports if PyTorch reports internal import failures.
