---
name: cosmos3-codebase-nav
description: Use when navigating the Cosmos3 integration in NPA or locating upstream Cosmos3 framework files, defaults, scripts, configs, recipes, and docs.
---

# Cosmos3 Codebase Navigation

## Source And Attribution

Adapted from NVIDIA cosmos-framework
`skills/atomic/cosmos3-codebase-nav/SKILL.md`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Used under OpenMDW-1.1.
See `skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1` and
`skills/NOTICE-NVIDIA-COSMOS3`.

## When To Use

Use this skill when the user asks where Cosmos3 setup, access checks, inference
defaults, parser settings, workflows, or post-training configs live. Also use it
when reviewing or modifying PRs that touch Cosmos3 support in NPA.

## NPA Integration Map

Primary NPA files:

| Need | File |
| --- | --- |
| Cosmos3 access, fetch, redaction, parser config, inference arg builder | `npa/src/npa/workbench/cosmos/cosmos3.py` |
| Human CLI commands for real workflows | `npa/src/npa/cli/cosmos/__init__.py` |
| SDK compatibility exports | `npa/src/npa/workbench/cosmos/__init__.py`, `npa/src/npa/sdk/workbench/cosmos.py` |
| Fetch workflow | `npa/workflows/workbench/skypilot/cosmos3-ea-fetch.yaml` |
| Text-to-image H100 smoke workflow | `npa/workflows/workbench/skypilot/cosmos3-text-to-image-inference.yaml` |
| Unit tests for access, fetch, inference YAML, and skill artifacts | `npa/tests/workbench/test_cosmos3_access.py` |
| CLI tests for `check` and `fetch` | `npa/tests/cli/test_cosmos3_cli.py` |

The agent-skill artifacts live in this repository under:

- `skills/atomic/cosmos3-setup/SKILL.md`
- `skills/atomic/cosmos3-codebase-nav/SKILL.md`
- `skills/atomic/cosmos3-env-troubleshoot/SKILL.md`
- `skills/workflows/cosmos3-inference/SKILL.md`
- `skills/workflows/cosmos3-post-training/SKILL.md`

## What Not To Look For

There is no supported Cosmos skill-display subcommand. Cosmos3 skills are not
platform commands, job templates, or SDK callables. They are SKILL.md files for
coding agents to read.

The only retained Cosmos3 SkyPilot YAMLs are real executable workflows:

- `cosmos3-ea-fetch.yaml`
- `cosmos3-text-to-image-inference.yaml`

Do not restore `cosmos3-setup.yaml`, `cosmos3-codebase-nav.yaml`,
`cosmos3-env-troubleshoot.yaml`, or `cosmos3-post-training.yaml` as skill
launchers unless the product direction changes to a real executable workflow.

## Upstream Cosmos Framework Map

When a task needs upstream package internals, inspect a clone of
`https://github.com/NVIDIA/cosmos-framework.git`.

Common upstream paths:

| Need | Upstream path |
| --- | --- |
| Sampling args, setup args, override models | `cosmos_framework/inference/args.py` |
| Per-modality defaults | `cosmos_framework/inference/defaults/<mode>/sample_args.json` |
| Inference script | `cosmos_framework/scripts/inference.py` |
| Training script | `cosmos_framework/scripts/train.py` |
| Ray serving presets | `cosmos_framework/inference/ray/configs/latency.yaml`, `throughput.yaml` |
| SFT recipe TOMLs | `examples/toml/sft_config/<recipe>.toml` |
| SFT launch shells | `examples/launch_sft_<recipe>.sh` |
| SFT schema | `cosmos_framework/configs/toml_config/sft_config.py` |
| Setup docs | `docs/setup.md` |
| Inference docs | `docs/inference.md` |
| Training docs | `docs/training.md` |

## Navigation Routine

1. Start with `rg` in NPA:

   ```bash
   rg -n "NPA_COSMOS3|Cosmos3|cosmos3-text-to-image|guardrails" npa
   ```

2. If the answer depends on upstream behavior, inspect the upstream file rather
   than guessing from NPA wrappers.

3. Keep NPA and upstream responsibilities separate. NPA owns CLI/SDK wrappers,
   SkyPilot workflows, image defaults, and tests. NVIDIA's repo owns Cosmos3
   framework semantics, inference arguments, recipes, and training internals.

4. When adding docs or tests, reference SKILL.md artifacts as repository files,
   not as commands.
