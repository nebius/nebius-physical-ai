---
name: cosmos3-post-training
description: Use when planning, reviewing, or explaining Cosmos3 supervised fine-tuning and post-training in NPA, including upstream recipes, dataset/checkpoint preparation, and why NPA does not expose post-training as a fake skill command.
---

# Cosmos3 Post-Training

## Source And Attribution

Adapted from NVIDIA cosmos-framework
`skills/workflows/cosmos3-post-training/SKILL.md`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Used under OpenMDW-1.1.
See `skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1` and
`skills/NOTICE-NVIDIA-COSMOS3`.

## When To Use

Use this skill when the user asks how Cosmos3 SFT works, how to review future
post-training support, where upstream recipes live, how to validate training
configs, or whether an NPA change should expose post-training.

For current NPA, treat Cosmos3 post-training as guidance and planning unless a
real executable workflow is implemented and tested. Do not add a Cosmos
skill-display subcommand or a SkyPilot YAML whose only purpose is to make this
agent skill runnable.

## Current NPA Boundary

Retained real Cosmos3 workflows:

- `npa/workflows/workbench/skypilot/cosmos3-ea-fetch.yaml`
- `npa/workflows/workbench/skypilot/cosmos3-text-to-image-inference.yaml`

Current NPA Cosmos commands such as `npa workbench cosmos train` cover the
existing Cosmos workbench/serverless training surface, not a proven Cosmos3 SFT
workflow. Do not present that as Cosmos3 post-training unless implementation and
tests explicitly support it.

## Upstream Post-Training Map

In a clone of `https://github.com/NVIDIA/cosmos-framework.git`, inspect:

| Need | Upstream path |
| --- | --- |
| Training guide | `docs/training.md` |
| Dataset JSONL/captioning guide | `docs/dataset_jsonl.md` |
| SFT recipes | `examples/toml/sft_config/<recipe>.toml` |
| Paired recipe launchers | `examples/launch_sft_<recipe>.sh` |
| Common launcher helper | `examples/_sft_launcher_common.sh` |
| Training script | `cosmos_framework/scripts/train.py` |
| DCP conversion | `cosmos_framework/scripts/convert_model_to_dcp.py` |
| HF export | `cosmos_framework/scripts/export_model.py` |
| TOML schema | `cosmos_framework/configs/toml_config/sft_config.py` |

## Planning Checklist

When reviewing or designing NPA Cosmos3 post-training support:

1. Define the exact executable outcome: config validation, dry run, training,
   checkpoint conversion, export, or inference from a trained checkpoint.
2. Require explicit dataset, base checkpoint, and Wan VAE paths where the
   upstream recipe requires them.
3. Keep training extras explicit: `cu130-train` or `cu128-train`.
4. Validate TOML/schema behavior with upstream `train.py --dryrun` before
   claiming training support.
5. Use temporary or user-selected output roots, not repository paths.
6. Preserve redaction for Hugging Face, GitHub, NGC, S3, and any other secret
   env values.
7. Add tests that prove NPA maps inputs into a real executable workflow. Do not
   use tests that only prove an agent skill can be listed or displayed by a CLI.

## Upstream Workflow At A Glance

The upstream flow is:

1. Install training extras and clear `LD_LIBRARY_PATH` if needed.
2. Prepare the dataset and any required VAE artifact.
3. Convert the base Hugging Face checkpoint to DCP when the recipe requires it.
4. Launch the paired `examples/launch_sft_<recipe>.sh` script or an equivalent
   raw `torchrun` command.
5. Find checkpoints under the upstream training output root.
6. Run inference with the trained DCP checkpoint and its `config.yaml`.
7. Optionally export to Hugging Face safetensors.

If the user needs inference after training, switch to
`skills/workflows/cosmos3-inference/SKILL.md` and point `--checkpoint-path`
plus any upstream config file at the trained checkpoint output.
