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

The experimental `workflows/testing/cosmos3-policy-model-factory.yaml` now
invokes native LIBERO-10 SFT through `npa workbench cosmos3 policy-train`, then
the matching `policy-eval`, `policy-feedback`, and `failure-candidates`
primitives. See `docs/workbench/cosmos3-policy-model-factory.md` and its linked
readiness record before claiming GPU qualification. The new shared modules are
`npa/src/npa/workbench/cosmos/policy_*.py`; CLI and SDK call these implementations.

The runtime synchronizes a separate pinned `cu130-train` environment. Do not
treat the baked inference environment as training support. Keep the native
20 Hz LIBERO action, camera, normalization, and gripper semantics paired with
the simulator. A reduced execution smoke cannot qualify the policy: selection
requires all ten tasks and fifty trials each. Absolute qualification does not
prove improvement against an incumbent. Generated videos remain ineligible for
action training until labels are independently validated. Cross-run training
resume and automatic subsequent learning rounds are not implemented.

Retained real Cosmos3 workflows:

- `workflows/testing/cosmos-fetch.yaml`
- `workflows/testing/cosmos3-text-to-image.yaml`

Current NPA Cosmos commands such as `npa workbench cosmos train` cover the
existing Cosmos workbench/serverless training surface, not a proven Cosmos3 SFT
workflow. Do not present that as Cosmos3 post-training unless implementation and
tests explicitly support it.

## Upstream Post-Training Map

For reserved B200 Slurm scaling, use the standalone application recipe at
`npa/workflows/workbench/cosmos3-wam-slurm/` and
`docs/workbench/cookbooks/cosmos3-wam-slurm.md`. It plans native LIBERO WAM runs
with eight ranks per node, a fixed nominal batch and explicit HSDP topology.
Its `validation.json` records completed 8/16-GPU 2,000-update schedules,
three timing repeats per topology, separate CUDA profiles and four 500-trial
checkpoint evaluations per topology. Full training took 7 h 50 m 33 s and
4 h 23 m 33 s; repeated steady-step speedup was 1.9334×, efficiency 96.7%.
Preserve the documented storage/cache conditions and small actual token-work
difference (0.0108%) when quoting these results. Full-run and steady-step
speedups differ; training GPU-hours exclude preparation, idle time and evaluation.

Final benchmark success was 95.0% and 95.8%. The first saved checkpoint above
90% was update 1,500, available after about 5 h 53 m and 3 h 19 m, respectively.
Quality was verified afterward; do not infer exact threshold crossing, online
early stopping, unseen-task generalization, or improved quality from GPU count
with only one training seed. All original numeric/visual evidence is linked
from the recipe. Historical raw reports retain their original
measurement-time scope; the later checkpoint-linked quality records supersede
pending campaign status without rewriting those bytes.

The reserved two-node run has sixteen-rank NCCL/InfiniBand proof, attributed
training processes and full-run telemetry. Native Slurm was tested; Soperator
remains an unvalidated deployment alternative. One FP32-plus-EMA B200 failed
before its first update. Four active B200s subsequently passed NCCL and initial
optimizer updates on an exclusive eight-GPU VM; see
`docs/workbench/evidence/cosmos3-wam-live-training-4/README.md`. Full four-GPU
duration, scaling and quality are pending in that record. Distinguish four active
training GPUs from eight provisioned GPUs. Two GPUs are untested; 32 are plan-only.
Current upstream LIBERO already describes two-node training; avoid claiming
that all public Cosmos3 recipes stop at one node.

Both bootstraps install FFmpeg and every node decodes both camera streams
before model loading. Keep the excluded failure and corrected receipts.
CUDA profiles observe rank zero on eight GPUs and ranks zero/eight on sixteen;
kernel groups use linked CPU operators where available. Overlapping kernel
durations are not exposed communication stalls or a wall-time breakdown.
Use a fresh `profile_report.py --output-path` when reanalyzing archived traces,
preserving the original report and trace hashes.

The Slurm recipe pins a newer framework than the experimental Workbench policy
workflow. Keep each run's source/model/data/evaluation revisions coherent;
do not use one path's successful execution as proof for the other.

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
