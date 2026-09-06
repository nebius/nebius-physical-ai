---
name: cosmos3-inference
description: Use when running or modifying Cosmos3 inference through NPA, including Nano diffusion continuation/augmentation, framework generation, prompt/input handling, and effective guardrail or sampling arguments.
---

# Cosmos3 Inference

## Source And Attribution

Adapted from NVIDIA cosmos-framework
`skills/workflows/cosmos3-inference/SKILL.md`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Used under OpenMDW-1.1.
See `skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1` and
`skills/NOTICE-NVIDIA-COSMOS3`.

## When To Use

Use this skill when the user wants to generate an image or video with Cosmos3,
change inference defaults, verify prompt handling, inspect guardrails behavior,
or connect NPA's Cosmos3 workflow to upstream Cosmos framework inference docs.
For environment errors, use
`skills/atomic/cosmos3-env-troubleshoot/SKILL.md`.

## Real NPA Workflows

Choose the implementation matching the deployment: persistent Nano vLLM-Omni
diffusion, containerized framework generation, or the text-to-image smoke.

### Persistent Nano diffusion video

Read `npa/deploy/cosmos3-nano-video/README.md` for the CLI/SDK contracts, image,
frame mapping, measured results and reusable commands. `nano-video-batch`
generates a text-to-video segment followed by tail-conditioned continuations.
`nano-video-augment` transforms a complete source MP4 using official Canny edge
controls from every corresponding source interval. A continuation result alone
does not demonstrate visual augmentation or full source-motion conditioning.

The augmentation client accepts S3 input/output paths at 832×480 and 24 fps,
validates the complete media and publishes immutable artifacts with readback.
Use the supported sampling/control flags; there is no generic `strength` or
unchecked extra-parameter bag. Later windows use the preceding augmented
five-frame RGB tail for continuity and matching original-source edges for
structure. Preserve exact effective prompts, control provenance and all joins.
Keep actual source, augmented output and synchronized comparison clearly labeled.

`nano-video-augment-recover` retrieves the same request or retries artifact
publication without submitting generation. Preserve the original destination
and submission marker after interrupted generation or artifact retrieval; a
missing result is ambiguous. Never repeat GPU generation merely to retry an
upload. Validate visual change, identity, source motion, contact and temporal
joins separately from decode/hash checks, with a prior rubric and disclosed
agent/VLM sampling limits. The README's selected settings are measured examples,
not universal quality defaults.

Changes to shared serving code require the FIFO/least-outstanding regression and
affected real continuation acceptance as well as complete augmentation evidence.
An eight-request continuation result does not establish eight-way augmentation.

### Containerized generate (preferred)

```text
npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml
npa workbench cosmos3 generate
npa.sdk.workbench.cosmos3.generate(...)
workbench.cosmos3.generate            # npa.workflow toolRef
```

All four surfaces run one implementation,
`npa/src/npa/workbench/cosmos/generate.py`, inside the `npa-cosmos3` image
(`npa/docker/workbench/cosmos3/Dockerfile`). The image bakes the framework at a
pinned commit plus its cu130 inference venv, so a run does not clone or resolve
dependencies on the node. Modes: `text2image`, `text2video`, `image2video` and
`video2video`. The `image2video` and `video2video` modes require `--input-path`.

No weights are baked. Public `nvidia/Cosmos3-Nano` downloads anonymously; when
guardrails are enabled, their gated weights download at run time with the
operator's own `HF_TOKEN`. `NPA_COSMOS3_REQUIRE_NGC=1` additionally demands
their `NGC_API_KEY`; `require_model_access` refuses only for the selected gated
or NGC-hosted path. Use
`--dry-run` to inspect the resolved input sample and inference argv from a CPU
host.

### Clone-at-job-time text-to-image smoke

```text
npa/workflows/workbench/npa-workflows/cosmos3-text-to-image.yaml
npa workbench cosmos3 text-to-image
workbench.cosmos3.text_to_image
```

A real H100 text-to-image smoke that needs no prebuilt image: it clones the
Cosmos framework, downloads the configured Hugging Face model, creates a
text-to-image JSON input, runs `python -m cosmos_framework.scripts.inference`,
validates the produced image, and optionally uploads the image plus success JSON
to S3. Keep it for BYO-fork / un-baked-image cases.

Do not replace these implementations with a skill-display subcommand; Cosmos3 skills are
SKILL.md files for agents, not commands.

## Guardrails

Defaults differ by implementation. `npa workbench cosmos3 generate` defaults
to guardrails on and exposes `--no-guardrails`. The separate
`npa workbench cosmos3 text-to-image` defaults to off and exposes `--guardrails`
for opt-in; its shipped workflow explicitly describes disabled guardrails.
The Nano diffusion recipe configures `--no-guardrails` at deployment time.
Inspect the chosen route and effective manifest, preserve the operator's
configuration, and check access to any newly selected gated guardrail payloads.
Do not infer one route's posture from another route's defaults.

## Running The Workflow

Before launch, confirm credentials and access:

```bash
npa/.venv/bin/npa workbench cosmos check --output json
```

For the text-to-image smoke, review the actual workflow configuration keys:
`cosmos_source_repo`, `cosmos_model_id`, `cosmos_cache_dir`, `t2i_prompt`,
`t2i_checkpoint_name`, `t2i_uv_group`, `t2i_seed` and `t2i_output_uri`.
The YAML calls `workbench.cosmos3.text_to_image`; the implementation lives in
`npa/src/npa/workbench/cosmos/text_to_image.py`. Retired raw shell-template
environment fields are not the current workflow contract.

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
  npa/tests/workbench/test_cosmos3_generate.py \
  npa/tests/cli/test_cosmos3_cli.py \
  npa/tests/docker/test_cosmos3_image_contract.py
```

Expected checks include:

- Route-specific guardrail defaults and explicit overrides remain intact;
  credential preflight checks the actual selected gated dependencies without
  demanding a token solely for anonymous public Nano weights.
- The `npa-cosmos3` Dockerfile pins the framework commit and never fetches
  weights in a build layer.
- The inference YAML name is `cosmos3-text-to-image` and its toolRef is
  `workbench.cosmos3.text_to_image`.
- `image_id` is not hard-coded in the resources.
- The command invokes `python -m cosmos_framework.scripts.inference`.
- S3 output remains optional.
- Nano augmentation rejects unsupported fields, covers the entire original
  source with structural controls and recovers immutable artifacts without
  repeating generation; see its core/client/server and live acceptance tests.
