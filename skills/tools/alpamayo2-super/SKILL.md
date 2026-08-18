---
name: alpamayo2-super
description: Package, preflight, run, validate, or troubleshoot NVIDIA Alpamayo 2 Super 34B VLA inference in NPA, including its OpenMDW model terms, separately gated PhysicalAI-AV dataset, public runtime-fetch image, B200/RTX PRO 6000 routing, workflow artifacts, and real-GPU evidence.
---

# Alpamayo 2 Super

Run NVIDIA's real VLM plus 2.3B diffusion expert. Never replace inference with
an import, fabricated trajectory, or manifest-only smoke.

## Legal gates

Keep the boundaries separate:

- Source: `NVlabs/alpamayo2@beb2977d9a7e9d66837d4a3ad5144ff59de37519`,
  Apache-2.0, baked with its license and the marked dataset-revision patch.
- Model: `nvidia/Alpamayo2-Super@00554695e729a6ff0b6281fd2c81b18d06e33dbe`,
  OpenMDW-1.1. Acceptance is by exercising granted rights. Fetch at runtime.
- Dataset: `nvidia/PhysicalAI-Autonomous-Vehicles@b719eea7f0a63619ef51ec7f54178af0937ef050`,
  gated by NVIDIA's AV Dataset License. It is non-transferable; accept it
  interactively on Hugging Face and fetch it only into the operator cache.
- Outputs: OpenMDW-1.1 imposes no output restriction, but Alpamayo is not an
  automotive-grade driving stack. Preserve model-card safety provenance.

Before provisioning, run:

```bash
npa workbench health preflight
npa workbench health access --capability alpamayo2-super
```

A missing dataset entitlement is terminal. Open the exact URL printed by the
access command; Hugging Face acceptance cannot be automated by NPA.

## Build and scan

Build only from the checked-in Dockerfile. Never pass `HF_TOKEN` as a build arg
or populate `/workspace/.cache/huggingface` during a build.

```bash
bash npa/docker/workbench/alpamayo2-super/build.sh
npa/.venv/bin/python npa/scripts/scan_image_alpamayo2_payload.py \
  <exact-local-image>
```

Require a complete clean scan over every layer for checkpoints, PhysicalAI-AV
payload, caches, and credentials. Inspect SBOM/provenance separately; the byte
scanner is not a license review.

## Run the workflow

Use B200 first. The workload is headless and does not need RT cores; its measured
H100 peak is 72,115 MiB, so one B200 has ample memory. Validate RTX PRO 6000
separately because B200 `sm_100` does not prove RTX `sm_120`.

```bash
npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/alpamayo2-super-inference.yaml
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/alpamayo2-super-inference.yaml \
  --infra <configured-infra-target> --var bucket=<operator-bucket> \
  --secret-env HF_TOKEN --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

The run must publish non-empty `trajectory.json`, `trajectory.png`, and
`result.json`. Verify exact model/dataset revisions, the runtime image digest,
`weights_baked=false`, `dataset_baked=false`, and the node-local ephemeral cache
tier. Treat success without all three artifacts as failure.

## Diagnose

- 401/403 before GPU allocation: accept the dataset agreement with the same HF
  account or replace the rejected token; do not add an NPA bypass boolean.
- CUDA OOM: confirm one trajectory sample, ten diffusion steps, and no competing
  process; do not silently reduce cameras or frames.
- `no kernel image`: inspect Torch arch flags and flash-attn build targets.
  B200 requires `sm_100`; RTX PRO 6000 requires `sm_120`.
- Missing camera projection: retain fail-closed `--require-camera-projection`.
- Pending job: inspect scheduling and image-pull evidence. Change GPU products
  only for scheduler/capacity evidence and record a separate architecture result.

Cancel the exact run before teardown. Never delete shared operator resources
merely because a validation job finished.

## Accepted release baseline

`0.1.0-cu128` is the accepted runtime-fetch baseline. Its OCI index digest is
`sha256:2164450f8baf57d8798f64063ea27bf11611f5b695c467de0c2e319e3134ebd5`.
On 2026-08-18 the exact digest completed the real upstream workflow on B200
(`sm_100`) and, independently, RTX PRO 6000 (`sm_120`). The 26-layer payload
scan was clean. RTX peak allocation was 71,447 MiB; the B200 run completed but
was not sampled for peak memory, so retain NVIDIA's 72,115 MiB H100 measurement
as the conservative documented reference rather than inventing a B200 number.

For RTX qualification, preserve the committed B200 default and override the
submitted accelerator with `NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO6000:1`. Require
the same three artifacts and provenance checks; do not infer RTX support from a
B200 result.

## Verify changes

```bash
npa/.venv/bin/python ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/tools/alpamayo2-super
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_alpamayo2_super.py \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py -q
```
