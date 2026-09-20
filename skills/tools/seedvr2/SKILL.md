---
name: seedvr2
description: Use when packaging, running, validating, or reviewing official SeedVR2-3B restoration of robot observation video, its S3 workflow, or objective and visual evidence.
---

# SeedVR2 video restoration

Use this skill for the first-class `npa-seedvr2` batch tool and its evidence.
Read these files before changing behavior:

- `npa/src/npa/workbench/seedvr2/`
- `npa/docker/workbench/seedvr2/REDISTRIBUTION.md`
- `workflows/testing/seedvr2-video-restoration.yaml`
- `docs/workbench/seedvr2.md`

Also load `npa-cli-conventions`, `toolref-argv-contract`,
`author-npa-workflow`, `real-components`, `solution-licensing`,
`secure-image-build`, `gpu-selection`, `testing-conventions`, and
`agent-visual-feedback` when their surfaces are involved.

## Ground truth

- Source is `ByteDance-Seed/SeedVR` at
  `e4de8c24441a67e1b7df56abea10645059bb1185`, Apache-2.0.
- Model is `ByteDance-Seed/SeedVR2-3B` at
  `37255ff8cccfb01071b87f635a5948ca8d53117c`, Apache-2.0 and public.
- Runtime requires all four payloads and verifies the fixed sizes and SHA-256
  values in `npa.workbench.seedvr2.schemas.MODEL_FILES`.
- The real entrypoint is upstream `projects/inference_seedvr2_3b.py` under
  one-process `torchrun`, seed 666 and sequence-parallel size 1.
- The candidate hardware contract is exactly one full-memory H100 (`sm_90`), a
  digest-bound `NPA_TASK_IMAGE`, and the full NPA source revision baked into
  that image. A caller-supplied revision, tag, compatible image, MIG slice, or
  successful import is not capability evidence.
- Model weights, customer video, outputs, credentials, and populated caches are
  runtime-only. No per-run model terms prompt is required for these public
  Apache-2.0 weights.

Do not silently replace upstream inference with interpolation, sharpening,
another restoration model, or a synthetic stub. The bicubic image exists only
as an explicit review baseline.

## Operate

```bash
npa workbench workflow validate-spec \
  workflows/testing/seedvr2-video-restoration.yaml
npa workbench workflow plan-spec \
  workflows/testing/seedvr2-video-restoration.yaml --run-id seedvr2-plan
npa workbench workflow submit \
  workflows/testing/seedvr2-video-restoration.yaml \
  --var bucket="<your-bucket>" \
  --var seedvr2_input_uri="s3://<your-bucket>/inputs/low-resolution.mp4"
```

The canonical graph is `probe -> restore -> verify -> review`. Keep the CPU
probe, GPU inference, digest-bound GPU identity/readback verification, and
review stages separate.
Every command uses `--input-path`, `--output-path`, and `--run-id`; restore also
receives mandatory `--probe-path` and must reject a changed run ID, input bytes,
or media. Artifacts move through S3 and are create-only per object. A partial
publication remains failure evidence; retry it under a new run ID and prefix.
Source and output frames must stay within the reviewed 1920x1080 area budget; output
dimensions must also preserve source aspect ratio and be divisible by 16.

Configure the optional service with `SEEDVR2_TOKEN` and
`SEEDVR2_ALLOWED_S3_ROOTS`. Keep authentication, request serialization,
canonical unescaped storage-root authorization, local-MP4-only media decoding,
and the purpose-specific credential-free subprocess environment allowlists.

## Evidence contract

Retain source, degraded input, bicubic baseline, restored candidate, and
light-degradation negative control as separate MP4s. Record exact commands,
frame mapping, source/model/image/commit identities, timestamps, hardware,
decoded media properties, and SHA-256 values. Verify S3 bytes after upload.
Treat `verification.json` as artifact/runtime consistency evidence only; pair it
with the platform workflow receipt that identifies the actual producer pod.

For the fixed real Aloha-Agilex proof, compare candidate and bicubic against the
original high-resolution reference. Report per-frame, per-stratum, and aggregate
LPIPS and SSIM; optical-flow warp error; bottle-mask centroid error and IoU.
Keep all failures and negative-case regressions. Run matched, blinded
image-capable annotation and calibrated VLM review. A presentation score cannot
override objective or provenance failure.

The review package must remain non-blended and visibly identify bicubic-left and
SeedVR2-right. Never describe generated texture as recovered sensor truth.

## Limits

SeedVR2 does not prove recovered geometry, camera calibration, policy success,
or suitability as training ground truth. Upstream warns that heavy degradation,
large motion, light degradation, and small inputs can cause incomplete
restoration, unpleasant generated detail, or oversharpening. Preserve the
source recording and state these limits in user-facing evidence.

## Publication and validation

The image version remains `0.1.0-cu130-unbuilt` and
`UNVALIDATED_PUBLICATION_TOOLS` must contain `seedvr2` until exact built-image
scans, anonymous registry verification, real H100 workflow evidence, objective
metrics, real VLM review, and independent review all pass for the same commit
and digest.

Use the repository interpreter:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_seedvr2.py \
  npa/tests/cli/test_seedvr2_cli.py \
  npa/tests/workflows/test_seedvr2_workflow.py \
  npa/tests/docker/test_seedvr2_container.py -q
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/smoke/test_golden_eval_manifest.py \
  npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py -q
```

Never report unit tests, a local image build, or the golden-eval declaration as
real GPU, objective-quality, VLM, workflow, or publication evidence.
