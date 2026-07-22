---
name: sonic
description: Use when working on SONIC whole-body-control training, export, evaluation, serving, GPU routing, validation, or CUDA alignment.
---

# SONIC

## When To Use

Use this skill for NVIDIA GEAR-SONIC whole-body-control workbench changes,
including standalone training, export, evaluation, serving, image routing, and
workflow composition with retargeting or MJLab.

## Procedure

1. Confirm the current CLI surface before editing:

   ```bash
   npa workbench sonic --help
   ```

2. Use `train` for policy training, `export` for export artifacts, `eval` for
   evaluation, and `serve` for runtime serving. Use `deploy`, `status`, and
   `list` for operational lifecycle.
3. Route first-party SONIC images through
   `npa/src/npa/deploy/sonic_image_manifest.json`; do not hardcode image tags in
   workflows or docs.
4. Keep S3 paths run-scoped so retraining and re-evaluation do not overwrite
   previous artifacts.

## Three-Tier Contract

- CLI: `deploy`, `train`, `export`, `eval`, `serve`, `status`, and `list`.
- SDK/API: keep train/eval/export/serve request construction shared with service
  payloads and tests where possible.
- YAML: SONIC workflows live under `npa/src/npa/workflows/skypilot/`, including
  `sonic-train-standalone.yaml`, `sonic-export.yaml`, `sonic-eval.yaml`,
  `sonic-export-eval.yaml`, and `sonic-locomotion-finetuning.yaml`.

## Routing And Validation

- Use the baked `npa-sonic:0.1.2` image for L40S VM targets.
- Use the host-mounted `npa-sonic:0.1.2-k8s` image for RTX PRO 6000 Blackwell
  Kubernetes targets with NVIDIA GPU Operator mounted drivers.
- SONIC render validation requires RT-capable GPUs. H100 can be useful for
  non-render training throughput, but it is not the default render-validation
  target.

## Gotchas

- Keep `SONIC_GPU_TYPE` and `SONIC_IMAGE_VARIANT` aligned with the image
  manifest. Do not assume one image works across VM and Kubernetes targets.
- Known issue: job ID reuse anomaly. Treat it as a deferred investigation unless
  the task directly targets scheduler identity handling.
- CUDA 13 alignment is vendor-paced on NVIDIA x86_64 CUDA 13 and is not a
  Nebius-blocked item.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes `npa workbench sonic export --help` and
`npa workbench sonic serve --help` so the skill cannot regress to the older
train/eval-only command list.
