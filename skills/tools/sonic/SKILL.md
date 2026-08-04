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
- Workflow: SONIC specs live under `npa/workflows/workbench/npa-workflows/`
  (`sonic-train.yaml`, `sonic-export.yaml`, `sonic-eval.yaml`,
  `sonic-export-eval.yaml`, `sonic-locomotion-finetuning.yaml`). Submit them with
  `npa workbench workflow submit`. `sonic export` / `sonic eval` accept `s3://`
  checkpoints, ONNX policies and outputs, so no staging glue is needed.
  Raw SkyPilot coverage for the submit wrapper lives in
  `npa/tests/fixtures/skypilot/`; shipped SONIC workflow authoring stays on the
  `npa.workflow` specs.

## Routing And Validation

- Use the baked `npa-sonic:0.1.2` image for L40S VM targets.
- Use the host-mounted image selected by
  `npa/src/npa/deploy/sonic_image_manifest.json` for RTX PRO 6000 Blackwell
  Kubernetes targets with NVIDIA GPU Operator mounted drivers. The CUDA 13
  image inherits the truthful `sm80-sm90-sm100-sm103-sm120` base alias; do not
  reconstruct that tag in callers.
- SONIC render validation requires RT-capable GPUs. H100 can be useful for
  non-render training throughput, but it is not the default render-validation
  target.

## Runtime Isaac bootstrap (the container ships no Isaac Sim)

The `npa-sonic` image contains **no NVIDIA Isaac Sim or Isaac Lab code**. It used to bake
Omniverse Kit, which made it non-redistributable; Isaac is now downloaded on first use of
`/isaac-sim/python.sh` from `https://pypi.nvidia.com`, into a cache volume, under the
**operator's own EULA acceptance**. Full rationale:
`docs/workbench/container-packaging.md` and `skills/atomic/solution-licensing/SKILL.md`.

What this changes in practice:

- **Set both variables, or Isaac will not start.** Missing either
  `OMNI_KIT_ACCEPT_EULA=YES` or `ISAACSIM_ACCEPT_EULA=YES` makes the container exit **78**
  with a message naming them. That refusal is deliberate and load-bearing — do not "fix"
  it by baking acceptance into the image; a guard fails the build if anyone does.
- **Reach Isaac through `/isaac-sim/python.sh`** (the value of `ISAAC_LAB_PYTHON`). That is
  the bootstrap shim, and it is what every SkyPilot template, the sim2real engine and the
  workbench CLI already use. A bare `python3` is the *system* interpreter and will not
  find Isaac.
- **Never invoke the shim from a Dockerfile `RUN`.** It would download and bake ~4.5 GB of
  Isaac into a layer. Build-time work uses the image's own venv python.
- **Budget the first start.** Measured on RTX PRO 6000: 111 s cold, 32 ms warm, 10.04 GiB
  of cache. Pre-warm a shared volume once per node/PVC with
  `npa/docker/workbench/common/warm-isaac-cache.yaml`, then run workload pods with
  `NPA_ISAAC_CACHE_READONLY=1`. Otherwise every pod pays it, and 8 GPU pods on a node
  download ~36 GB.
- `isaac-bootstrap status` reports what is cached without needing acceptance or network;
  `isaac-bootstrap verify` additionally launches Isaac Sim headless (needs a GPU).
- No NGC credentials are needed to build or run this image.

Sonic's entrypoint picks its interpreter **per mode**: `smoke`, `eval`, `train`,
`finetune` and `serve` go through the Isaac bootstrap and therefore need acceptance, while
`mujoco-eval` and the S3/GPU-proof helpers run on the baked venv and need neither the
download nor the EULA variables. So `npa-sonic-mujoco`'s eval stays fast and offline.

The image also no longer bakes NVIDIA driver userspace libraries: the container runtime
injects the host driver and the Vulkan ICD given `NVIDIA_DRIVER_CAPABILITIES=all`
(verified on RTX PRO 6000 — `vulkaninfo --summary` reports the discrete GPU at driver
580.95.05). `VK_ICD_FILENAMES` is deliberately not pinned.

The image must carry `lxml` and `open3d` in its baked Python environment. The real
training path imports both while constructing the motion library, although the pinned
upstream training extra declares neither. Keep the Dockerfile's build-time import
assertions and the one-iteration real fine-tune smoke together; an import-only SONIC
check is not enough.

## Gotchas

- Keep `SONIC_GPU_TYPE` and `SONIC_IMAGE_VARIANT` aligned with the image
  manifest. Do not assume one image works across VM and Kubernetes targets.
- Known issue: job ID reuse anomaly. Treat it as a deferred investigation unless
  the task directly targets scheduler identity handling.
- The CUDA 13 Kubernetes image's real fine-tune smoke currently reaches Isaac
  environment construction on RTX PRO 6000 but can fail in the runtime-fetched
  URDF extension while opening `/tmp/IsaacLab/.../pelvis.tmp.usd`; a same-pod
  warm retry reproduced it on 2026-08-03. Do not count native-SASS, imports, or
  environment startup as SONIC capability validation: require the checkpoint.
- CUDA 13 alignment is vendor-paced on NVIDIA x86_64 CUDA 13 and is not a
  Nebius-blocked item.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes `npa workbench sonic export --help` and
`npa workbench sonic serve --help` so the skill cannot regress to the older
train/eval-only command list.
