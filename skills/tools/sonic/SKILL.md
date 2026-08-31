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

- The legacy baked L40S and inherited MuJoCo variants are quarantined because
  their built bytes contain restricted NVIDIA payloads; resolvers must reject them.
- `sonic-mujoco-runtime-fetch` is an independently rebuilt public release on
  a digest-pinned Python base and hash-locked CUDA Toolkit/MuJoCo closure. It is
  not derived from either quarantined image; release promotion is bound to its
  exact clean, GPU-accepted public development digest.
- Use the active host-mounted runtime-fetch image selected by
  `npa/src/npa/deploy/sonic_image_manifest.json` for RTX PRO 6000 Blackwell
  Kubernetes targets with NVIDIA GPU Operator mounted drivers. A B300
  validation image uses the same Dockerfile but must be supplied explicitly by
  immutable digest until its own accepted release updates the manifest. B300
  is compute capability 10.3: that development image must use the
  digest-pinned CUDA 13 base, PyTorch `cu130`, CUDA 13 NVRTC, and the truthful
  `sm80-sm90-sm100-sm103-sm120` target contract. The official cu130 PyTorch
  wheel exposes Blackwell 10.x family SASS as `sm_100`; the literal `sm_103`
  requirement applies to CUDA 13 NVRTC JIT compilation on the B300 device.
  CUDA 12.8 NVRTC rejects `sm_103`; do not treat environment startup before
  that JIT boundary as B300 training evidence.
- SONIC render validation requires RT-capable GPUs. Use RTX PRO 6000 Blackwell;
  do not silently fall back to the quarantined H100/L40S images.
- The built-in serverless compute-only default is intentionally unavailable:
  its L40S/H100/H200 images are quarantined, while the active image requires
  Kubernetes GPU Operator driver mounts. A serverless run must supply an
  independently validated compute-only `--image`, or fail before provisioning.

## Runtime Isaac bootstrap (the container ships no Isaac Sim)

Before an Isaac-backed SONIC run, load
`skills/atomic/third-party-eula-preflight/SKILL.md`.

The `npa-sonic` image contains **no NVIDIA Isaac Sim or Isaac Lab code**. It used to bake
Omniverse Kit, which made it non-redistributable; Isaac is now downloaded on first use of
`/isaac-sim/python.sh` from `https://pypi.nvidia.com`, into a cache volume, under the
**operator's own EULA acceptance**. Full rationale:
`docs/workbench/container-packaging.md` and `skills/atomic/solution-licensing/SKILL.md`.

What this changes in practice:

- **Isaac-backed modes default acceptance; explicit opt-out refuses.** NPA
  defaults `ACCEPT_EULA=Y`. Empty, `N`, `NO`, `0`, `FALSE`, or
  `--no-accept-eula` exits **78** before download. `Y`, `YES`, `1`, and `TRUE`
  are accepted case-insensitively; other values are invalid. The launcher derives
  `OMNI_KIT_ACCEPT_EULA=YES` internally; do not expose duplicate user plumbing.
  Keep `PRIVACY_CONSENT` and telemetry off.
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

SONIC's entrypoint picks its interpreter **per mode**: `smoke`, `eval`, `train`,
`finetune` and `serve` go through the Isaac bootstrap and therefore need acceptance, while
`mujoco-eval` and the S3/GPU-proof helpers run on the baked venv and need neither the
download nor the EULA variables. So `npa-sonic-mujoco`'s eval stays fast and offline.

The image also no longer bakes NVIDIA driver userspace libraries: the container runtime
injects the host driver and the Vulkan ICD given `NVIDIA_DRIVER_CAPABILITIES=all`
(verified on RTX PRO 6000 — `vulkaninfo --summary` reports the discrete GPU at driver
580.95.05). `VK_ICD_FILENAMES` is deliberately not pinned.

The image must carry `lxml`, `open3d`, and `vector_quantize_pytorch` in its baked Python
environment. Real training imports the first two while constructing the motion library
and instantiates `vector_quantize_pytorch.FSQ` while building the actor-critic, although
the pinned upstream training extra declares none of them. Keep the Dockerfile's
build-time import assertions and the one-iteration real fine-tune smoke together; an
import-only SONIC check is not enough.

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
