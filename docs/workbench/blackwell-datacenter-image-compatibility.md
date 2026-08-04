# Blackwell datacenter image compatibility (B200 `sm_100` / B300 `sm_103`)

How to make a Workbench container image deployable on datacenter Blackwell, and how to prove it.

The per-image verdicts are machine-readable in `npa/docker/workbench/blackwell-dc-images.json` and gated by `npa/tests/docker/test_blackwell_dc_manifest.py`. This page is the human companion: the compatibility model, the build conventions, and the validation bar.

For the vendor-CUDA axis (which NVIDIA stacks are pinned to CUDA 12.8 on x86_64), see [NVIDIA platform architecture coverage](../nvidia-platform-architecture-coverage.md).

## 1. The compatibility model

Read this before touching a Dockerfile. "Blackwell" is a marketing family covering two incompatible CUDA majors:

| GPU | Family | Compute capability | SM target | Nebius platform |
|---|---|---|---|---|
| RTX PRO 6000 Blackwell | Blackwell workstation (GB20x) | 12.0 | `sm_120` | `gpu-rtx6000` |
| B200 | Blackwell datacenter (GB100) | 10.0 | `sm_100` | `gpu-b200-sxm` (us-central1) |
| B300 (Blackwell Ultra) | Blackwell datacenter (GB300) | 10.3 | `sm_103` | `gpu-b300-sxm` (uk-south1) |

Four rules follow:

1. **SASS does not cross a CUDA major.** `sm_120` binaries do not run on `sm_100`/`sm_103` and vice versa. Validating on RTX PRO 6000 proves nothing about B200/B300 — this is the single most common way a "Blackwell-ready" claim turns out to be false.
2. **Minor-version forward compatibility holds within a major.** `sm_100` SASS runs on a `sm_103` device, not the reverse. So prioritize `sm_100`: a build that covers B200 generally covers B300, and you confirm rather than rebuild.
3. **There are two arch knobs, and only one of them is yours.**
   - A *prebuilt torch wheel* ships a fixed fat-binary arch set. `TORCH_CUDA_ARCH_LIST` cannot change it — you have to pick a wheel index that includes the arch. cu128 and cu130 include `sm_100` and `sm_120`; cu124 and cu126 stop at `sm_90`. Check with `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"`.
   - *Source-compiled extensions* (flash-attn from source, Taichi/Genesis, natten, custom ops) obey `TORCH_CUDA_ARCH_LIST` at build time. Omit an arch and the failure surfaces at runtime as `no kernel image is available for execution on the device`.
4. **Datacenter Blackwell has no RT cores**, exactly like H100/H200. Isaac Lab and SONIC rendering must stay on L40S / RTX PRO 6000; only headless, state-based training routes to B200/B300. `npa.workbench.sonic.routing` classifies `b200`/`b300`/`sm_100`/`sm_103` as datacenter-headless and rejects render workloads on them.

Datacenter-only features (NVFP4, 2nd-gen Transformer Engine, NVLink, MIG) are exercised only on real B200/B300 — a workstation-Blackwell smoke never touches them.

Floor: R570+ driver, CUDA 12.8 or 13.0.

### `npa-base` sets the arch list for the tree

`npa/docker/workbench/base/cuda13-b300/Dockerfile` previously pinned `TORCH_CUDA_ARCH_LIST=10.3`, which meant every source-compiled extension inheriting the base carried B300 SASS only. It now defaults to `8.0 9.0 10.0 10.3 12.0` (A100, Hopper, B200, B300, RTX PRO 6000) as a build arg, and the build asserts the wheel reports `sm_80 sm_90 sm_100 sm_120`.

`sm_103` is deliberately absent from that assertion: stock cu130 wheels ship `sm_100` SASS, and B300 is reached by forward compatibility. Asserting `sm_103` would fail a perfectly good wheel.

`npa-base` also builds flash-attn-4 (CuTe), which JIT-compiles at runtime. That was assumed to make it usable on any Blackwell part; **real-GPU testing disproved it for `sm_120`.** The CuTe forward kernel partitions its epilogue with a TMA (Tensor Memory Accelerator) copy atom, and TMA is a datacenter feature — `sm_90`, `sm_100`, and `sm_103` have it, RTX PRO 6000 (`sm_120`) does not. On `sm_120` the kernel raises `AttributeError: 'NoneType' object has no attribute '_trait'` inside `cpasync.tma_partition`, for every dtype, head dim, and sequence length tried, while torch SDPA and bf16 matmul run fine on the same device.

The same image on a real H100 (`sm_90`, which has TMA) runs the kernel correctly — max abs error 0.00186 against SDPA. Same wheel, same flash-attn build, only the architecture differs, which isolates the cause to TMA and shows the kernel path is sound wherever TMA exists. B200 and B300 have TMA, so the datacenter path is expected to work; that remains unverified on B200/B300 silicon itself.

`import flash_attn` still succeeds, which is exactly why this went unnoticed: the golden eval only imported. It is now a real capability smoke. Details, including the A/B against the previously published image that proves this is pre-existing rather than a regression, are in the `known_gaps` block of `npa/docker/workbench/blackwell-dc-images.json`. Callers on `sm_120` should use torch SDPA.

## 2. Build, tag, and register

**Additive tags only.** Never overwrite an existing tag; this mirrors the `sm_120` rollout and the SONIC catalog rule. Encode the architectures in the tag so routing is auditable:

```
cuda13-b300-sm80-sm90-sm100-sm103-sm120-<UTC>
```

The `cuda13-b300` prefix is required — `npa/docker/workbench/check_tag_consistency.py` rejects tags outside the families declared in `npa/docker/workbench/tags.yaml`.

Reuse the existing build scripts rather than inventing a detached build. `base/cuda13-b300/build.sh` now takes the arch knobs directly:

```bash
npa/docker/workbench/base/cuda13-b300/build.sh \
  --tag "sm80-sm90-sm100-sm103-sm120-$(date -u +%Y%m%dT%H%M%SZ)" \
  --arch-list "8.0 9.0 10.0 10.3 12.0" \
  --require-archs "sm_80 sm_90 sm_100 sm_120" \
  --registry "$NPA_REGISTRY" --push
```

Other scripts and their base overrides: `genesis/build_sm120.sh --base-image`, `sim2real-build.sh` (`BASE_IMAGE`, `GENESIS_IMAGE`), `sonic/build.sh --variant baked|k8s|mujoco` (extend the existing `NPA_CUDA_ARCHITECTURES` build arg), `cosmos2-transfer/build.sh`, `lerobot/build.sh`, `lancedb/build.sh`, `groot/build.sh`.

Push new tags to both the primary (`eu-north1`) and mirror (`us-central1`) registries. Resolve the registry through `${NPA_REGISTRY}` or `npa.deploy.images`; never hardcode a registry id.

The packaging contract still applies (`npa/docker/workbench/packaging-contract.yaml`): non-root final user, `EXPOSE`/`HEALTHCHECK` per tier, no baked secrets, and a declared `redistribution` class.

After building, register the tag so the CLI, SDK, and YAML resolve it:

- `npa/pyproject.toml` → `[tool.npa.supported-tools]` (source of truth)
- `npa/docker/workbench/blackwell-dc-images.json` (datacenter Blackwell verdicts) and `npa/docker/workbench/sm120-images.json` (workstation Blackwell)
- `npa/src/npa/deploy/sonic_image_manifest.json` (`gpu_compatibility` + `gpu_selection`) if a SONIC datacenter variant is added
- GPU aliases and presets in `npa/src/npa/serverless_common/platform.py`, which `npa.cli.workbench.lerobot` now reads directly

## 3. Validation

An image is not done until it passes on real hardware.

**Wheel arch check (no GPU needed).** Catches a cu124/cu126 wheel before it costs a GPU hour:

```bash
npa/scripts/validate_blackwell_image.sh "$NPA_REGISTRY/npa-base:<tag>" --target b200
```

**Arch assertion on a real node.** Adds the device-capability check, which is the only trustworthy proof of what you actually landed on:

```bash
npa/scripts/validate_blackwell_image.sh "$NPA_REGISTRY/npa-lerobot:<tag>" --target b300 --gpu
```

Equivalent one-liner if you are already inside the container:

```bash
python -c "import torch; cap=torch.cuda.get_device_capability(); \
  print(cap, torch.cuda.get_arch_list()); \
  assert cap in {(10,0),(10,3)}, cap"   # (10,0)=B200, (10,3)=B300
```

**A real capability smoke, not a CUDA probe.** Run the image's golden/functional smoke so the custom kernels actually execute: flash-attn / natten for cosmos and lerobot-b300, `gs.init(gpu)` plus a `FrankaPickPlaceEnv` step for genesis and loop-eval, a real video-to-video transfer for cosmos2-transfer, a CLIP embed for lancedb, a detector training step for detection-training. This is what caught the `loop-eval:0.1.1` `sm_120` regression, and it is what caught the flash-attn TMA gap above — in both cases an import check passed and the first real kernel failed.

For the base image that smoke is committed as `npa/docker/workbench/base/cuda13-b300/scripts/gpu_capability_smoke.py`. To run it on an already-deployed Kubernetes GPU pool rather than provisioning a node, use `npa/scripts/blackwell-gpu-validation-job.yaml`; it runs the arch check positively for the target architecture, negatively for a different CUDA major (so a pass on the wrong GPU family cannot be mistaken for success), and then the capability smoke.

**Provisioning.** `--gpu-type b300` resolves to `gpu-b300-sxm` (presets `1gpu-24vcpu-346gb` / `8gpu-192vcpu-2768gb`, uk-south1); `--gpu-type b200` resolves to `gpu-b200-sxm` (`1gpu-20vcpu-224gb` / `8gpu-160vcpu-1792gb`, us-central1). Both platform ids and their presets were confirmed against `nebius compute platform list`. `npa/benchmark_b300_h200.sh` is a working deploy invocation. For host-mounted-driver images, use a Managed K8s pool with the NVIDIA GPU Operator.

> **Behaviour change:** the bare `b200` alias previously resolved to `gpu-b200-sxm-a`, which the tenant used for validation does not expose. It now resolves to `gpu-b200-sxm`. `gpu-b200-sxm-a` stays independently resolvable, with the same presets, for regions that offer that variant — so only callers passing the bare alias see a different platform id.

Datacenter Blackwell is also offered as `gpu-gb300` (Grace-Blackwell Ultra). That host is aarch64, so the x86_64 workbench images do not run on it — the `sm_103` GPU is the same, but the CPU architecture is not.

**Guardrails before pushing registry or docs changes:**

```bash
npa/.venv/bin/python npa/scripts/audit_workbench_image_tags.py
npa/.venv/bin/python npa/docker/workbench/check_tag_consistency.py
npa/.venv/bin/python -m pytest npa/tests/docker/test_packaging_contract.py \
  npa/tests/docker/test_blackwell_dc_manifest.py \
  npa/tests/deploy/test_public_publish.py \
  npa/tests/guardrails/test_skills_index.py -q
```

**Tear down explicitly.** Datacenter Blackwell is expensive; `npa/benchmark_b300_h200.sh destroy` shows the pattern. Confirm no clusters, jobs, or services remain.

## 4. Order of work

1. `npa-base` arch widen, then `npa-workbench-cuda-base` — they gate most of the tree.
2. The ready set: `npa-lerobot` (+ policy), `npa-lancedb`, `npa-detection-training`.
3. `npa-cosmos3-reason` rebuild on the widened base, plus a real `npa-cosmos2-transfer` transfer.
4. `npa-cosmos` (Predict2) now uses NVIDIA's complete v1.2.0 cu128/torch-2.7 wheel set. Its flash-attn and real Predict2 NATTEN module passed on B200; B300 remains blocked by Predict2 v1.0.9's compute-capability allowlist until upstream admits capability 103.
5. Genesis subtree and Isaac Lab / SONIC / GR00T: blocked. Track upstream; add headless-only datacenter variants only where the work is compute-only.
6. Update `pyproject.toml`, the manifests, CLI routing, and this page.

**Per-image definition of done:** an additive `sm_100` tag built and pushed to both registries; `get_device_capability()` matching the target on a real B200/B300 node with the real smoke passing; the tag registered and resolvable via CLI/SDK/YAML; guardrails green. Blocked items carry a tracked upstream reason, never a stub — `test_blackwell_dc_manifest.py` fails a `blocked` verdict that names no upstream.

Where each image currently stands, including what has been built and pushed versus what is still waiting on hardware, is in the `validation_evidence` block of `npa/docker/workbench/blackwell-dc-images.json`.
