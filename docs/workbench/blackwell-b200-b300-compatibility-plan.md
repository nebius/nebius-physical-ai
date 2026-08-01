# Blackwell B200 / B300 Image Compatibility — Build Context & Runbook

**Status:** planning / handoff. **Audience:** the agent executing the
"max compatibility between workbench images + Nebius GPU types" effort.

This document is the execution context for making every Workbench container
image deployable on **B200 (`sm_100`)** and **B300 / Blackwell Ultra
(`sm_103`)** *where reasonable*. It records the current per-image state, the
compatibility rules that govern what "reasonable" means, the exact build/tag/
validate mechanics used in this repo, and a per-image plan with verdicts and
blockers.

Read first, then work top-down: `npa-base` and `npa-workbench-cuda-base` gate
most of the tree, so building and validating them first unblocks their
children.

Companion references:

- Current audit + CUDA/GPU matrix: this doc's tables below.
- `docs/nvidia-platform-architecture-coverage.md` — the NVIDIA x86_64 vs
  aarch64 CUDA split that gates the vendor-pinned tools.
- `docs/workbench/sm120-image-catalog.md` — the existing RTX PRO 6000
  (`sm_120`) validation catalog; B200/B300 additions follow the same pattern.
- `npa/docker/workbench/sm120-images.json` and
  `npa/src/npa/deploy/sonic_image_manifest.json` — machine-readable image
  manifests to extend.
- `npa/src/npa/deploy/images.py` + `npa/pyproject.toml`
  `[tool.npa.supported-tools]` — the tool→tag registry to bump after a build.

---

## 1. The compatibility model (read this before changing any Dockerfile)

"Blackwell" is a marketing family, not a single CUDA target. The images must be
built for the correct **compute capability (SM arch)**:

| GPU | Family | Compute capability | SM target | Nebius platform string |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 Blackwell | Blackwell workstation (GB20x) | 12.0 | `sm_120` | `gpu-rtx6000` |
| B200 | Blackwell datacenter (GB100) | 10.0 | `sm_100` | `gpu-b200-sxm` (verify in console) |
| B300 (Blackwell Ultra) | Blackwell datacenter (GB300) | 10.3 | `sm_103` | `gpu-b300-sxm` (confirmed, `uk-south1`) |
| H100 / H200 | Hopper | 9.0 | `sm_90` | `gpu-h100-sxm` / `gpu-h200-sxm` |
| L40S | Ada | 8.9 | `sm_89` | `gpu-l40s-a` / `gpu-l40s-d` |

Rules that decide feasibility:

1. **SASS is per-arch, no cross-major compatibility.** `sm_120` (major 12) and
   `sm_100`/`sm_103` (major 10) binaries are mutually incompatible. **Validating
   on RTX PRO 6000 (`sm_120`) does NOT prove B200/B300 (`sm_100`/`sm_103`).**
2. **Minor-version forward compat within a major holds.** `sm_100` SASS runs on
   `sm_103` (10.0 → 10.3), but not the reverse. So **a build that covers B200
   (`sm_100`) generally also covers B300 (`sm_103`)**; prioritize `sm_100` and
   treat B300 as the forward-compat beneficiary, then confirm on B300 hardware.
3. **Two independent knobs decide arch coverage:**
   - **The prebuilt torch wheel** ships a fixed fat-binary arch set you cannot
     change with an env var — you must select a wheel/index that includes the
     target arch. Check with
     `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"`.
     `cu128` and `cu130` wheels include `sm_100` and `sm_120`; `cu124`/`cu126`
     top out at `sm_90` (Hopper) and are **not** Blackwell-capable.
   - **Source-compiled CUDA extensions** (flash-attn built from source,
     Taichi/Genesis kernels, natten, custom ops) obey `TORCH_CUDA_ARCH_LIST`.
     Set it to include `10.0` (B200), `10.3` (B300), and `12.0` (RTX PRO 6000)
     — e.g. `TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 10.3 12.0"` — or the extension
     will `no kernel image is available for execution on the device` on the
     uncovered arch.
     - Note: `npa-base:cuda13-b300` currently pins `TORCH_CUDA_ARCH_LIST=10.3`
       and builds flash-attn-4 (CuTe). CuTe DSL JIT-compiles kernels at runtime,
       which is why the same Dockerfile also runs on `sm_120`. **Verify** the
       JIT assumption on real B200/B300 before relying on it; if any AOT
       extension is added, widen the arch list as above.
4. **RT cores.** Datacenter Blackwell (B200/B300), like H100/H200, has **no RT
   cores**. Do **not** route Isaac Lab / SONIC *rendering* to B200/B300 —
   headless/state-based training is fine, RT-core rendering is not (keep that on
   L40S / RTX PRO 6000). See `.claude/skills/atomic/gpu-selection/SKILL.md`.
5. **Datacenter-only features** (NVFP4, 2nd-gen Transformer Engine, NVLink,
   MIG) are exercised only on real B200/B300 — a workstation-Blackwell smoke
   never touches them. Live validation must run on the actual datacenter part.
6. **Driver/toolkit floor.** B200/B300 need a recent driver (R570+) and CUDA
   12.8+/13.0. On Nebius Managed Kubernetes this is the GPU-Operator-mounted
   host driver; on VM/serverless it is the compute host driver.

---

## 2. Current state → B200/B300 plan (per image)

Legend for **Verdict**:
`READY` = cu128/cu130 base already Blackwell-capable, just needs an additive
rebuild + live B200/B300 validation; `PORT` = needs a CUDA-base/torch bump and
possibly extension arch flags; `BLOCKED` = gated on an upstream that does not
yet ship the target arch (vendor-paced — do not fake it); `N/A` = CPU/GPU-
agnostic, already runs on a B200/B300 host without GPU-arch work.

### 2a. Base images (build these first)

| Image | Base / CUDA / torch | Current arch | Verdict | Action |
| --- | --- | --- | --- | --- |
| `npa-base` (`base/cuda13-b300`) | `nvidia/cuda:13.0.1` / torch 2.9.0+cu130 | `TORCH_CUDA_ARCH_LIST=10.3` (+ cu130 wheel archs) | READY→PORT | Widen `TORCH_CUDA_ARCH_LIST` to `"8.0 9.0 10.0 10.3 12.0"`; confirm `get_arch_list()` includes `sm_100`; push additive tag (see §3). This base already targets B300 by design. |
| `npa-workbench-cuda-base` | `pytorch/pytorch:2.12.1-cuda13.0` (cu130) | cu130 wheel archs | READY | Confirm `get_arch_list()` shows `sm_100`/`sm_120`; no source extensions here. Validate `npa-lancedb` + `npa-detection-training` on B200/B300. |

### 2b. Ready / low-effort (cu128/cu130 already Blackwell-capable — validate + tag)

| Image | Base / CUDA | Verdict | Notes |
| --- | --- | --- | --- |
| `npa-lerobot` (`Dockerfile.b300`) | `npa-base` cu130, `target_gpu=b300-sm_103` | READY | **B300 already Tier-1**; add B200 (`sm_100`) validation. Highest-confidence path. |
| `npa-lerobot` (default `Dockerfile`) | cu124 base, but pip-upgrades torch 2.12.1 (bundled cu128) | READY | Effective arch = torch wheel (Blackwell-capable). Validate on B200/B300; prefer the b300 Dockerfile for pinned reproducibility. |
| `npa-lerobot-policy` | cu124 base + torch 2.12.1 wheel | READY | Same as default lerobot; validate. |
| `npa-cuda-base` children: `npa-lancedb`, `npa-detection-training` | cu130 (2.12.1) | READY | No custom CUDA build; CLIP / COCO-metric ops use stock torch/torchvision. Validate. |
| `npa-cosmos3-reason` | `npa-base` cu130 sm120 | READY | Inherits arch from `npa-base`; rebuild after the base arch widen, then validate. |
| `npa-cosmos2-transfer` | upstream transfer2.5 + `uv sync --extra=cu128` | READY | Already cu128 (Blackwell-capable); flash-attn is cp310 cu128. Validate real video→video transfer on B200/B300. |

### 2c. Port (bump CUDA base / torch off Hopper-capped wheels)

| Image | Current | Blocker to clear | Verdict |
| --- | --- | --- | --- |
| `npa-cosmos` (predict2) | `nvidia/cuda:12.6.3` / torch 2.6.0+**cu126** + flash-attn/natten cu126 wheels | Needs cu128 torch + cu128 Blackwell wheels from `nvidia-cosmos/cosmos-dependencies` (flash-attn, natten). Coverage doc notes a Blackwell nightly exists. | PORT (upstream-gated) |
| `npa-genesis` (default) | `nvidia/cuda:12.4.1` / torch 2.6.0+**cu124** | Move to cu130 (the `Dockerfile.sm120` path already does) **and** Taichi arch — see BLOCKED below. | PORT→BLOCKED |

### 2d. Blocked / vendor-paced (do not stub; track upstream)

| Image(s) | Why blocked for B300 (`sm_103`) | Notes |
| --- | --- | --- |
| `npa-genesis` (sm120) and its children: `npa-envgen`, `npa-reference-policy`, `npa-explore-policy`, `npa-loop-eval`, `npa-lerobot-vlm-rl`, the sim2real chain | Genesis physics kernels run on **Taichi**, which is **upstream-blocked on `sm_103`** (see `docs/nvidia-platform-architecture-coverage.md`). `sm_120` support exists (RTX PRO 6000 works); `sm_100`/`sm_103` do not until Taichi ships them. | Re-test Taichi `sm_100` (B200) specifically — it may land before `sm_103`. If B200 works, B300 follows by minor-version forward compat once Taichi accepts `sm_103`. The whole Genesis-derived sim2real subtree inherits this. |
| `npa-isaac-lab`, `npa-sonic` (all variants), `npa-groot` | NVIDIA vendor stack pinned to **x86_64 CUDA 12.8** (Isaac Sim/Lab, GR00T). B200/B300 support is gated on NVIDIA's x86_64 CUDA-13 alignment. Also **Omniverse-restricted** (never publish prebuilt publicly). | *Rendering* must not target B200/B300 (no RT cores) regardless. *Headless* SONIC fine-tune / GR00T finetune are compute-only and cu128 already carries `sm_100`; a headless-only B200 variant may be reasonable — validate compute-only first. Keep RT-core render on L40S/RTX PRO 6000. |

### 2e. N/A — CPU / GPU-agnostic (already deployable on a B200/B300 host)

`npa-fiftyone`, `npa-retargeting`, `npa-rerun-viewer`, `npa-lichtblick`,
`npa-foxglove-embed`, `npa-sonic-export` (CPU torch). No GPU-arch work; they run
on any node. Only confirm they schedule on a B200/B300 node pool (they will not
use the GPU).

---

## 3. Build, tag, and registry conventions (follow exactly)

- **Additive tags only. Never overwrite an existing tag** (mirrors the sm_120
  rollout and the SONIC catalog rule). Suggested convention, matching
  `cuda13-b300-sm80-sm90-sm120-<UTC>`:
  - base: `cuda13-b300-sm80-sm90-sm100-sm103-sm120-<UTC>` (widened arch list),
  - tools: `<version>-blackwell-dc-<UTC>` or reuse the tool's existing scheme
    with an explicit `-sm100`/`-sm103` marker so routing is auditable.
- **Build scripts already exist — reuse them** (they take `--registry`,
  `--tag`, `--push`, and base-image overrides):
  - `npa/docker/workbench/base/cuda13-b300/build.sh` (accepts
    `CUDA_BASE_TAG`, `FLASH_ATTN_COMMIT` via env; add a `TORCH_CUDA_ARCH_LIST`
    build-arg pass-through when widening arch).
  - `npa/docker/workbench/genesis/build_sm120.sh` (pattern for an additive
    Genesis arch build).
  - `npa/docker/workbench/sim2real-build.sh` (builds the Genesis-derived
    subtree from `BASE_IMAGE` + `GENESIS_IMAGE`; set both to the new tags).
  - `npa/docker/workbench/sonic/build.sh --variant baked|k8s|mujoco`
    (`NPA_CUDA_ARCHITECTURES` build-arg already exists — extend it).
  - `npa/docker/workbench/cosmos2-transfer/build.sh`,
    `npa/docker/workbench/lerobot/build.sh`, `lancedb/build.sh`,
    `groot/build.sh`.
- **Registries.** Primary `cr.eu-north1.nebius.cloud/<id>`, mirror
  `cr.us-central1.nebius.cloud/<id>` (see `npa/src/npa/deploy/images.py`). Every
  tool image is mirrored to both for region-agnostic pulls; push new tags to
  both. Do **not** hardcode a registry id — use `${NPA_REGISTRY}`.
- **Packaging contract.** New/edited Dockerfiles must satisfy
  `npa/docker/workbench/packaging-contract.yaml` (non-root final user,
  EXPOSE/HEALTHCHECK per tier, no baked secrets). Isaac-Sim-bearing images stay
  `redistribution: restricted` — **never** publish `isaac-lab`/`sonic`/`groot`
  to a public/anonymous registry.
- **After building**, update the registry of record so CLI/SDK/YAML resolve the
  new tags:
  - `npa/pyproject.toml` → `[tool.npa.supported-tools]` (source of truth read by
    `npa.deploy.images.supported_tool_version`),
  - `npa/docker/workbench/sm120-images.json` (add B200/B300 entries or a sibling
    manifest, e.g. `blackwell-dc-images.json`),
  - `npa/src/npa/deploy/sonic_image_manifest.json` `gpu_compatibility` +
    `gpu_selection` if a SONIC datacenter-Blackwell variant is added,
  - the SONIC/GPU aliases in `npa.deploy.images` and
    `npa/src/npa/cli/workbench/lerobot.py` (`_lerobot_gpu_platform`,
    preset table) if a new `gpu-b200-sxm`/`gpu-b300-sxm` routing is introduced.

---

## 4. Validation (a build is not "done" until it passes on real hardware)

Per-arch SASS means **you must validate on the actual B200 and/or B300 node** —
a green RTX PRO 6000 run is not evidence. Follow the sm_120 catalog's live-smoke
shape (`docs/workbench/sm120-image-catalog.md`):

1. **Provision** a Nebius node/pool for the target: VM/serverless via
   `--gpu-type b300` (→ `gpu-b300-sxm`, preset `1gpu-24vcpu-346gb` /
   `8gpu-192vcpu-2768gb`, region `uk-south1`) or B200 (`gpu-b200-sxm` — confirm
   the platform string + region in the Nebius console), or a Managed K8s pool
   with the NVIDIA GPU Operator for the host-mounted-driver images. See
   `npa/benchmark_b300_h200.sh` for the working B300 deploy invocation.
2. **Arch assertion** on every torch-bearing image:
   ```bash
   python -c "import torch; cap=torch.cuda.get_device_capability(); \
     print(cap, torch.cuda.get_arch_list()); \
     assert cap in {(10,0),(10,3)}, cap"  # (10,0)=B200, (10,3)=B300
   ```
3. **Real capability smoke, not a CUDA probe.** Run each image's existing golden
   eval / functional smoke so custom kernels actually execute:
   flash-attn/natten (cosmos, lerobot b300), a `FrankaPickPlaceEnv` step +
   `gs.init(gpu)` (genesis/loop-eval), a real video→video transfer
   (cosmos2-transfer), CLIP embed (lancedb), detector train step
   (detection-training). This is what caught the `loop-eval:0.1.1` `sm_120`
   regression.
4. **Guardrails/CI** (run before pushing docs/registry changes):
   ```bash
   npa/.venv/bin/python npa/scripts/audit_workbench_image_tags.py
   npa/.venv/bin/python -m pytest npa/tests/docker/test_packaging_contract.py \
     npa/tests/deploy/test_public_publish.py \
     npa/tests/guardrails/test_skills_index.py -q
   ```
5. **Teardown.** Explicitly destroy the SkyPilot job cluster / VMs and confirm
   no clusters, managed jobs, or services remain (datacenter Blackwell is
   expensive). `npa/benchmark_b300_h200.sh destroy` shows the pattern.

---

## 5. Suggested execution order

1. `npa-base` (widen arch) + `npa-workbench-cuda-base` → validate on B200/B300.
2. `npa-lerobot` (b300 already done → add B200), `npa-lerobot-policy`,
   `npa-lancedb`, `npa-detection-training` — the READY set; fast wins.
3. `npa-cosmos3-reason` (rebuild on new base), `npa-cosmos2-transfer` (validate).
4. `npa-cosmos` (predict2) — PORT to cu128 once upstream Blackwell wheels are
   confirmed.
5. Genesis subtree + Isaac/SONIC/GR00T — BLOCKED: file/track upstream (Taichi
   `sm_100`/`sm_103`; NVIDIA x86_64 CUDA-13 Isaac/GR00T). Add headless-only
   datacenter variants only where compute-only and cu128/cu130 already carry
   `sm_100`. Do not stub.
6. Update `pyproject.toml`, manifests, and CLI routing; refresh this doc's
   verdict tables and `docs/nvidia-platform-architecture-coverage.md`.

## 6. Definition of done (per image)

- Additive tag built for `sm_100` (and `sm_103` confirmed by forward compat or
  direct build), pushed to both registries.
- `get_device_capability()` == target on a real B200/B300 node, and the image's
  real capability smoke passes there.
- Tag registered in `pyproject.toml` + the relevant manifest; CLI/SDK/YAML
  routing resolves it; guardrail tests green.
- BLOCKED items have a tracked upstream reason recorded here — not a stub.
