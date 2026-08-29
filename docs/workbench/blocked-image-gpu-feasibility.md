# Can the blocked images support every Nebius GPU?

Eight published images carry at least one `blocked` cell in the
[image ↔ Nebius GPU compatibility matrix](image-gpu-compatibility-matrix.md):
`npa-content-agents`, `npa-cosmos`, `npa-cosmos2-transfer`, `npa-cosmos3-serving`,
`npa-groot`, `npa-isaac-lab`, `npa-leisaac`, and `npa-sonic`. This page evaluates,
per image, whether that can be closed.

A ninth row, `npa-sonic-mujoco`, still shows blocked cells on B200 and B300 in the
matrix as written, but that is stale rather than substantive: its published digest
has a recorded B200 rollout, and the cells are reconciled against that evidence in
[#325](https://github.com/nebius/nebius-physical-ai/pull/325). It is treated as
unblocked below.

The short answer is that "blocked" covers four different situations that need
different remedies, and only some of them can be closed at all:

| Kind of block | Remedy | Images |
| --- | --- | --- |
| **Physical.** The workload rasterizes, which needs RT cores. H100, H200, B200, and B300 have none. | None. No software change adds RT cores. | `npa-content-agents` and `npa-leisaac` (whole service), the render paths of `npa-isaac-lab` and `npa-sonic` |
| **Software gate.** The kernels are present; a version check refuses to dispatch on the part. | Widen the gate upstream or patch it, then prove the kernel numerically on each part. | `npa-cosmos` |
| **Toolchain or validation.** The image needs a different CUDA build, or it has simply never run there. | Rebuild on the other extra, or spend the GPU hour. | `npa-cosmos2-transfer`, `npa-groot`, the headless paths of `npa-isaac-lab` and `npa-sonic` |
| **Sizing.** The architecture is fine; the part does not have the memory the pinned configuration needs. | A different parallel decomposition, which is a serving-config decision. | `npa-cosmos3-serving` on L40S |

Two constraints bound every answer below. Rasterized rendering cannot move to a
datacenter part, and no published image runs on aarch64 `gpu-gb300`, because all
30 are `linux/amd64`. "Every Nebius GPU" therefore means the five x86_64
platforms: L40S, H100, H200, RTX PRO 6000, B200, and B300.

## Measured dependency coverage

A torch wheel's arch set is not the whole story. Separately installed CUDA
extensions carry their own, and that is where the `npa-cosmos` question actually
lives. These were read out of the pinned artifacts with
[`npa/scripts/measure_extension_arches.py`](../../npa/scripts/measure_extension_arches.py)
on 2026-08-22 — no GPU and no container required:

| Pinned artifact | Used by | Measured SASS | PTX |
| --- | --- | --- | --- |
| `natten 0.21.0+cu128.torch27` | `npa-cosmos` | `sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120` | none |
| `flash_attn 2.7.3+cu128.torch27` | `npa-cosmos` | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | none |
| `flash_attn 2.7.4.post1+cu12torch2.7` | `npa-groot` | `sm_80 sm_90 sm_100 sm_120` | none |

Read with the compatibility rules from the matrix: SASS does not cross a CUDA
major, but within a major it is forward compatible, so `sm_86` covers L40S
(`sm_89`) and `sm_100` covers B300 (`sm_103`). None of the three carries PTX, so
nothing here depends on driver JIT. Every one of these artifacts can reach all
five platforms.

Coverage is necessary, never sufficient. flash-attn-4 ships `sm_120` SASS and
still raises on `sm_120` because its epilogue needs TMA, which is the finding
recorded in the matrix. A cell moves on a capability run, not on a wheel scan.

## `npa-cosmos2-transfer` — yes, and it is the most contained

Blocked only on B300, because CUDA 12.8 NVRTC rejects a runtime-generated
capability-10.3 kernel during `Control2WorldInference` construction, after the
gated downloads and before inference. The same image completed two real 35-step
transfers on B200.

The fix does not need a new upstream release. At the already-pinned commit
`67d56b7d550a3911024a32dc23ae0bae5258e633`, upstream's `pyproject.toml` declares
both extras and marks them mutually exclusive:

```toml
cu128 = ["cosmos-oss[cu128_torch27]"]
cu130 = ["cosmos-oss[cu130_torch29]"]
```

The Dockerfile hardcodes `uv sync --locked --no-dev --no-editable --extra=cu128`.
Moving to `cu130` is a build change on the same source pin, so it does not
reopen the redistribution question about which upstream bytes are baked.

It is a build change, not a one-line change. It has to carry the digest-pinned
CUDA 13 devel and runtime bases, the `libcudart` symlink and `LD_LIBRARY_PATH`
block, the SAM2 source substitution, the hash-pinned security override and
`npa-cli` requirement files, both `assert_no_forbidden_payload.sh` passes, and
the tests that assert `--extra=cu128` by name (`test_cosmos2_transfer_image.py`,
`test_cosmos_oss_images.py`). Then B300 needs the real depth-conditioned
Video2Video smoke: an arch check would have passed the run that failed, since
the image already reported `sm_100` and capability `(10, 3)` before dying in
NVRTC.

## `npa-groot` — yes for the headless paths, and it needs no build at all

Blocked on B200 and B300 with the reason recorded as the NVIDIA x86_64 CUDA 12.8
pin. The measurements above say that is not a kernel gap. GR00T's pinned lock at
commit `3df8b3825d67f755e69141446f4315f281b9b7e6` takes `torch==2.7.1` from the
`download.pytorch.org/whl/cu128` index and `flash-attn==2.7.4.post1` from the
upstream release wheel, and that wheel carries `sm_100` and `sm_120` natively,
reaching B300 by forward compatibility and L40S from `sm_80`.

GR00T inference and finetuning also run entirely in `GROOT_VENV` with no Isaac
involvement, and the finetune golden eval already runs headless on H100. So the
blocked cells here record "never validated", which is what the manifest's own
note anticipated. The cheapest close of the six is a GR00T finetune step on a
B200 node with no rebuild.

One caveat keeps this short of a whole-image claim: `groot eval --sim` delegates
to Isaac Lab and inherits every Isaac constraint below. The honest target is
headless inference and finetune on all five platforms, with the sim path
following Isaac.

## `npa-cosmos` — possible, but it is a decision about an upstream gate

Blocked on L40S, RTX PRO 6000, and B300. The gate is eleven lines of Python in
the pinned release, `cosmos_predict2/module/neighborhood_attn.py`:

```python
# Only allowing on Hopper and Blackwell for now, since Hopper FNA and
# Blackwell FNA can deliver excellent speedup over SOL baselines.
# Other architectures will be enabled as soon as good kernels for them
# land in NATTEN.
ALLOWED_COMPUTE_CAPS = [90, 100]
```

It is enforced inside `NeighborhoodAttention.forward`, so it fires only when a
sparse-attention model variant runs. Three things follow.

First, the kernels upstream is waiting for have landed, and they are already in
the image. The pinned NATTEN wheel measurably carries `sm_89`, `sm_120`, and
`sm_100` SASS, and NATTEN's changelog adds SM120 in 0.21.0 and native SM103 in
0.21.1. Nothing about these cells is a missing kernel.

Second, the blocked cells are narrower than the row implies. The image's own
functional smoke starts the server and generates from
`nvidia/Cosmos-1.0-Diffusion-7B-Text2World`, and its other workbench role is the
default self-hosted VLM for `workbench.vlm_eval.*` stages. Neither path
constructs `NeighborhoodAttention`. The recorded blocked and verified cells alike
came from kernel validation runs that exercised the NATTEN module deliberately.
Distinguishing "sparse-attention variants blocked" from "image blocked" is worth
doing regardless of whether the gate ever moves.

Third, closing the cells properly is an upstream question, not a patch question.
The repository does patch pinned upstreams elsewhere — `npa-leisaac` applies
`.patch` files, `npa-wan2-2` uses `sed -i`, `npa-alpamayo2-super` carries a
dataset-revision patch — so the mechanism exists. But this particular gate
encodes which architectures NVIDIA validated, and stepping outside it puts NPA
on a path upstream has not qualified. If it is widened, the validation owed is a
real NATTEN forward checked numerically against the base attention op on L40S,
RTX PRO 6000, and B300, not an import: the fallback path selects the
capability-80 performance config for any unlisted architecture, and NATTEN
0.21.0 predates the 0.21.7 fix for an SM120/SM121 config bug in exactly that
frontend. Expect that to surface. A Predict2 version bump that ships the wider
allowlist is the supported route; B300 additionally wants native `sm_103`, which
means a dependency release built from NATTEN ≥ 0.21.1. L40S also needs its VRAM
headroom checked against the served model, which is a separate question from
architecture.

## `npa-cosmos3-serving` — yes in principle, but it is a serving-config decision

Blocked only on L40S, and uniquely among these eight the reason is neither
architecture nor an upstream gate. The pinned strategy is an 8-GPU decomposition
over a checkpoint that is about 124 GB on disk plus 17 GB of guardrail weights,
roughly 145 GB to hold. Its own runbook already treats the 8-GPU shape as a hard
precondition, failing with `the pinned parallel config needs 8 GPUs, found N`.

So the L40S cell is a memory floor, not an `sm_89` problem: the wheel closure runs
there, the model does not fit the way this configuration shards it. Closing it
means qualifying a different decomposition against 48 GB cards and accepting
whatever it costs in latency, which is a product decision about whether an L40S
serving tier is wanted at all — not a packaging fix. Everything else about this
image is already strong: B200 and 8xH200 both have real generation runs behind
them, and RTX PRO 6000 and B300 are supported at the same 8-GPU shape.

## `npa-sonic` — partly, and it is queued behind an unrelated failure

Blocked on B200 and B300. The policy layer is not what blocks it:
`npa.workbench.sonic.routing` already classifies `train`, `finetune`, and
`mujoco-eval` as headless work that may run on datacenter parts, and restricts
only `isaac-render` to RT-core GPUs. The published Kubernetes variant is already
built for `sm80,sm90,sm100,sm103,sm120` and measures `2.9.0+cu130` with `sm_100`
and `sm_120`.

Image resolution used to block it in the worst possible way. `gpu_selection` maps
a B200 target to the MuJoCo variant, and `sonic_image_variant_for_gpu()` keyed on
the GPU alone, so asking for a fine-tune on B200 quietly resolved the *evaluation*
runtime — a wrong image rather than a missing one. That is fixed here: every
variant declares the workloads it serves, resolution intersects the GPU rule with
the requested workload, and a fine-tune on a datacenter target now fails with a
message naming the capability gap instead of substituting a different image.
`DEFAULT_GPU_TARGET` stays on RTX PRO 6000, because no published variant can
fine-tune on a datacenter part yet.

What still blocks the cell is an unrelated bug. SONIC's real fine-tune does not
currently pass anywhere: on RTX PRO 6000, cold and warm, it reaches Isaac
environment construction and then fails in Isaac's runtime-fetched URDF
extension while opening the temporary G1 pelvis USD layer, before a learning step
or a checkpoint. That failure has no recorded architecture dependence, and it is
the smoke a new cell would have to pass. Fixing it comes first; only then is a
compute-only datacenter variant worth building. Rendering stays on RT-core parts
permanently, so `npa-sonic` can at best reach "supported (headless)" on B200 and
B300.

`npa-sonic-mujoco` was the predicted tractable subset — MuJoCo evaluation touches
no Isaac at all — and it has since been published and validated exactly that way:
its `0.2.0-runtime` digest ran a real Unitree G1 rollout on B200, EGL headless
physics with no Omniverse and no RT-core rendering. It is no longer blocked, which
is the strongest available evidence that SONIC's datacenter problem is Isaac
rather than the architecture.

## `npa-isaac-lab` — no; headless is the ceiling

Blocked on B200 and B300 for two independent reasons, and only one of them can
move.

Rendering cannot. `workbench.isaac_lab.capture_frames` launches with
`enable_cameras=True`, and the Sim2Real Isaac backend rasterizes; both need RT
cores that datacenter Blackwell does not have. The Sim2Real GPU fallback encodes
this by admitting only L40S and RTX PRO 6000, and that is correct rather than
conservative.

The CLI used to encode more than that. A single RT-core gate covered every Isaac
Lab submit, so headless state-based training was refused on H100/H200/B200 even
though nothing in it rasterizes. This change splits the gate by workload the way
SONIC's routing already did: headless training may target a datacenter part,
rendering may not, and a task id that declares camera or rendered observations
(`Isaac-Cartpole-RGB-Camera-Direct-v0` and friends) is classified as rendering
rather than trusted to a `--headless` argument. That last rule matters because
NVIDIA's own B200 reports show the camera path deadlocking when the PhysX GPU
pipeline falls back to software while the render pipeline still waits on
GPU-backed physics. `npa workbench isaac-lab deploy` still requires RT cores,
because a deployed workbench is the render surface.

Whether headless RL then *works* on a datacenter part is still a vendor and
driver question. The manifest
attributes the block to the x86_64 CUDA 12.8 pin, and the measurements above
show cu128 wheels are not the constraint — they carry `sm_100` and `sm_120`.
NVIDIA's own forum thread records headless Isaac Lab 2.3 running on a B200 DGX
once the pre-built Isaac Lab container was used, with the camera path deadlocking
when the PhysX GPU pipeline fell back to software. Isaac Sim 5.1 (Kit 107.3.3) is
validated against the R580 driver branch, and R590 breaks the RTX renderer
outright, so any attempt has to pin the node's driver branch too.

Isaac is also the one family whose coverage cannot be measured offline. The
pinned payload is not in the wheel: `isaacsim_extscache_physics-5.1.0.0` for
x86_64 is a 17 KB stub containing a licence and metadata, with the extension
content fetched by Kit at first run. Whether PhysX has GPU kernels for `sm_100`
is answerable only by running it there.

So the achievable end state is `supported (headless)` on B200 and B300 — never a
full-capability cell, because the image's headline capability is rendering.

## `npa-leisaac` — no; two RT-core parts is the ceiling

The entire service is Isaac Sim rasterized WebRTC viewport streaming. There is no
headless entrypoint in the repository: the container's `session_server.py`
entrypoint runs live simulation with WebRTC signalling and video, and the golden
eval waits on the signalling port. RT cores are not an optimization here, they
are the product. H100, H200, B200, and B300 can never host it, and the Kubernetes
launcher's hard-selected `NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition` affinity
is a correct encoding of that.

The one cell that can legitimately change is L40S, which has RT cores and is
currently recorded as not routed or validated by the launcher. Closing it means
broadening the hard-coded GPU product and node affinity, then running the real
`LeIsaac-SO101-PickOrange-v0` teleoperation smoke with WebRTC on an L40S node.
That would take `npa-leisaac` to its ceiling of two platforms out of six.

## `npa-content-agents` — no; the same physical wall as LeIsaac

Blocked on H100, H200, B200, and B300, and this one is not close to the line. The
accepted workflow ends in OVRTX path tracing, and the recorded run is a one-GPU
RTX PRO 6000 pass with 6 material, 6 physics, and 1 validation render. Path
tracing is the deliverable, so a datacenter part cannot substitute for it at any
quality setting.

Its L40S cell already reads supported, so its ceiling — two of the six platforms
— is one validation run away rather than a build away. Closing it means running
the real Material Agent, Physics Agent, OVRTX, and Validation Agent path on an
L40S node. The hosted-VLM portion of the workflow is unaffected by GPU choice.

## What this evaluation does not change

Two routing fixes land with this evaluation, because they were defects rather
than judgments: SONIC image resolution no longer substitutes a variant with a
different capability, and Isaac Lab no longer refuses headless training on a
datacenter GPU. Neither claims a new GPU result.

No cell moves on the strength of this evaluation, and no verdict in
`npa/docker/workbench/blackwell-dc-images.json` changes. Wheel-arch measurement
is a screen that cheaply rules architectures out; the repository's rule that a
cell needs a real capability run on the real part still decides what is written
down. Two cells did move recently, but on recorded runs rather than on analysis:
`npa-sonic-mujoco` on B200 and `npa-cosmos3-serving` on B200 were reconciled
against evidence their publication had already produced.

What the measurements change is what is worth attempting and in what order: a
`cu130` rebuild for Cosmos Transfer, a validation run for GR00T, an upstream
decision for Cosmos Predict2, a serving-shape decision for Cosmos3-Super on L40S,
a bug fix before SONIC, an L40S run to finish Content Agents, and no expectation
at all of rendering on a datacenter part.

## Reproducing the measurements

```bash
# Dependency arch coverage from the pinned release artifacts, no GPU needed.
curl -sLO "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.2.0/natten-0.21.0%2Bcu128.torch27-cp310-cp310-linux_x86_64.whl"
npa/.venv/bin/python npa/scripts/measure_extension_arches.py \
  natten-0.21.0+cu128.torch27-cp310-cp310-linux_x86_64.whl

# Or gate a whole site-packages tree inside a built image.
npa/.venv/bin/python npa/scripts/measure_extension_arches.py \
  /opt/cosmos/venv/lib/python3.10/site-packages --require sm_100

# The torch wheel's own arch set, and the on-GPU capability check.
npa/scripts/validate_blackwell_image.sh "$NPA_REGISTRY/npa-cosmos:<tag>" --target b200
```
