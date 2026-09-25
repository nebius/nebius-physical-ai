# FlashAttention 4 on RTX PRO 6000

[Workbench docs](README.md) · [GPU compatibility matrix](image-gpu-compatibility-matrix.md)

The CUDA 13 base recipe includes upstream FA4 fixes for RTX PRO 6000 Blackwell
Server Edition (`sm_120`). Its qualification checks real attention outputs and
Q/K/V gradients. All 24 cases passed on a reserved Nebius RTX PRO 6000 on
2026-09-25 UTC; the [measured results and image identity](validation/fa4-rtx6000-20260925.json)
record the exact scope. Rebuilding this recipe does not update published base images,
derived images, or a customer's own container.

The separate [full-model rendering qualification](fa4-sdxl-validation.md)
completed 24 SDXL image generations at three resolutions with explicit FA4
UNet attention and a same-GPU SDPA comparison. It includes actual renders,
timings, output differences and CUDA kernel evidence. Performance was mixed;
it is not proof of a general FA4 speedup or customer training convergence.

## Default behavior

New builds of the `cuda13-b300` base use the updated FA4 pin by default. The
recipe already selected FA4; this change fixes that dependency set and its RTX
qualification. Its legacy `flash_attn` root shim also dispatches to FA4.

This is not a Workbench-wide backend switch. Existing published image digests
stay unchanged until their own rebuild, qualification and release. Models that
select PyTorch SDPA or install their own FA2 keep that selection. The explicit
FA4 integration below is the qualified starting point; unsupported model
features must not silently fall back or be discarded.

## Why the old guidance changed

RTX PRO 6000 is compute capability 12.0; B200 is 10.0. Both are Blackwell, but
they use different FA4 kernel paths. NVIDIA lists Tensor Memory Accelerator
(TMA) support for capability 12.0. The previous Workbench claim that RTX lacks
TMA was incorrect. See NVIDIA's [GPU capability list](https://developer.nvidia.com/cuda/gpus)
and [CUDA feature table](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#features-and-technical-specifications).

The old FA4 pin, `0409f9adcbdebff6cc19eb95f370d40e896980bc`, selected a TMA
output epilogue with an uninitialized copy atom in an SM80-derived kernel on
SM120. This matches [upstream issue 2477](https://github.com/Dao-AILab/flash-attention/issues/2477).
The updated [SM120 implementation](https://github.com/Dao-AILab/flash-attention/blob/eed1971f5132630dc296fe37601e834d4b57a248/flash_attn/cute/flash_fwd_sm120.py)
keeps its internal code path at SM80 while compiling for the resident GPU.
It also includes [backward and compile-argument fixes](https://github.com/Dao-AILab/flash-attention/pull/2671)
and [variable-length tile guards](https://github.com/Dao-AILab/flash-attention/pull/2763).

Historical Workbench runs waived this FA4 failure on RTX while validating other
operations. Those success markers did not establish working FA4. The current
smoke fails on any kernel or numerical error on every architecture. The
`--allow-no-tma` option has been removed.

## Pinned recipe and integration

Build the checked-in [`cuda13-b300` recipe](../../npa/docker/workbench/base/cuda13-b300/Dockerfile).
The directory name is historical; its Torch wheel includes `sm_120`.

| Component | Source recipe |
| --- | --- |
| CUDA | Digest-pinned CUDA 13.0.1 base |
| PyTorch | 2.13.0, cu130 |
| FA4 | `eed1971f5132630dc296fe37601e834d4b57a248` |
| CUTLASS DSL | 4.6.2, cu13 extra |
| Quack kernels | 0.6.4 |

The tested local image used driver 580.173.02 and installed FA4 as
`4.0.0b33.dev10+geed1971`. Its source commit is
`db7ddac12ff175738cf2a5fa49780197ad88490d`, and its Docker image/config ID is
`sha256:23da87a39c48f0f726a674b2e2399abb8395ade41e63d2d65e5ccac7845f347e`.
This is a local image ID, not a published registry manifest digest.

| Precision | Cases passed | Worst output relative L2 | Worst gradient relative L2 |
| --- | --- | --- | --- |
| FP16 | 12/12 | 0.000272 | 0.000318 |
| BF16 | 12/12 | 0.002169 | 0.002618 |

The run used UID 1000, a read-only root, writable scratch, dropped Linux
capabilities and no network. Native SM120 checks passed and an intentional
SM100 device assertion failed. These results do not requalify any previously
published image or another GPU architecture.

Update the FA4/CUTLASS/Quack trio together: Quack 0.6.4 requires exactly
CUTLASS DSL 4.6.2. A successful package import is not GPU qualification.

Use the explicit FA4 namespace in model integrations. The generic
`flash_attn` import can resolve FA2 in another container. The base's legacy
root-import shim does not translate arbitrary FA2 arguments into FA4 arguments.

```python
from flash_attn.cute import flash_attn_func

# q: [batch, query_tokens, query_heads, head_dim]
# k/v: [batch, key_tokens, kv_heads, head_dim]
# CUDA FP16 or BF16 tensors; use the tested SM120 options explicitly.
output = flash_attn_func(
    q, k, v, causal=True, pack_gqa=False, num_splits=1
)
```

For packed sequences, use `flash_attn.cute.flash_attn_varlen_func` with named
`cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, and `max_seqlen_k` arguments.
Preserve the caller's return-value contract when replacing an existing backend;
FA4's API and supported features differ from FA2. Do not force another compute
capability or silently change the attention backend to make a check pass.

## Qualification scope

The baked smoke covers 24 combinations: dense, grouped-query and packed
variable-length attention × FP16/BF16 × head dimension 64/128 × causal/noncausal.
Every combination compares its output and dQ/dK/dV to independently computed
FP64 attention, checking finiteness, elementwise error and relative L2 error.
Causal cross-attention uses FA's bottom-right mask alignment.

Dense/GQA cases use batch 2, 129 query tokens and 193 key tokens. GQA uses four
query heads and two KV heads. Packed cases use query lengths 17/65/129/33 and
key lengths 31/97/129/65 with four heads, exercising uneven tiles and sequence
boundaries. The controls also execute BF16 matmul and Torch SDPA.

This is a correctness qualification for those configurations. It does not
establish performance, full-model convergence, distributed scaling, or support
for every shape that upstream's API accepts. Qualify the actual model's longest
sequences, layouts and training features before deployment.

Restrictions in the pinned [upstream interface](https://github.com/Dao-AILab/flash-attention/blob/eed1971f5132630dc296fe37601e834d4b57a248/flash_attn/cute/interface.py):

- SM120 forward rejects block sparsity, paged KV and split-KV. Use `num_splits=1`.
- Use `pack_gqa=False` for this baseline; packed-GQA optimization has separate
  [upstream reports](https://github.com/Dao-AILab/flash-attention/issues/2444).
- SM120 backward rejects deterministic mode and custom score/mask modifiers.
- Head dimension 192 can exceed backward shared-memory capacity; the proposed
  [upstream guard](https://github.com/Dao-AILab/flash-attention/pull/2902) is not
  evidence that such a configuration works. Dimensions beyond 64/128, dropout,
  sliding windows, FP8 and other features are outside this qualification.

## Reproduce on a reserved RTX GPU

Build on the operator's Nebius GPU VM from a clean committed checkout. The
commands below create a local candidate, with no registry publication:

```bash
FA4_SOURCE_SHA=$(git rev-parse HEAD)
npa/docker/workbench/base/cuda13-b300/build.sh --tag "dev-${FA4_SOURCE_SHA}"
FA4_IMAGE="npa-base:cuda13-b300-dev-${FA4_SOURCE_SHA}"

mkdir -p fa4-results
docker run --rm --gpus all --network none \
  --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,exec,mode=1777 \
  -e HOME=/tmp -e CUDA_CACHE_PATH=/tmp/cuda-cache \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch-extensions \
  -v "$PWD/fa4-results:/output" \
  "$FA4_IMAGE" python /npa/gpu_capability_smoke.py \
  --expect-capability 12.0 --json-output /output/fa4-capability.json
```

The output directory must be writable by the image's UID 1000. First execution
compiles kernels and needs writable scratch space. A successful run exits zero,
prints `GPU_CAPABILITY_SMOKE_OK`, and records 24 passing cases. Any failure
exits nonzero and records the active case and error in JSON. The committed
[Kubernetes validation job](../../npa/scripts/blackwell-gpu-validation-job.yaml)
also checks the intended GPU and rejects a different CUDA major.

Keep source SHA, exact local image ID or registry digest, package versions,
driver, device capability and numerical results with each qualification. An
RTX result does not qualify the new dependency set on B200, B300 or Hopper.

## Applying this to an FA2 migration

Rebuild the workload's own container with the qualified dependency set and
explicit FA4 calls, then compare outputs, gradients and complete training/eval
runs. The base-image smoke cannot identify an unknown customer image's failure
without its package versions, traceback and model attention configuration.

Measure FA2 and FA4 on the same RTX GPU with identical shapes, precision and
batching. Separate attention latency from end-to-end step time and from
multi-node communication. Comparing B200/FA4 with RTX/FA2 changes both hardware
and backend, so it cannot establish the speedup this update will deliver.
