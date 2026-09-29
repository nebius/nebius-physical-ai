# Compare FA2 and FA4 on RTX PRO 6000

[RTX FA4 application guide](rtx6000-fa4.md) · [Workbench guides](README.md)

Use separate FA2 and FA4 images on the same Nebius RTX PRO 6000 GPU. The images
share one CUDA/PyTorch recipe and the same FlashAttention source revision. The
FA2 variant compiles the standalone CUDA extension for SM120; the FA4 variant
installs the CuTe implementation and guarded root import adapter. Installing both
into one Python environment would make the root `flash_attn` import ambiguous.

**Status:** this is a comparison procedure and experimental tuning harness.
The new FA2 variant and tile candidates still require a completed RTX GPU run.
The existing [SDXL evidence](../fa4-sdxl-validation.md) compares FA4 with PyTorch
SDPA, not standalone FA2. Do not use those numbers as an FA2 comparison or claim
that these tile candidates improve performance before measuring them.

## Build the two base images

On the Linux x86-64 GPU build host, from a committed Workbench checkout:

```bash
ATTENTION_SOURCE_SHA="$(git rev-parse HEAD)"
FA4_IMAGE="npa-base:cuda13-blackwell-dev-${ATTENTION_SOURCE_SHA}"
FA2_IMAGE="npa-base:cuda13-blackwell-fa2-dev-${ATTENTION_SOURCE_SHA}"
npa/docker/workbench/base/cuda13-blackwell/build.sh --tag "dev-${ATTENTION_SOURCE_SHA}"
npa/docker/workbench/base/cuda13-blackwell/build.sh --tag "dev-${ATTENTION_SOURCE_SHA}" \
  --attention-backend fa2
docker image inspect "$FA4_IMAGE" "$FA2_IMAGE" --format '{{.Id}} {{json .RepoTags}}'
```

Both are local images until explicitly published. The FA4 build remains the
default. The FA2 variant's `FA2_CUDA_ARCHS=120` default is specific to the RTX
comparison; it does not qualify that extension on B200/B300. The legacy entrypoint
also accepts `--attention-backend fa2`, with the separate
`cuda13-b300-fa2-<suffix>` compatibility tag. Existing FA4 tags are never reused
for FA2.

FA2 compilation can require substantial host RAM. On a smaller build host,
prefix the FA2 build command with `FA2_NVCC_THREADS=1` to reduce compiler
parallelism from its default of four. This does not change runtime kernels or
benchmark settings. GPU validation still requires the physical RTX GPU even
when the image is compiled on a CPU host.

## Qualify before timing

Create an output directory writable by the image's UID 1000. Run each image's
baked output/gradient checks on the idle GPU:

```bash
mkdir -p results/attention-comparison
for image in "$FA2_IMAGE" "$FA4_IMAGE"; do
  docker run --rm --gpus device=0 "$image" \
    python /npa/gpu_capability_smoke.py --expect-capability 12.0
done
```

The checker selects the baked backend, checks dense/GQA/variable-length
attention, and compares outputs and gradients with an independent FP64
reference. Imports alone do not pass this qualification.

The kernel benchmark below additionally checks its actual timed shapes against
PyTorch's **math** attention implementation before timing. It records forward
latency for eight SDXL attention shapes and forward plus forward/backward latency
for four causal transformer/GQA shapes. It reports CUDA events and synchronized
wall time separately, with warmup, repeated blocks and untimed kernel profiling.

```bash
for backend in fa2 fa4; do
  image="$FA4_IMAGE"
  if [ "$backend" = fa2 ]; then image="$FA2_IMAGE"; fi
  image_id="$(docker image inspect "$image" --format '{{.Id}}')"
  docker run --rm --gpus device=0 \
    -e "NPA_BENCHMARK_IMAGE_ID=$image_id" \
    -v "$PWD/npa/scripts:/validation:ro" \
    -v "$PWD/results/attention-comparison:/results" "$image" \
    python /validation/attention_kernel_benchmark.py --backend "$backend" \
      --output-path "/results/${backend}-kernels.json"
done
```

## Measure full model generation and retain renders

Use the [SDXL reproduction procedure](../fa4-sdxl-validation.md#reproduce) to
prepare the pinned public model snapshot and `/runtime/venv` with benchmark-only
dependencies. Use the **same** runtime overlay and model directory for both
images. The overlay refers to the current container's base packages; do not
install either attention package into it. Run `pip check` in both images and
retain their package inventories. Set `SDXL_MODEL_PATH` and `ATTENTION_RUNTIME_PATH`
to those local directories.

Run separate containers in FA2/FA4/FA4/FA2 order to expose order and thermal drift.
Each block performs an untimed coverage/warmup generation followed by five
measured 30-step generations at each of three resolutions. Both backends use the
same SDXL attention processor, with shape recording disabled during timing.
Image decode and watermarking are included in the generation measurement;
model loading and image-file writes are excluded.

```bash
block=0
for backend in fa2 fa4 fa4 fa2; do
  image="$FA4_IMAGE"
  if [ "$backend" = fa2 ]; then image="$FA2_IMAGE"; fi
  image_id="$(docker image inspect "$image" --format '{{.Id}}')"
  docker run --rm --gpus device=0 \
    -e "NPA_BENCHMARK_IMAGE_ID=$image_id" \
    -v "$PWD/npa/scripts:/validation:ro" \
    -v "$SDXL_MODEL_PATH:/model:ro" \
    -v "$ATTENTION_RUNTIME_PATH:/runtime" \
    -v "$PWD/results/attention-comparison:/results" "$image" \
    /runtime/venv/bin/python /validation/attention_sdxl_benchmark.py \
      --backend "$backend" --model-path /model \
      --output-path "/results/block-${block}-${backend}"
  block=$((block + 1))
done
npa/.venv/bin/python npa/scripts/compare_attention_sdxl.py \
  results/attention-comparison/block-*/report.json \
  --output-path results/attention-comparison/comparison.json
```

Each block saves a PNG and final latent array per scene, along with all timing
samples, untimed attention coverage, installed package versions, and hashes of
the mounted benchmark sources. Compare the renders and latent errors;
floating-point differences can amplify across diffusion steps, so pixel
identity is not a kernel-correctness test. The summary refuses incomplete runs,
mixed image identities within a backend, differing software/model/settings,
and fewer than two blocks per backend. A ratio above 1 favors FA4; below 1
favors FA2. Small differences within block variation do not establish a win.

## Experiment with FA4 inference tiles

The two benchmark workers accept `--tile 64x64`, `64x128`, `128x64`, or `128x128`
with `--backend fa4`. These options use the pinned upstream private forward API
to explore SM120 tile sizes; they are **inference-only**, reject another source
revision, and never change the default base adapter. The kernel worker omits
backward measurement when testing a tile. A candidate must pass numerical
checks and repeated full-model measurements before it can be adopted.

Compare every candidate with fresh default-FA4 and FA2 baselines on the same
GPU. Retain regressions as well as wins. Do not extrapolate SDXL results to
training, other sequence lengths, other GPUs or multi-node communication.
