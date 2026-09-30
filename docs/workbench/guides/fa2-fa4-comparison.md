# Compare FA2 and FA4 on RTX PRO 6000

[RTX FA4 application guide](rtx6000-fa4.md) · [Workbench guides](README.md)

Use separate FA2 and FA4 images on the same Nebius RTX PRO 6000 GPU. The images
share one CUDA/PyTorch recipe and the same FlashAttention source revision. The
FA2 variant compiles the standalone CUDA extension for SM120; the FA4 variant
installs the CuTe implementation and guarded root import adapter. Installing both
into one Python environment would make the root `flash_attn` import ambiguous.

**Measured result:** the [latest RTX optimization and actual renders](../fa4-rtx-optimization.md)
record 108 complete SDXL generations and repeated attention measurements.
Optimized FA4 reached 1.76–2.08× FA2 on four short cross-attention calls and
1.07–1.27× on four causal inference calls. Complete SDXL generation was only
0.4–0.8% faster, with higher first-generation compilation cost. The profile remains opt-in. These are qualified local
images, separate from the published Workbench release inventory. Earlier
[SDXL evidence](../fa4-sdxl-validation.md) compared FA4 with PyTorch SDPA.

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
comparison; it does not qualify that extension on B200/B300. The legacy build
entrypoint also accepts `--attention-backend fa2` and emits the canonical FA2 tag.
This new variant has no B300 alias. Existing FA4 tags are never reused for FA2.

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
      --warmup 1000 --iterations 200 --repeats 9 \
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
Each block records a first coverage/warmup generation separately, followed by five
steady-state 30-step generations at each of three resolutions. Both backends use the
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
mixed image identities or helper hashes within a backend, differing software/model/settings,
and fewer than two blocks per backend. A ratio above 1 favors FA4; below 1
favors FA2. Small differences within block variation do not establish a win.

## Experiment with FA4 inference tiles

The FA4 image provides an explicit inference factory. Its
[measured launch and tile profile](../fa4-rtx-optimization.md#every-attention-call-result)
uses cached upstream kernels for qualified contiguous inputs. It improves
selected attention calls substantially, but complete SDXL generation by less
than 1% versus FA2. Keep it opt-in and measure your application:

```python
import torch
from flash_attn.rtx import make_inference_attention

# Create once after selecting this process's CUDA device.
attention = make_inference_attention("cuda:0")
with torch.inference_mode():
    output = attention(query, key, value, causal=False)
```

The factory requires the exact pinned FA4 revision and SM120. It rejects
enabled gradients, tensors requiring gradients, mismatched devices and options
outside its Q/K/V plus `causal` interface. It does not change the root adapter
or register a model backend automatically. Its tile selection is limited to the
benchmark's batch-2 FP16 SDXL shapes and BF16 causal transformer shapes.
Contiguous, 16-byte-aligned qualified inputs directly launch the cached kernel
with fresh input pointers and output allocation. Strided inputs retain the
private tiled API; unlisted shapes use native FA4. It never substitutes FA2 or
SDPA. Use the standard FA4 API for training and advanced attention semantics.

### Qualify the helper and retain compiled code

Cold compilation can outweigh steady-state gains for a short job. To reuse
compiled kernels across processes, mount a persistent directory writable by
UID 1000 and set both upstream cache variables before importing FA4. The cache
fingerprints the helper, upstream code and compiler versions; changing the
qualified stack requires fresh validation.

The following exercises all optimized shapes against FP64, fresh values,
multiple streams and graph replay, then starts another container with new
compilation forbidden. Use the helper hash from the exact source used to build
the selected image. Do not derive the expected hash from the image under test.

```bash
mkdir -p results/attention-comparison/cute-cache
sudo chown -R 1000:1000 results/attention-comparison
RTX_HELPER_SHA=$(sha256sum npa/docker/workbench/base/cuda13-blackwell/scripts/flash_attn_rtx.py | cut -d ' ' -f1)
FA4_IMAGE_ID=$(docker image inspect "$FA4_IMAGE" --format '{{.Id}}')
for mode in populate reload; do
  extra=()
  if [ "$mode" = reload ]; then extra=(--require-cache-hit); fi
  docker run --rm --gpus device=0 --network none \
    --cap-drop ALL --security-opt no-new-privileges \
    --read-only --tmpfs /tmp:rw,exec,mode=1777 -e HOME=/tmp \
    -e "NPA_BENCHMARK_IMAGE_ID=$FA4_IMAGE_ID" \
    -e FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    -e FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/cache \
    -v "$PWD/results/attention-comparison/cute-cache:/cache" \
    -v "$PWD/results/attention-comparison:/results" \
    -v "$PWD/npa/scripts:/validation:ro" "$FA4_IMAGE" \
    python /validation/validate_rtx_attention.py \
      --expect-helper-sha256 "$RTX_HELPER_SHA" \
      --output-path "/results/qualification-${mode}.json" "${extra[@]}"
done
```

Both receipts must report `status: passed`: 64 checks for population and 61 for
reload. The reload excludes native-path checks and fails if the optimized
launcher attempts any compilation. Use a fresh cache directory for population
when measuring cold behavior. Persistent caching is optional; the in-process
compiled-kernel cache works without those variables.

### Compare the optimized callable

Both workers accept `--backend fa4 --tuning rtx6000-inference` to measure this
baked callable. Compare it with fresh FA2 and default-FA4 blocks using the same
worker sources. `--tuning` and `--tile` are mutually exclusive. Keep default and
tuned FA4 reports in separate comparisons so mixed modes cannot hide a regression.

The two benchmark workers accept `--tile 64x64`, `64x128`, `128x64`, or `128x128`
with `--backend fa4`. These options use the pinned upstream private forward API
to explore SM120 tile sizes; they are **inference-only**, reject another source
revision, and never change the default base adapter. The kernel worker omits
backward measurement when testing a tile. A candidate must pass numerical
checks and repeated full-model measurements before it can be adopted.

Compare every candidate with fresh default-FA4 and FA2 baselines on the same
GPU. Retain regressions as well as wins. Do not extrapolate SDXL results to
training, other sequence lengths, other GPUs or multi-node communication.
