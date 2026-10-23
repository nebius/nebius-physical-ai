<!-- register: operator recipe | reader: training engineer | consumed: prepare inputs and verify native execution -->
# Cosmos3-DROID on eight H200 GPUs

This standalone application recipe adds a native S3 video reader and an eager
execution profile for NVIDIA's `action_policy_droid_nano` recipe. It uses
32 samples per rank, eight ranks and accumulation 32: 8,192 samples per update.
Four loader workers per rank, inner batch 16 and prefetch factor two match the
tested configuration. The reader keeps one 1 MiB block per handle. The native
decoder retains its 64-file LRU cache. Metadata, actions, timestamps, filtering
and augmentation keep the pinned upstream semantics.

This is a short execution check. Evaluation and every checkpoint save, including
the final save, are disabled. The number of updates is an explicit operator input.
No `npa.workflow` toolRef, cloud provisioning or training-quality claim is added.

The [measured result](../../../../docs/workbench/evidence/cosmos3-droid-h200/README.md)
contains five complete updates from the original instrumented implementation.
The portable reader and launcher in this directory passed a separate
[one-update integration check](../../../../docs/workbench/evidence/cosmos3-droid-h200/integration.json)
on October 15. Two fixed decoded windows and one controlled augmented window
matched the original reader. Native plan/dryrun and all eight terminal rank
receipts passed: 32 microsteps and one optimizer operation per rank, 8,192 global
samples, finite losses and no new checkpoints. The historical timing does not
measure this refactor. The integration check makes no timing or quality claim.

## Prepared inputs

Use a single eight-H200 node with the following inputs already available:

- A clean Git checkout of [NVIDIA cosmos-framework](https://github.com/NVIDIA/cosmos-framework)
  at the revision in [sources.json](sources.json). Its `.venv` must contain the
  frozen `train` extra and `cu130-train` dependency group. The measured runtime
  used Python 3.13, PyTorch 2.10.0+cu130, torchcodec 0.10.0+cu130, PyAV 18.1.0
  and LeRobot 0.4.4. Keep this environment separate from the NPA client.
- The `success/` split of [nvidia/Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID)
  at the pinned revision. Place `success/meta/` and `success/data/` locally.
  Store video objects under `success/videos/` in the selected bucket. The measured
  split contains 3,133 files and 628,036,574,065 bytes, including 3,063 MP4 files.
- A manifest with `files` entries containing each bucket-relative `path` and
  integer `size`. Generate it from the pinned dataset inventory and verify the
  bucket listing against it before use. Every selected video must be present;
  a missing or malformed range stops the reader.
- A converted base DCP from the pinned `nvidia/Cosmos3-Nano`, including
  `model/.metadata`. Use the pinned framework's
  [DCP converter](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/scripts/convert_model_to_dcp.py).
  The text tokenizer is `Qwen/Qwen3-VL-8B-Instruct`, with its tokenizer/config
  JSON files. The VAE is `Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`. Use the
  exact revisions in `sources.json`, not moving default revisions.
- A separately materialized AVAE sound-tokenizer cache from that Nano revision.
  Copy Nano's `sound_tokenizer/config.json` and
  `sound_tokenizer/diffusion_pytorch_model.safetensors` into a writable cache.
  Materialize them there with the pinned framework's
  [`_materialize_avae_ckpt(cache_directory)`](https://github.com/NVIDIA/cosmos-framework/blob/cf5d68c00d97ccd2480a2320ed652b92dec63102/cosmos_framework/inference/common/checkpoints.py#L67).
  Required outputs are `avae_48k_noncausal_25hz_64ch.ckpt` and
  `avae_48k_noncausal_25hz_64ch.json`. The entry
  point redirects the registry to this cache, avoiding writes beside original
  model weights. Input preparation and DCP conversion are prerequisites, not
  operations performed by this launcher.
- The `keep_ranges_1_0_1.json` filter fetched separately from
  [KarlP/droid](https://huggingface.co/KarlP/droid) at the pinned revision.
  That repository declares no license. This contribution does not redistribute
  the filter or grant permission to redistribute it.
- Read access to the selected bucket through the standard Boto3 credential
  chain, or `--aws-profile`. The endpoint must use HTTPS without embedded
  credentials or query parameters. Credential values never belong in arguments,
  manifests or Git. Runtime settings contain private local paths and bucket
  configuration; retain generated output outside the repository.

Use operator-owned writable caches and sufficient shared memory. The original
test used private 128 GiB shared memory and a writable RAM workspace; it did not
require all videos to fit on local disk. The launcher does not allocate or stop
resources. Set `LD_LIBRARY_PATH` for the selected CUDA environment when needed.

## Plan, validate and run

Set the input variables to your prepared inputs and select fresh output paths.
Run these commands from the Workbench repository root with the framework Python:

```bash
: "${FRAMEWORK_ROOT:?Set the pinned framework checkout}"
: "${DROID_ROOT:?Set the directory containing success/meta and success/data}"
: "${DROID_MANIFEST:?Set the verified path and size manifest}"
: "${BASE_DCP:?Set the converted Nano DCP directory}"
: "${WAN_VAE:?Set the pinned Wan VAE file}"
: "${QWEN_TOKENIZER:?Set the pinned local tokenizer directory}"
: "${AVAE_CACHE:?Set the materialized sound-tokenizer cache}"
: "${DROID_FILTER:?Set the separately fetched window filter}"
: "${S3_BUCKET:?Set your dataset bucket}"
: "${S3_ENDPOINT:?Set your HTTPS S3 endpoint}"
: "${S3_REGION:?Set the bucket region}"
: "${RUN_ROOT:?Set a new operator-owned output directory outside Git}"

recipe=npa/workflows/workbench/cosmos3-droid-h200/recipe.py
inputs=(
  --framework "$FRAMEWORK_ROOT" --dataset-root "$DROID_ROOT"
  --manifest "$DROID_MANIFEST" --base-dcp "$BASE_DCP" --vae "$WAN_VAE"
  --tokenizer "$QWEN_TOKENIZER" --avae-cache "$AVAE_CACHE" --filter "$DROID_FILTER"
  --bucket "$S3_BUCKET" --endpoint-url "$S3_ENDPOINT" --region "$S3_REGION"
  --updates 1
)
"$FRAMEWORK_ROOT/.venv/bin/python" "$recipe" plan "${inputs[@]}" --output-dir "$RUN_ROOT/plan"
"$FRAMEWORK_ROOT/.venv/bin/python" "$recipe" dryrun "${inputs[@]}" --output-dir "$RUN_ROOT/dryrun"
"$FRAMEWORK_ROOT/.venv/bin/python" "$recipe" run "${inputs[@]}" --output-dir "$RUN_ROOT/run"
```

Input revision and content verification is an operator prerequisite. The launcher
checks the clean framework revision, expected input layout and video manifest;
it does not prove the provenance of arbitrary files supplied at those paths.

Each output directory must be new. `plan` writes the resolved TOML, settings and
argv without starting training. `dryrun` validates the native configuration.
`run` launches eight ranks through torchrun, with no automatic process restarts.
All three retain the upstream global batch. Torch compilation, model compilation
and the VAE compile callback are disabled for H200 eager execution.

Keep terminal output in private operator logs. The run command verifies all eight
terminal rank receipts, actual sample counts, finite losses and optimizer-operation
counts before returning success. Receipts are under the run output's
`verification/` directory. This
launcher suppresses checkpoints and thus supplies no resumable training result.
Preserve evidence before removing only your run-owned outputs. Resource teardown
is a separate operator action.

## Recompute the published result without a GPU

```bash
npa/.venv/bin/python npa/workflows/workbench/cosmos3-droid-h200/report.py \
  --evidence-dir docs/workbench/evidence/cosmos3-droid-h200
```

Expected: five updates, 40,960 samples, post-first mean 1,571.5598394984845
seconds, population coefficient of variation 0.015146991108656234 and
88.70521688461304 GiB peak allocated memory. The reducer validates the published
numeric extracts, their checksums, all eight rank sequences and finite losses.
