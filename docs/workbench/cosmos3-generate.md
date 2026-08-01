# Cosmos 3 generation on the workbench (`npa-cosmos3`)

Cosmos 3 is NVIDIA's omni model: one checkpoint that both reasons and generates.
This guide covers the **generation** half as a containerized workbench tool —
image and video synthesis for Physical AI data — through the CLI, the SDK, and a
SkyPilot workflow that all share one implementation.

| Piece | Path |
| --- | --- |
| Image | `npa/docker/workbench/cosmos3/Dockerfile` (`npa-cosmos3`) |
| Runner (single source of truth) | `npa/src/npa/workbench/cosmos/generate.py` |
| CLI | `npa workbench cosmos3 generate` |
| SDK | `npa.sdk.workbench.cosmos3.generate(...)` |
| Workflow | `npa/src/npa/workflows/skypilot/cosmos3-generate.yaml` |
| `npa.workflow` toolRef | `workbench.cosmos3.generate` |
| Golden eval | `npa.smoke.test_cosmos3_generate_functional` (`gpu-gated`) |

Modes: `text2image`, `text2video`, `image2video`, `video2video`. The last two
condition on an input asset, so they require `--input-path` (a local path, an
`http(s)` URL, or an `s3://` URI).

## Weights are never in the image

The image ships the OpenMDW-1.1 `cosmos-framework` **source at a pinned commit**
plus its cu130 inference environment. It contains **no model weights**, which is
what makes it redistributable under the packaging contract's `public` class.

Every gated artifact — the Cosmos 3 checkpoint, the Wan VAE it pulls, the
guardrail models — downloads **at run time** with credentials the operator
supplies, under the operator's own license acceptance:

| Credential | When | Effect if missing |
| --- | --- | --- |
| `HF_TOKEN` (or the env named by `NPA_COSMOS3_HF_TOKEN_ENV`) | Always, for a named checkpoint such as `Cosmos3-Nano` | `generate` refuses to start and names the license you must accept |
| `NGC_API_KEY` (or `NPA_COSMOS3_NGC_API_KEY_ENV`) | Only when `NPA_COSMOS3_REQUIRE_NGC=1` | Same fail-fast, naming the NGC key |
| neither | When `--checkpoint` is a local path or `s3://` URI you already staged | Runs; the token check is skipped |

This is enforced in three places: `require_model_access` refuses to launch
inference without the token, the build fails if a checkpoint file lands in a
layer, and `verify_env.py` re-asserts the absence of weights inside the image.

Guardrails are **on** unless you pass `--no-guardrails`, and every result
manifest records `guardrails` so a run's posture stays auditable.

## Build

```bash
# Defaults to the pinned framework commit and the supported-tools tag.
bash npa/docker/workbench/cosmos3/build.sh --registry <your-registry>

# Push it so the workflow's NPA_COSMOS3_IMAGE can resolve.
bash npa/docker/workbench/cosmos3/build.sh --registry <your-registry> --push

# Pin a different upstream commit (re-validates at build time).
bash npa/docker/workbench/cosmos3/build.sh --ref <40-char-sha>
```

The build runs `verify_env.py`, which walks the inference graph for every mode
above — flags, torch/flash-attn, the guardrail package, checkpoint-URI
resolution, and setup/sample resolution — stopping short of weight loading. If an
upstream bump needs a dependency the trimmed set lacks, the build fails there
instead of on your first GPU run.

### Size and node disk

The image is **27.3 GB** uncompressed, so plan node ephemeral storage and first-pull
time accordingly (the checkpoint cache is on top of this — `Cosmos3-Nano` is a 16B
model). It breaks down as:

| Layer | Size | Note |
| --- | --- | --- |
| `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` base | ~17 GB | Dominates. Kept as the `devel` variant because upstream builds against it and Triton JIT expects a real `ptxas`; a `runtime` base would be far smaller but is unvalidated here |
| framework venv (torch cu130, flash-attn, natten, guardrail deps) | 9.3 GB | Appears once; installs run as the runtime user so no layer duplicates it |
| apt (ffmpeg, git, git-lfs) | 534 MB | |
| `npa` venv | 474 MB | |
| framework source + uv binaries | ~90 MB | |

## Run

### CLI (inside the image, on a GPU)

```bash
npa workbench cosmos3 generate \
  --mode text2image \
  --prompt "a robot arm sorting colored blocks on a white workbench" \
  --output-path s3://<bucket>/cosmos3/<run-id>/ \
  --checkpoint Cosmos3-Nano \
  --seed 0
```

`--output-path` takes a local directory or an `s3://` prefix; an S3 target
uploads the artifact plus a `generate.json` manifest. Add `--run-id` to carry a
run identifier into that manifest.

`--dry-run` works anywhere, including a laptop with no GPU: it resolves the input
sample and the exact upstream argv, so you can confirm mode, checkpoint, and
guardrail posture before spending GPU time.

### SDK

```python
from npa.sdk.workbench import cosmos3

result = cosmos3.generate(
    prompt="a robot arm pours water into a glass",
    mode="image2video",
    input_path="s3://<bucket>/frames/robot_153.jpg",
    output_path="s3://<bucket>/cosmos3/<run-id>/",
)
print(result["output_kind"], result["artifact_uri"])
```

### SkyPilot

```bash
sky launch -y --infra kubernetes/<context> -c cosmos3-generate \
  --env NPA_COSMOS3_IMAGE=<your-registry>/npa-cosmos3:1.2.2-cu130 \
  --env NPA_COSMOS3_PROMPT="a robot arm sorting colored blocks" \
  --env NPA_COSMOS3_OUTPUT_URI=s3://<bucket>/cosmos3/<run-id>/ \
  --secret HF_TOKEN \
  npa/src/npa/workflows/skypilot/cosmos3-generate.yaml
```

The YAML requests `H100:1` and fails fast with an explicit message if no HF token
is present, rather than discovering it after the pod is scheduled.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `Cosmos 3 weights are not baked into this image` | No HF token, or the token has not accepted the checkpoint's license. Accept it on the model page, or stage a checkpoint and pass it via `--checkpoint`. |
| `the Cosmos 3 inference runtime is not present` | Running outside the image. Use `npa-cosmos3`, or point `COSMOS3_REPO` at a framework checkout with a built `.venv`. |
| `mode ... conditions on an input image/video` | An `image2video` / `video2video` / `image2image` run without `--input-path`. |
| `Found no NVIDIA driver` | The container reached real inference but has no GPU. Generation is GPU-only. |
| `cosmos-framework produced no image/video artifact` | Inference exited 0 but wrote nothing; check the upstream log above the error for a guardrail rejection. |

For access checks before a run (`gh`/HF/NGC reachability) see
`npa workbench cosmos check`. For the un-baked, clone-at-job-time text-to-image
smoke, see `cosmos3-text-to-image-inference.yaml`.
