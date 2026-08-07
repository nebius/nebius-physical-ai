# Cosmos 3 generation on the workbench (`npa-cosmos3`)

Cosmos 3 is NVIDIA's omni model: one checkpoint that both reasons and generates.
This guide covers the **generation** half as a containerized workbench tool —
image and video synthesis for Physical AI data — through the CLI, the SDK, and a
declarative `npa.workflow` spec that all share one implementation.

| Piece | Path |
| --- | --- |
| Image | `npa/docker/workbench/cosmos3/Dockerfile` (`npa-cosmos3`) |
| Runner (single source of truth) | `npa/src/npa/workbench/cosmos/generate.py` |
| CLI | `npa workbench cosmos3 generate` |
| SDK | `npa.sdk.workbench.cosmos3.generate(...)` |
| Workflow | `npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml` |
| `npa.workflow` toolRef | `workbench.cosmos3.generate` |
| Golden eval | `npa.smoke.test_cosmos3_generate_functional` (`gpu-gated`) |

Modes: `text2image`, `image2image`, `text2video`, `image2video`, `video2video`.
The last three
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
| `HF_TOKEN` (or the env named by `NPA_COSMOS3_HF_TOKEN_ENV`) | For a named checkpoint such as `Cosmos3-Nano`, **and** whenever guardrails are on | `generate` refuses to start and names the assets whose licenses you must accept |
| `NGC_API_KEY` (or `NPA_COSMOS3_NGC_API_KEY_ENV`) | Only when `NPA_COSMOS3_REQUIRE_NGC=1` | Same fail-fast, naming the NGC key |
| neither | `--checkpoint` is a staged local/`s3://` path **and** `--no-guardrails` | Runs; the token check is skipped |

A run pulls more than the checkpoint from Hugging Face: with guardrails on (the
default) it also fetches the gated `nvidia/Cosmos-Guardrail1`. So staging a
checkpoint on its own does **not** remove the token requirement — if it did, the
preflight would pass and the run would still die mid-inference fetching the
guardrail models, which is the failure the check exists to prevent.

This is enforced in three places: `require_model_access` refuses to launch
inference without the token, the build fails if a checkpoint file lands in a
layer, and `verify_env.py` re-asserts the absence of weights inside the image.

Clearing the license for this repo's own gated guardrail model
(`nvidia/Cosmos-Guardrail1`) does not clear the license for the *different*
gated guardrail repo a vLLM-Omni serving deployment pulls
(`nvidia/Cosmos-1.0-Guardrail`). See
[`cosmos3-access-preflight.md`](cosmos3-access-preflight.md) for account
setup, the two-repo table, the 401-vs-403 diagnostic for a gated-download
failure, and the Xet download workaround for a specific Hugging Face client
pin.

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

### Workflow

`npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml` runs the same stage
through the `workbench.cosmos3.generate` toolRef, which resolves to the
`npa-cosmos3` image automatically:

```bash
npa workbench workflow submit npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml \
  --infra k8s/<context> --registry <your-registry> \
  --var bucket=<bucket> \
  --secret-env HF_TOKEN --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

`--secret-env HF_TOKEN` is required: the plan only *hints* the secret, and without
it the stage fails fast on the credential preflight. The submit path also refuses
to run when the task image's registry does not match the Docker credentials in
`SKYPILOT_DOCKER_SERVER`.

The older raw SkyPilot template for this path was retired after this spec reached
a terminal live success through the submit matrix. Keep new workflow authoring on
the `npa.workflow/v0.0.1` surface.

SkyPilot still launches the rendered task under the hood. Its Kubernetes
bootstrap replaces the image entrypoint with its own shell and installs an SSH
runtime as the pod user, so the image ships `sudo` and `openssh-server` for it.
Without them a workbench image cannot host a SkyPilot k8s task (it dies with
`sudo: command not found`). Do not clear the workflow image pin as a workaround
for Cosmos 3 submits; fix the image bootstrap contract instead.

## View the result in the NPA agent

The generated artifact is an ordinary run artifact, so the agent's viewer renders
it. To do that without provisioning an agent VM, run the agent locally — the
script renders the same embedded backend + UI the VM bootstrap ships:

```bash
sudo mkdir -p /opt/npa-agent && sudo chown "$(id -u)":"$(id -g)" /opt/npa-agent
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_ENDPOINT_URL=...
export NPA_AGENT_S3_BUCKET=<bucket-holding-the-run>
npa/.venv/bin/python npa/scripts/run_agent_local.py     # http://127.0.0.1:8088/
```

Then load the artifact and open the **Rerun → IMAGE** viewer tab:

```bash
curl -s -X POST http://127.0.0.1:8088/api/sim-viz/load-artifact \
  -H 'content-type: application/json' \
  -d '{"s3_uri":"s3://<bucket>/runs/<run-id>/cosmos3-generate/generated/vision.jpg"}'
```

The Runs & Artifacts panel also finds the run by name (`cosmos3-`), and
**Describe this** sends the frame to the vision tier for a critique.

## Validated on real GPUs

Verified on `npa-rtxpro-mk8s` (NVIDIA RTX PRO 6000 Blackwell Server Edition,
sm_120) with `nvidia/Cosmos3-Nano`. The workflow path produced a non-blank
960x960 JPEG in S3 with `guardrails: true` and `weights_baked: false`:

| Path | Result |
| --- | --- |
| Direct Kubernetes Job (image args) | generated + published |
| `cosmos3-generate` npa.workflow via `workflow submit` | job 338 SUCCEEDED; `generated/generate.json` plus `generated/vision.jpg` |

Notes from those runs: the cu130 wheel set works on sm_120 (no NATTEN/flash-attn
kernel gap surfaced for text2image); the guardrail model, `Cosmos3-Nano`, and the
Wan 2.2 VAE all download at runtime, taking roughly two minutes on a warm node
before generation starts.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `Cosmos 3 weights are not baked into this image` | No HF token, or the token has not accepted the checkpoint's license. Accept it on the model page, or stage a checkpoint and pass it via `--checkpoint`. |
| `the Cosmos 3 inference runtime is not present` | Running outside the image. Use `npa-cosmos3`, or point `COSMOS3_REPO` at a framework checkout with a built `.venv`. |
| `mode ... conditions on an input image/video` | An `image2video` / `video2video` / `image2image` run without `--input-path`. |
| `Found no NVIDIA driver` | The container reached real inference but has no GPU. Generation is GPU-only. |
| `cosmos-framework produced no image/video artifact` | Inference exited 0 but wrote nothing; check the upstream log above the error for a guardrail rejection. |
| `Unable to parse string as hex hash value` from `huggingface_hub`'s Xet client | A download failure specific to the `hf-xet 1.5.1` + `huggingface_hub 1.23.0` pin pair (`huggingface/xet-core#895`), observed on a gated guardrail-repo download. Set `HF_HUB_DISABLE_XET=1` and retry; see [`cosmos3-access-preflight.md`](cosmos3-access-preflight.md). |

For access checks before a run (`gh`/HF/NGC reachability) see
`npa workbench cosmos check`. For the un-baked, clone-at-job-time text-to-image
smoke, use `npa/workflows/workbench/npa-workflows/cosmos3-text-to-image.yaml`.
