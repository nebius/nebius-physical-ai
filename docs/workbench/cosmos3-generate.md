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
| `HF_TOKEN` (or the env named by `NPA_COSMOS3_HF_TOKEN_ENV`) | Optional for public `Cosmos3-Nano`; required when guardrails are on or for a gated/private checkpoint override | `generate` anonymously checks public assets and uses the token only where repository access requires it |
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

The r2 image keeps the faster Xet transfer path enabled with its measured,
compatible baked versions (`huggingface_hub==0.36.2`, `hf-xet==1.3.2`). The
image build records this pair and fails if the known-bad `1.23.0` / `1.5.1`
combination is ever resolved; only non-image/custom environments need the
runtime diagnostic and `HF_HUB_DISABLE_XET=1` fallback described there.

Guardrails are **on** unless you pass `--no-guardrails`, and every result
manifest records `guardrails` so a run's posture stays auditable.

Known limitation: a prior live run requested guardrails but upstream reported
`No safety models found, returning safe`. The manifest currently records the
requested posture, not proof of effective safety-model execution. This is
tracked separately in [issue #270](https://github.com/nebius/nebius-physical-ai/issues/270);
this release does not redesign guardrail behavior.

## Build

The supported/default image release is `npa-cosmos3:1.2.2-cu130-r2`. It is an
additive successor to `1.2.2-cu130`: the old immutable tag is retained for
rollback and provenance and must never be overwritten or deleted. Pre-merge
validation builds use a branch-specific candidate tag in a private registry;
the official `1.2.2-cu130-r2` tag is built and published only from the reviewed
trusted commit.

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

`--secret-env HF_TOKEN` is required when guardrails are enabled: the plan only *hints* the secret, and without
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
  -d '{"run_id":"<run-id>","s3_uri":"s3://<bucket>/runs/<run-id>/cosmos3-generate/generated/vision.jpg"}'
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

## Measured timing: Cosmos3-Super text2video on H200 and B200

Measured 2026-08-08 with `npa-cosmos3:1.2.2-cu130` (npa `3fe85845`, cosmos-framework `5e67049c`), mode `text2video`, checkpoint `Cosmos3-Super` at the serving lane's anchor shape: 1280x720, 189 frames, 24 fps, 35 sampling steps, seed 17, guardrails on, NVIDIA's example text-to-video prompt. One job is one full model load: every invocation loads the checkpoint, samples, decodes, applies the guardrail postprocessor, encodes, and publishes, then exits.

| Platform | Wall per invocation | Sampling | Steady state per step | Notes |
| --- | --- | --- | --- | --- |
| 8x H200 SXM (141 GB HBM3e per GPU) | 819 s mean (n=3: 819 / 820 / 821 s) | 756.8 s mean | ~21 s/step | Model load 39-41 s warm page cache plus ~10 s container overhead; requires `PYTORCH_ALLOC_CONF=expandable_segments:True` for 720p VAE decode (see operational notes) |
| 8x B200 SXM (192 GB HBM3e per GPU) | ~473 s warm (n=2: 473 / 472 s; the very first invocation took 740 s including one-time guardrail-asset downloads) | ~394 s | ~11.3 s/step | The full 124 GB checkpoint fits one card; no allocator workaround needed |

The batch-path single job uses one GPU: at this commit the runner launches one process on one card, and upstream's multi-GPU path (`torchrun` full-shard) is not exposed through the CLI. The remaining 7 GPUs of the node are idle during a batch job.

### Batch versus served: the load-every-time cost structure

The same workload (anchor shape, guardrails on) served from a resident vLLM-Omni endpoint with the card-recommended 8-GPU config measures **142.1 s** (8x H200) and **87 s** (8x B200) per clip; the server's startup cost is paid once and amortizes across requests. The batch path pays its overhead on every invocation:

- Sampling is the dominant term: ~92% of wall on H200, ~83% on B200.
- The remainder, ~62 s per invocation on H200 and ~79 s on B200, covers container start, model load, VAE decode, the guardrail postprocessor, MP4 encode, and publish. (Warm model load alone measures 39-41 s on H200 and ~80 s load-to-first-step on B200; component boundaries overlap, so the wall and sampling figures are the authoritative totals.)

Net: per anchor clip the batch path costs about **5.7x** (H200: 819 vs 142.1 s) or **5.4x** (B200: 473 vs 87 s) the served-resident cost, before accounting for the serving path's own multi-minute boot. Use the batch path for one-off or low-volume generation; keep a resident server for any sustained synthetic-data volume.

### Determinism per platform

Same seed, same config, separate invocations of the batch job:

- **B200: 4 of 4 invocations byte-identical** (one sha256 across all outputs), including across separate container starts and a host cache rebuild.
- **H200: 5 invocations split into two byte-stable groups** (3 runs share one sha256, 2 share another). Within each group the output is bit-exact across cold starts; the two groups differ, most plausibly from per-invocation compile/autotune kernel selection (hypothesis, not established).

The serving path's rule is different on both platforms: output is byte-identical only within one running server instance; a server restart changes the bytes (restart-drift medians measured in the serving-lane study: 26.8-29.0 dB PSNR on H200, 32.1-32.2 dB on B200). Plan verification around hash equality where measured, and metadata plus perceptual checks otherwise.

### Operational notes for single-GPU operation

- On H200-class cards (141 GB per GPU) at 720p, set `PYTORCH_ALLOC_CONF=expandable_segments:True` or the VAE decode runs out of memory with the full 64B (124 GB) checkpoint in one process; the B200's 192 GB per card does not need it.
- The job publishes `vision.mp4` plus `generate.json` to the output path verbatim, so multiple cells to one output prefix overwrite each other: give every cell its own prefix.
- Container invocations need `USER`/`LOGNAME` exported for the runtime uid, or torch import fails on a passwd-less uid, and the HF cache mount must be writable by the container's uid.

### Guardrail posture on this path

At this framework ref the video content-safety classifier is commented out upstream ("Too many false positives, add back when fixed"): the runtime posture is the text guardrail (Blocklist + Qwen3Guard) plus the RetinaFaceFilter face-blur postprocessor. Manifests record `guardrails: true` either way; do not describe this path as screening video content.
