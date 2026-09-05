# Cosmos3-Nano: 30-second diffusion video on NPA mk8s

This deployment runs 16 Ray Serve replicas on 16 B200 GPUs, with one GPU and
tensor parallelism 1 per replica. Each complete request generates a 30-second
480p video through `vllm serve --omni` and `Cosmos3OmniDiffusersPipeline`.
Guardrails are **off**, explicitly selected with `--no-guardrails`. Audio is
disabled. This is the diffusion generation route.

## Model and frame contract

The [official Nano model card](https://huggingface.co/nvidia/Cosmos3-Nano/blob/7a312c868bcce8e40b3eb40861300a9d0ba3fde1/README.md)
was checked before generation. It currently specifies **5–400 output frames**,
up to **five conditioning input frames**, and output at the requested fps. The
often-quoted 4 fps recommendation applies to reasoner inputs. Only BF16 is
officially tested. This recipe deliberately retains a stricter **300-frame
per-request ceiling**.

The [baked diffusion pipeline](https://github.com/vllm-project/vllm-omni/blob/9c1b7504b178afcf541867c1a2d30db48c69cda8/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py)
rounds requested frame counts upward to `4k+1`; 300 would become 301. At 24 fps
and 832×480, the rollout requests **297, 297, and 137 frames**. The first request
is text-to-video. Subsequent requests upload the previous segment as MP4 and
set `condition_frame_indexes_vision: [0, 1]` and `condition_video_keep: last` in
`extra_params`. These are latent indices: two latent positions represent five
pixel frames, which remain clean conditioning through denoising.

Stitching discards the duplicated five-frame prefix from each continuation,
then trims one final frame: `297 + 292 + 132 - 1 = 720`, exactly 30 seconds.
There is no blending or interpolation to hide discontinuities. The client
retains the chunks and eight-frame contact sheets around joins at 12.375 and
24.5417 seconds. Generation uses 35 steps, guidance 6, flow shift 10, and a
4096-token sequence setting inside `extra_params` (the image's HTTP handler
does not parse a top-level `max_sequence_length` form field).

## Infrastructure and image

Use NPA `cluster up` or the NPA fleet mk8s backend with a task-owned project in
`us-central1`, 16 `gpu-b200-sxm` GPUs, managed CUDA 13 drivers,
a CPU pool, and shared filesystem storage. Ray uses 16 one-GPU worker pods,
so the cluster may provide two eight-GPU nodes or a validated mix of B200 node
sizes. Keep NPA's quota, fabric,
stabilization and per-node CUDA validation enabled. If using reserved capacity,
supply its exact group through owner-only runtime configuration and require
`STRICT` reservation placement. Follow the [fleet instructions](../../../skills/tools/fleet/SKILL.md)
and [driver strategy](../../../docs/workbench/mk8s-gpu-driver-strategy.md).

The Dockerfile extends `vllm/vllm-omni:cosmos3`, pinned to manifest digest
`sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587`.
The Dockerfile pins that index's Linux AMD64 manifest,
`sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c`.
The extension supplies Ray Serve, FFmpeg and the NPA adapters. It does not
replace the diffusion engine or bake weights, data, credentials or acceptance
state. Its inherited vendor runtime makes this an **operator-private** image;
the NPA public publisher excludes it. Build and push only to the operator's own
registry, then deploy the verified immutable image digest.

The staging Job is the single writer of the shared model cache. It downloads
the pinned Nano revision once, verifies all diffusion and VAE tensor dtypes as
BF16, hashes the staged files, and atomically publishes `READY.json`. Replicas
mount the cache read-only and run with Hugging Face offline mode. Their vLLM
processes preserve `--init-timeout 1800`. The synchronous HTTP generation route
has no additional workload deadline. Each replica runs one complete rollout
at a time; Ray's custom router considers all replicas at one rank and chooses
the least outstanding queue, with capacity rejection protecting concurrent
admission.

## Deploy and operate

1. Verify provider credentials with `npa workbench health preflight --checks
   nebius --json`, and exact model payload access with `npa workbench health
   access --capability cosmos3 --json`. Verify the selected image and storage
   before starting generation.
2. Provision the NPA mk8s cluster and shared filesystem. Install KubeRay operator
   **1.7.0** with its namespace watch restricted to `workbench`. Resolve all
   manifest placeholders from owner-only configuration.
3. Apply [shared-pvc.yaml](shared-pvc.yaml) after NPA installs its shared-filesystem
   CSI driver. Create the API token Secret and operator-private registry pull
   Secret, then run the weight staging
   Job to completion. Run the RayService only after the immutable cache is ready.
4. Require all 16 model replicas to be healthy, confirm B200 placement, and
   verify the API rejects unauthenticated requests. Expose access through an
   authenticated private route or a local port-forward.
5. Set `NPA_COSMOS3_VIDEO_ENDPOINT`, `NPA_COSMOS3_VIDEO_TOKEN` and
   `NPA_COSMOS3_VIDEO_RECOVERY_DIR` outside Git. Configure the normal NPA S3
   endpoint and credentials in the client process. Use distinct artifact
   prefixes for each batch, separate from any agent trajectory dataset.

```bash
npa workbench cosmos3 nano-video-batch --concurrency 1 \
  --output-path "$NPA_COSMOS3_SINGLE_OUTPUT_URI"
npa workbench cosmos3 nano-video-batch --concurrency 8 \
  --output-path "$NPA_COSMOS3_FANOUT_OUTPUT_URI"
```

The SDK entry is `npa.sdk.workbench.cosmos3.nano_video_batch`. The authenticated
service accepts `/run`; `/artifacts/<request_id>/<filename>` serves the retained
files. CLI and SDK verify returned hashes, fully decode all three chunks and the
stitched MP4, and
publish immutable S3 objects with read-after-write verification. Publication
failure retains the local recovery copy; **do not repeat GPU generation to
retry an upload**.

## Required measured acceptance

Live validation is required before publication of this change. The acceptance
test in `npa/tests/e2e/test_cosmos3_nano_video_live.py` runs one complete video
through the SDK followed by eight complete concurrent videos through the CLI,
including immutable S3 publication and read-after-write verification. Measurements will be recorded
here after that real workload completes.

Each chunk records client wall time, the engine's `X-Inference-Time-S` and
`X-Peak-Memory-MB`, and stage durations. The CUDA allocator peak is distinct
from total GPU residency: a second measurement samples the assigned B200's
`nvidia-smi memory.used` every 0.5 seconds. Sampling may miss shorter spikes.
The eight-way gate requires eight successful videos, eight distinct model
replicas, eight overlapping server execution intervals, and eight overlapping
diffusion chunk requests. Visual seam review
is a separate observation; low pixel differences alone do not establish
perceptual continuity.

Exact resource identities, private endpoints, registry coordinates, artifacts
and operational evidence remain in access-controlled runtime storage. This
recipe publishes no such identifiers.
