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

The head explicitly advertises zero Ray GPUs and hides NVIDIA/CUDA devices.
Its required affinity selects nodes without `nvidia.com/gpu.count`; verify that
the NPA CPU pool has no such label and every validated GPU node has a positive
count before applying the manifest. Worker device visibility stays under the
Kubernetes device plugin and Ray's one-GPU assignment.

For a mixed cluster, declare the planned per-node GPU counts through
`GpuHealthConfig.expected_gpu_counts`. Its optional `nvswitch_gpu_counts` subset
identifies node sizes that require fabric checks. Every multi-GPU SXM/NVL size
must be included. An explicitly unattached one-GPU guest can report fabric as
not applicable while still passing driver, device-plugin and CUDA checks.
The mixed live health test verifies the declared total, node-count distribution,
stability and CUDA on every GPU ordinal before serving begins.

The Dockerfile extends `vllm/vllm-omni:cosmos3`, pinned to manifest digest
`sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587`.
The Dockerfile pins that index's Linux AMD64 manifest,
`sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c`.
The extension supplies Ray Serve, FFmpeg and the NPA adapters. It preserves the
diffusion engine and adds no model weights, task or customer data, credentials
or acceptance state. The inherited image contains public vendor test fixtures
and example media. Its vendor runtime makes this an **operator-private** image;
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

The router uses Ray's public FIFO fulfillment mixin to keep pending requests
reachable when another scheduler has already consumed their routing metadata.
FIFO selects the pending request; fresh queue snapshots across all replicas
select the least outstanding replica. The router follows Ray's normal retry backoff.
It disables Ray 2.56's queue-length cache because the cached-success path can
spin after out-of-order assignments and accumulate background probes in the
head proxy. Every selection still compares the complete replica rank; strict
admission prevents two requests from occupying the same replica.

The adapter writes an explicit single diffusion stage configuration with
`model_config.sound_gen: false`, BF16 and TP=1, then passes it through
`--stage-configs-path`. This supported legacy flag is needed for the pinned
engine: its single-stage diffusion fallback drops `--stage-overrides`, and the
checkpoint's sound dimensions would otherwise enable an unstaged audio
tokenizer. The image's CPU validation resolves the actual engine configuration
and checks that sound remains disabled; CLI parsing alone is insufficient.

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

For `proxy_location: HeadOnly`, route port 8000 through KubeRay's stable
`cosmos3-nano-video-head-svc`, for example
`http://cosmos3-nano-video-head-svc.workbench.svc.cluster.local:8000` inside the
cluster, or a local port-forward to that Service. KubeRay updates this Service's
head-only selector when the active cluster changes. The generated
`cosmos3-nano-video-serve-svc` is unused: KubeRay 1.7 always labels worker pods
as serving endpoints, while these workers have no HTTP proxy. The explicit
worker readiness probe checks Ray's native health endpoint on port 52365;
all 16 model replicas must separately pass application readiness. Custom
`serveService.spec.selector` values are
[overwritten by KubeRay](https://github.com/ray-project/kuberay/blob/v1.7.0/ray-operator/controllers/ray/common/service.go#L216).

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

## Measured B200 acceptance

The completed acceptance ran one full 30-second video through the SDK, then eight complete generation requests concurrently through the CLI. All nine clips passed full decoding at 832×480, 24 fps and 720 frames. The two batches published 13 and 97 immutable objects respectively, with read-after-write hash verification. Models were already initialized and BF16 weights prestaged before timing.

The single-request batch took **153.61 s**; the eight-request batch took **163.04 s**. These batch times include generation, downloads and local validation, and exclude S3 publication. The complete sequential acceptance test case, including both batches and their S3 publication/readback, took **323.61 s**; pytest reported **324.41 s** for the full invocation. Deployment, model initialization and prestaging are excluded.

| Request | Chunk 1 (s) | Chunk 2 (s) | Chunk 3 (s) | Server total (s) | Client total (s) | Peak allocator reserved (MiB) | Peak device used (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 59.22 | 60.18 | 22.52 | 149.64 | 153.58 | 37408.00 | 38316.00 |
| C1 | 59.36 | 60.45 | 22.57 | 150.26 | 156.37 | 37408.00 | 38316.00 |
| C2 | 59.43 | 60.26 | 22.57 | 151.26 | 160.39 | 37408.00 | 38316.00 |
| C3 | 59.94 | 60.54 | 22.85 | 151.81 | 160.07 | 37408.00 | 38316.00 |
| C4 | 59.45 | 60.28 | 22.55 | 151.42 | 158.55 | 37408.00 | 38316.00 |
| C5 | 60.86 | 61.10 | 22.87 | 154.05 | 162.15 | 37408.00 | 38316.00 |
| C6 | 60.11 | 60.30 | 22.55 | 151.65 | 159.97 | 37408.00 | 38316.00 |
| C7 | 60.12 | 60.65 | 22.95 | 152.80 | 162.24 | 37408.00 | 38316.00 |
| C8 | 61.28 | 61.98 | 23.45 | 155.37 | 162.90 | 37408.00 | 38316.00 |

S1 is the single request; C1–C8 belong to the concurrent batch. Chunk time is the synchronous diffusion HTTP round trip inside the replica. Server total includes generation, full decode, stitching, seam extraction and GPU sampler teardown. Client total additionally includes routing, artifact downloads and client validation; it excludes S3 publication.

Both memory columns use MiB. Despite its name, the engine’s `X-Peak-Memory-MB` header divides bytes by `1024**2` and reports the peak CUDA allocator reserved pool. Device usage comes from the assigned B200’s `nvidia-smi memory.used` sampled every 0.5 seconds; this includes total device residency and can miss shorter spikes. These are separate measurements.

The eight-request batch completed **8/8** videos on **eight distinct replicas**, with **eight overlapping rollout intervals** and **eight overlapping diffusion chunk requests**. These results cover one eight-request batch on the 16-replica deployment; sustained saturation throughput was not measured.

After the timed acceptance, the exact deployed image separately passed the repository B200 checker with Torch 2.11.0+cu130 and native `sm_100` support. Its shipped golden-evaluation command also generated a complete three-chunk rollout, and all four MP4s independently decoded successfully. All 16 serving replicas and their pods remained unchanged. This separate check ran alongside the serving model; its timing and memory are excluded from the production measurements above.

Across 75 successful CPU-head samples taken every 5 seconds, proxy RSS peaked at **214.68 MiB** (private USS **111.47 MiB**) and whole-head cgroup usage peaked at **1639.55 MiB**. The same proxy process and head pod remained present, with zero sampled OOM events. This observation window includes baseline and completion handoff; it is separate from generation latency, and five-second samples can miss brief spikes.

Independent review of all 18 joins across the nine final clips found no obvious hard cuts or scene/vehicle identity resets in the eight-frame contact sheets. Small wheel, reflection and edge changes remain visible around some joins. Static contact sheets do not establish imperceptible motion continuity; the stitch uses direct concatenation without blending.

The private image scan retained 218 HIGH and six unfixed CRITICAL findings, with zero fixable CRITICAL findings and zero detected secrets. The targeted dependency updates remove the base image’s fixable CRITICAL findings. One inherited package metadata mismatch remains: `nixl` requires `nixl-cu13==1.3.0`, while the vendor image installs 1.3.1; the selected TP=1 single-stage diffusion route does not configure NIXL transfer. No new dependency conflicts were introduced.

Exact resource identities, private endpoints, registry coordinates and generated artifacts remain in access-controlled runtime storage. This README includes measurements and generic configuration only.
