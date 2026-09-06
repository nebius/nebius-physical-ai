# Cosmos3-Nano: 30-second diffusion video on NPA mk8s

This deployment runs 16 Ray Serve replicas on 16 B200 GPUs, with one GPU and
tensor parallelism 1 per replica. It supports 30-second text-to-video
continuation and source-conditioned visual augmentation through `vllm serve
--omni` and `Cosmos3OmniDiffusersPipeline`.
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
rounds non-transfer frame counts upward to `4k+1`; 300 would become 301. At 24 fps
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

## Source-conditioned visual augmentation

Continuation preserves a generated tail and extends time. Visual augmentation
instead transforms an existing video's appearance while conditioning every
output interval on the corresponding original frames. The earlier continuation
benchmark is evidence for continuation and concurrency, not augmentation quality.

`nano-video-augment` accepts a real S3 source MP4 at **832×480, 24 fps**, with at
least six frames. It rejects invalid hashes, incomplete decoding, changing
dimensions, incorrect timestamps and unsupported controls before GPU work. Output
duration matches the complete source; no silent resampling or truncation occurs.
The CLI and SDK share one client and the server's request/report contract.

The installed upstream `make_edge_control` computes Canny edges from each
original source interval. Controls are stored as lossless FFV1 videos and must
round-trip pixel-exactly through the actual transfer loader. No extra learned
preprocessor checkpoint is needed. The first interval uses zero RGB conditioning
frames. Later intervals upload the preceding **augmented five-frame tail** for
continuity while using edges from the matching **original** interval for motion.
This is the image's structural transfer path, not continuation prefix conditioning.

With the default 121-frame windows, a 720-frame source uses intervals starting
at frames **0, 116, 232, 348, 464, 580 and 696**. Their lengths are six windows
of 121 frames and one of 24. The final model window is 25 frames; native padding
is trimmed back to 24. Removing the repeated five-frame prefixes yields
`121 + 5×116 + 19 = 720` frames, without blending or interpolation. All seven
requests run sequentially on one selected replica. The service retains the
original 16-replica least-outstanding plus FIFO policy and holds its GPU lock
until accepted generation finishes, including repeated client cancellations.

`--chunk-frames 297` uses three source intervals starting at **0, 292 and 584**,
with lengths **297, 297 and 136**. The last model window is 137 frames and is
trimmed to the source length. Dropping the two five-frame prefixes gives
`297 + 292 + 131 = 720` frames, with joins at frames 297 and 589. Longer windows
reduce the number of independently sampled boundaries; they do not remove the
need to inspect each join or guarantee better motion.

The default transfer parameters are 35 steps, text guidance **3**, control
guidance **1.5**, flow shift **10**, medium Canny thresholds **100/200**, BF16,
TP=1, shared control/target temporal positions, and a 4096-token limit. These
start from the installed single-edge transfer defaults; increasing guidance or
steps is not inherently a quality improvement. Steps and text guidance also
respect the actual video API's upper bounds of 200 and 20. There is no generic
`strength`, `control_weight`, arbitrary server-side file path, or unchecked
extra-parameter bag.

Use a detailed scene description, preferably a structured JSON object covering
subjects, setting, lighting, materials, camera motion and temporal continuity,
following [NVIDIA's transfer guidance](https://github.com/NVIDIA/Cosmos/tree/main/cookbooks/cosmos3/generator/transfer).
The exact submitted and effective positive, negative and system prompts are
retained per interval. The installed formatter rewrites JSON duration to the
integer local model-window duration even when metadata templates are disabled;
the retained source-frame map supplies the exact global timestamps.

```bash
npa workbench cosmos3 nano-video-augment \
  --input-path "$NPA_COSMOS3_AUGMENT_INPUT_URI" \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" \
  --prompt "$NPA_COSMOS3_AUGMENT_PROMPT" \
  --negative-prompt "$NPA_COSMOS3_AUGMENT_NEGATIVE_PROMPT" \
  --seed 42 --num-inference-steps 35 --guidance-scale 3 \
  --control-guidance 1.5 --flow-shift 10 --edge-threshold medium \
  --chunk-frames 121 --output-format json

# Retrieve completed work or retry artifact downloads/publication; never generates again.
npa workbench cosmos3 nano-video-augment-recover \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" --output-format json
```

SDK entries are `npa.sdk.workbench.cosmos3.nano_video_augment` and
`nano_video_augment_recover`. The authenticated API accepts multipart `/run`
with exactly `request` (JSON) and `input_reference` (the complete MP4). Existing
JSON continuation requests remain supported. `GET /result?request_id=...`
retrieves durable state through any replica on the shared output filesystem.
A missing result after an interrupted POST is ambiguous and never authorizes an
automatic generation retry.

Artifacts keep **`input.mp4`**, **`augmented.mp4`** and the synchronized, labeled
**`comparison.mp4`** separate, alongside source controls, augmented RGB tails,
chunk outputs, exact requests, hashes and measurements. The comparison places
the actual source on the left. S3 reservation and publication use conditional
immutable writes with hash-verified readback. Recovery accepts existing objects
only when their bytes match and can finish publication without a serving token
when the completed local result is already verified.
If artifact retrieval fails after generation, recovery uses authenticated GETs
to retrieve the same completed request before validating and publishing it;
the original submission marker remains unchanged.

Technical validity is separate from visual quality. A changed hash does not
prove meaningful augmentation. Evaluate requested appearance change, lighting,
materials/reflections, identity, source motion, wheel contact and every actual join
against a rubric fixed before candidate scoring. Record agent/VLM judgments and
their sampling limitations separately; do not present them as a standardized
quality score or claim eight-way augmentation from the continuation benchmark.

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

After adding augmentation and shared-server cancellation handling, acceptance was rerun on the updated image: one full 30-second continuation video through the SDK, then eight complete continuation requests concurrently through the CLI. All nine clips passed full decoding at 832×480, 24 fps and 720 frames. The two batches published 13 and 97 immutable objects respectively, with read-after-write hash verification. Models were already initialized and BF16 weights prestaged before timing.

The updated-image single-request batch took **155.85 s**; the eight-request batch took **161.23 s**. These batch times include generation, downloads and local validation, and exclude S3 publication. The complete sequential acceptance test case, including both batches and their S3 publication/readback, took **323.87 s**; pytest reported **324.88 s** for the full invocation. Deployment, model initialization and prestaging are excluded.

| Request | Chunk 1 (s) | Chunk 2 (s) | Chunk 3 (s) | Server total (s) | Client total (s) | Peak allocator reserved (MiB) | Peak device used (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 59.89 | 60.04 | 22.46 | 149.99 | 155.82 | 37408.00 | 38316.00 |
| C1 | 59.60 | 60.45 | 22.74 | 150.82 | 155.66 | 38220.00 | 39130.00 |
| C2 | 60.74 | 61.29 | 23.07 | 154.47 | 161.14 | 37408.00 | 38316.00 |
| C3 | 60.12 | 60.68 | 22.98 | 152.85 | 160.05 | 37408.00 | 38316.00 |
| C4 | 59.18 | 59.83 | 22.40 | 150.07 | 154.55 | 37408.00 | 38316.00 |
| C5 | 60.91 | 61.07 | 22.79 | 153.94 | 161.08 | 37408.00 | 38316.00 |
| C6 | 60.00 | 60.62 | 23.01 | 152.24 | 158.89 | 37408.00 | 38316.00 |
| C7 | 60.48 | 60.95 | 22.89 | 153.46 | 161.01 | 37408.00 | 38316.00 |
| C8 | 59.64 | 60.59 | 22.95 | 152.18 | 159.24 | 37408.00 | 38316.00 |

S1 is the single request; C1–C8 belong to the concurrent batch. Chunk time is the synchronous diffusion HTTP round trip inside the replica. Server total includes generation, full decode, stitching, seam extraction and GPU sampler teardown. Client total additionally includes routing, artifact downloads and client validation; it excludes S3 publication.

Both memory columns use MiB. Despite its name, the engine’s `X-Peak-Memory-MB` header divides bytes by `1024**2` and reports the peak CUDA allocator reserved pool. Device usage comes from the assigned B200’s `nvidia-smi memory.used` sampled every 0.5 seconds; this includes total device residency and can miss shorter spikes. These are separate measurements.

The eight-request batch completed **8/8** videos on **eight distinct replicas**, with **eight overlapping rollout intervals** and **eight overlapping diffusion chunk requests**. These results cover one eight-request batch on the 16-replica deployment; sustained saturation throughput was not measured.

After the timed acceptance, the exact deployed image separately passed the repository B200 checker with Torch 2.11.0+cu130 and native `sm_100` support. Its shipped golden-evaluation command also generated a complete three-chunk rollout, and all four MP4s independently decoded successfully. All 16 serving replicas and their pods remained unchanged. This separate check ran alongside the serving model; its timing and memory are excluded from the production measurements above.

The original pre-augmentation acceptance also remains valid: its single and eight-request batches took **153.61 s** and **163.04 s**, respectively, and all nine videos and 110 published objects passed validation. Those measurements use the earlier image.

During that original run, across 75 successful CPU-head samples taken every 5 seconds, proxy RSS peaked at **214.68 MiB** (private USS **111.47 MiB**) and whole-head cgroup usage peaked at **1639.55 MiB**. The same proxy process and head pod remained present, with zero sampled OOM events. This observation window includes baseline and completion handoff; it is separate from generation latency, and five-second samples can miss brief spikes.

Separate independent reviews of the original nine clips and all 18 joins in the updated-image nine-clip acceptance found no obvious hard cuts or scene/vehicle identity resets in the eight-frame contact sheets. Small wheel, reflection and edge changes remain visible around some joins. Static contact sheets do not establish imperceptible motion continuity; the stitch uses direct concatenation without blending.

The private image scan retained 218 HIGH and six unfixed CRITICAL findings, with zero fixable CRITICAL findings and zero detected secrets. The targeted dependency updates remove the base image’s fixable CRITICAL findings. One inherited package metadata mismatch remains: `nixl` requires `nixl-cu13==1.3.0`, while the vendor image installs 1.3.1; the selected TP=1 single-stage diffusion route does not configure NIXL transfer. No new dependency conflicts were introduced.

Exact resource identities, private endpoints, registry coordinates and generated artifacts remain in access-controlled runtime storage. This README includes measurements and generic configuration only.

## Measured full-source augmentation

Eleven complete variants were generated from the same task-generated warehouse
robot video. The selected 30-second result changes the bright clean scene into
a dimmer warehouse with warm overhead illumination, rough dark damp concrete,
localized shallow puddles and restrained reflections. The original orange robot,
aisle geometry, camera travel and scene timing remain recognizable. The retained
comparison labels the **actual source on the left** and **augmentation on the
right**, correcting the earlier source/output labeling confusion.

The selected run used **35 steps, text guidance 5, control guidance 1, flow shift
10, high Canny thresholds 200/300, 297-frame windows, seed 480240 and 4096 tokens**.
BF16, TP=1, shared control/target temporal positions, audio off and guardrails off
remain as described above. Its structured positive prompt specifies matte orange
paint, transparent shallow water over rough concrete, warm-white practical lamps,
restrained haze and source-matched travel. The structured negative prompt targets
glowing orange floor smears, trench-like lane markings, plastic materials and
motion/contact defects. Exact submitted and effective prompts are retained with
the private artifacts. Increasing steps to 50 or guidance did not consistently
improve realism; lower Canny thresholds did not improve wet-floor realism in
the same 297-frame configuration.

| Source interval, zero-based inclusive | Preparation (s) | Diffusion HTTP (s) | Peak allocator reserved (MiB) | Peak device used (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 0–296 | 24.84 | 167.09 | 38012 | 38918 |
| 292–588 | 6.85 | 165.86 | 38012 | 38918 |
| 584–719 | 6.51 | 53.08 | 38012 | 38920 |

Server wall time was **440.21 s**, including control preparation, generation,
validation and stitching. The client submission/download/validation interval was
**450.51 s**, excluding source S3 setup and output publication. The complete live
test case, including source setup, immutable publication/readback and recovery
verification, took **455.38 s**; JUnit reports **456.05 s** for the pytest suite. The
selected request overlapped two other augmentation requests on distinct existing
replicas for part of its execution; this is not an isolated throughput benchmark.
The memory sampling limits above apply. Input and output each fully decoded to
**720 frames, 24 fps, 832×480 and 30.000 s**; the synchronized comparison is
1664×480 with the same frame/timestamp contract.

The rubric was frozen before generation. Independent agent review produced the
following task-specific ordinal judgments; the hosted VLM results use a separate
0–1 scale and are retained without forcing agreement.

| Component | Agent (0–4) | Hosted VLM (0–1) |
| --- | ---: | ---: |
| Requested environmental change | 3 | 0.75 |
| Lighting realism | 3 | 0.50 |
| Materials and reflections | 3 | 0.50 |
| Robot identity | 3 | 0.75 |
| Source geometry, camera and timing | 4 | 0.75 |
| Motion and floor contact | 3 | 0.50–0.75 |
| Temporal continuity | 3 | 0.50 |

The motion rating initially was 2. A separately retained review of unmodified
native tire/contact crops supported qualitative source-following travel and
stable floor contact, yielding 3 with moderate confidence under the unchanged
anchors. The original review remains preserved. Exact tire angle, speed and
no-slip physics remain unverified: repetitive dark tread and hub covers provide
weak angular cues in both source and output. Automated CPU estimates did not
establish rolling fidelity. Complete browser playback decoded all 720 frames
without dropped frames; agent review also inspected dense sequences across both
joins. These observations are not a human rating or a standardized aggregate
quality score. The actual hosted VLM was MiniMax-M3; its 17 sampled-frame calls
sometimes inferred freezing from weak motion cues, reversed front/rear wheel
descriptions or used inconsistent verbal anchors, so its scores cannot certify
physical motion.

Remaining imperfections include weaker puddles and robot reflections late in the
clip, small hub/tire-detail changes and occasional rectangular floor sheen. The
result is a noticeable, plausible augmentation with these limits. It does not
establish an eight-way augmentation batch; the separate single-plus-eight test
above measures continuation.

To reuse the selected settings, provide a new immutable destination and the
desired structured positive, negative and system prompts through the environment:

```bash
npa workbench cosmos3 nano-video-augment \
  --input-path "$NPA_COSMOS3_AUGMENT_INPUT_URI" \
  --output-path "$NPA_COSMOS3_AUGMENT_OUTPUT_URI" \
  --prompt "$NPA_COSMOS3_AUGMENT_PROMPT" \
  --negative-prompt "$NPA_COSMOS3_AUGMENT_NEGATIVE_PROMPT" \
  --system-prompt "$NPA_COSMOS3_AUGMENT_SYSTEM_PROMPT" \
  --seed "${NPA_COSMOS3_AUGMENT_SEED:-480240}" \
  --num-inference-steps 35 --guidance-scale 5 --control-guidance 1 \
  --flow-shift 10 --edge-threshold high --chunk-frames 297 \
  --max-sequence-length 4096 --output-format json
```
